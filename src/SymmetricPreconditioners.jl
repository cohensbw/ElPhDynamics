module SymmetricPreconditioners

using  UnsafeArrays
using  LinearAlgebra
import LinearAlgebra: ldiv!, mul!, transpose

using ..HolsteinModels: HolsteinModel
using ..Checkerboard:   checkerboard_mul!, checkerboard_transpose_mul!
using ..TimeFreqFFTs:   TimeFreqFFT, τ_to_ω!, ω_to_τ!
using ..Utilities:      get_index

export SymmetricPreconditioner, setup!

mutable struct SymmetricPreconditioner{T1<:AbstractFloat,T2<:Number}

    "holstein model."
    holstein::HolsteinModel{T1,T2}

    "TimeFreqFFT object for mapping between τ ⟷ ω"
    timefreqfft::TimeFreqFFT{T1}

    "Represents (1/L)⋅∑[exp(-Δτ⋅V(τ))]"
    expnΔτV_bar::Vector{T2}

    "Represents (1/L)⋅∑[exp(-2⋅Δτ⋅V(τ))]"
    expn2ΔτV_bar::Vector{T2}

    "Array of complex phase factors needed."
    expniϕ::Vector{Complex{T1}}

    """
    A₀[ω,ω] = M₀ᵀM₀[ω,ω] where M₀ is the M matrix evaluated in the single-site limit.
    """
    A0ω::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z1::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z2::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z3::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z4::Vector{Complex{T1}}

    function SymmetricPreconditioner(holstein::HolsteinModel{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

        N  = holstein.nsites
        L  = holstein.Lτ
        NL = N*L

        timefreqfft  = TimeFreqFFT(holstein.lattice,L)
        expnΔτV_bar  = zeros(T1,NL)
        expn2ΔτV_bar = zeros(T1,NL)
        expniϕ       = zeros(Complex{T1},NL)
        A0ω          = zeros(Complex{T1},NL)
        z1           = zeros(Complex{T1},NL)
        z2           = zeros(Complex{T1},NL)
        z3           = zeros(Complex{T1},NL)
        z4           = zeros(Complex{T1},NL)

        for ω in 1:L
            expniϕ_ω = exp(-im*2*π/L*(ω+1/2))
            for i in 1:N
                expniϕ[get_index(ω,i,L)] = expniϕ_ω
            end
        end

        return new{T1,T2}(holstein, timefreqfft, expnΔτV_bar, expn2ΔτV_bar, expniϕ, A0ω, z1, z2, z3, z4)
    end
end


function setup!(op::SymmetricPreconditioner)

    N       = op.holstein.nsites
    L       = op.holstein.Lτ
    expnΔτV = op.holstein.expnΔτV

    # calcualte (1/L)⋅∑[exp(-Δτ⋅V(τ))] and (1/L)⋅∑[exp(-2⋅Δτ⋅V(τ))]
    @fastmath @inbounds for i in 1:N
        expnΔτVi  = 0.0
        expn2ΔτVi = 0.0
        for τ in 1:L
            indx       = get_index(τ,i,L)
            expnΔτVi  += expnΔτV[indx]
            expn2ΔτVi += expnΔτV[indx]*expnΔτV[indx]
        end
        expnΔτVi  /= L
        expn2ΔτVi /= L
        for ω in 1:L
            indx = get_index(ω,i,L)
            op.expnΔτV_bar[indx]  = expnΔτVi
            op.expn2ΔτV_bar[indx] = expn2ΔτVi
        end
    end

    # calculate A₀[ω,ω] = M₀ᵀM₀[ω,ω]
    @. op.A0ω = 1.0 - op.expniϕ*op.expnΔτV_bar - conj(op.expniϕ)*op.expnΔτV_bar + op.expn2ΔτV_bar

    return nothing
end


"""
Apply symmetric diagonal preconditioner.
"""
function ldiv!(vout::AbstractVector{T},op::SymmetricPreconditioner,vin::AbstractVector{T}) where {T<:AbstractFloat}

    N  = op.holstein.nsites
    L  = op.holstein.Lτ
    z1 = op.z1
    z2 = op.z2

    # 1. apply phase factor to go from (anit-periodic)⟶(periodic) in τ
    # 2. FFT from τ ⟶ ω
    τ_to_ω!(z1,op.timefreqfft,vin)

    # calculate [2⋅P⁻¹-P⁻¹⋅A[ω,ω]⋅P⁻¹]⋅z where P=M₀ᵀM₀[ω,ω] is a diagonal matrix
    # and A[ω,ω]=MᵀM[ω,ω].
    @. z1 /= op.A0ω
    mul!(z2,op,z1)
    @. z2 /= op.A0ω
    @. z1  = 2*z1 - z2

    # 1. iFFT from ω ⟶ τ
    # 2. apply inverse phase factor to go from (periodic)⟶(anti-periodic) in τ
    ω_to_τ!(vout,op.timefreqfft,z1)

    return nothing
end


"""
Multiply by A[ω,ω]=MᵀM[ω,ω].
"""
function mul!(zout::AbstractVector{T},op::SymmetricPreconditioner,zin::AbstractVector{T}) where {T<:Complex}

    neighbor_table_tij = op.holstein.neighbor_table_tij
    coshtij            = op.holstein.coshtij
    sinhtij            = op.holstein.sinhtij

    # z3 = exp(-ΔτK) ⋅ zin
    copyto!(op.z3,zin)
    checkerboard_mul!(op.z3, neighbor_table_tij, coshtij, sinhtij)

    # zout = exp(-ΔτK)ᵀ ⋅ exp(-2ΔτV)_bar ⋅ exp(-ΔτK) ⋅ zin
    @. zout = op.expn2ΔτV_bar * op.z3
    checkerboard_transpose_mul!(zout, neighbor_table_tij, coshtij, sinhtij)

    # z3 = exp(-iϕ) ⋅ exp(-ΔτV)_bar ⋅ exp(-ΔτK) ⋅ zin
    @. op.z3 *= op.expniϕ * op.expnΔτV_bar

    # z4 = exp(+iϕ) ⋅ exp(-ΔτK)ᵀ ⋅ exp(-ΔτV)_bar ⋅ zin
    @. op.z4  = op.expnΔτV_bar * zin
    checkerboard_transpose_mul!(op.z4, neighbor_table_tij, coshtij, sinhtij)
    @. op.z4 *= conj(op.expniϕ)

    # zout = [I - exp(-iϕ)⋅exp(-ΔτV)_bar⋅exp(-ΔτK) - exp(+iϕ)⋅exp(-ΔτK)ᵀ⋅exp(-ΔτV)_bar
    #         + exp(-ΔτK)ᵀ⋅exp(-2ΔτV)_bar⋅exp(-ΔτK)] ⋅ zin
    @. zout = zin - op.z3 - op.z4 + zout

    return nothing
end

end