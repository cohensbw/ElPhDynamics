module SingleSitePreconditioners

using  UnsafeArrays
using  LinearAlgebra
import LinearAlgebra: ldiv!, mul!, transpose

using ..HolsteinModels: HolsteinModel
using ..Checkerboard: checkerboard_mul!, checkerboard_inverse_mul!, checkerboard_transpose_mul!, checkerboard_inverse_transpose_mul!
using ..TimeFreqFFTs: TimeFreqFFT, τ_to_ω!, ω_to_τ!
using ..Utilities: get_index

export LeftSingleSitePreconditioner, RightSingleSitePreconditioner, setup!

abstract type SingleSitePreconditioner end

mutable struct LeftSingleSitePreconditioner{T1<:AbstractFloat,T2<:Number} <: SingleSitePreconditioner

    "holstein model."
    holstein::HolsteinModel{T1,T2}

    "TimeFreqFFT object for mapping between τ ⟷ ω"
    timefreqfft::TimeFreqFFT{T1}

    "regularizaiton parameter"
    η::Vector{T1}

    "exp{-i⋅Δτ⋅η}"
    expniΔτη::Vector{Complex{T1}}

    "exp{-i⋅Δτ⋅ϕ(ω)} where Δτ⋅ϕ(ω)=(2π/L)⋅(ω+1/2) or ϕ(ω)=(2π/β)⋅(ω+1/2)"
    expniΔτϕ::Vector{Complex{T1}}

    "M₀⁻¹[ω,ω] = G₀[ω,ω]"
    G0::Vector{Complex{T1}}

    "temporary storage vector"
    vtemp::Vector{T2}

    "temporary vector needed for perturbative expansion"
    zin::Vector{Complex{T1}}

    "temporary vector needed for perturbative expansion"
    zout::Vector{Complex{T1}}

    "temporary vector need when multiplying vector by Γ"
    zΓ::Vector{Complex{T1}}

    function LeftSingleSitePreconditioner(holstein::HolsteinModel{T1,T2};ηmax::T1=0.0,ηpower::T1=0.0) where {T1<:AbstractFloat,T2<:Number}

        N  = holstein.nsites
        L  = holstein.Lτ
        NL = N*L
        Δτ = holstein.Δτ

        timefreqfft = TimeFreqFFT(holstein.lattice,L)
        G0          = zeros(Complex{T1},NL)
        vtemp       = zeros(T2,NL)
        zin         = zeros(Complex{T1},NL)
        zout        = zeros(Complex{T1},NL)
        zΓ          = zeros(Complex{T1},NL)

        η = zeros(T1,L)
        for ω in 1:cld(L,2)
            η[ω]     = ηmax/ω^ηpower
            η[L-ω+1] = -η[ω]
        end

        expniΔτη = zeros(Complex{T1},NL)
        for ω in 1:L
            expniΔτηω = exp(-im*Δτ*η[ω])
            for i in 1:N
                expniΔτη[get_index(ω,i,L)] = expniΔτηω
            end
        end

        expniΔτϕ = zeros(Complex{T1},NL)
        for ω in 1:L
            expniΔτϕω = exp(im*2*π/L*((ω-1)+1/2))
            for i in 1:N
                expniΔτϕ[get_index(ω,i,L)] = expniΔτϕω
            end
        end

        return new{T1,T2}(holstein,timefreqfft,η,expniΔτη,expniΔτϕ,G0,vtemp,zin,zout,zΓ)
    end
end


mutable struct RightSingleSitePreconditioner{T1<:AbstractFloat,T2<:Number} <: SingleSitePreconditioner

    "holstein model."
    holstein::HolsteinModel{T1,T2}

    "TimeFreqFFT object for mapping between τ ⟷ ω"
    timefreqfft::TimeFreqFFT{T1}

    "regularizaiton parameter"
    η::T1

    "exp{-i⋅Δτ⋅η}"
    expniΔτη::Complex{T1}

    "exp{-i⋅Δτ⋅ϕ(ω)} where Δτ⋅ϕ(ω)=(2π/L)⋅(ω+1/2) or ϕ(ω)=(2π/β)⋅(ω+1/2)"
    expniΔτϕ::Vector{Complex{T1}}

    "M₀⁻¹[ω,ω] = G₀[ω,ω]"
    G0::Vector{Complex{T1}}

    "temporary vector needed for perturbative expansion"
    zin::Vector{Complex{T1}}

    "temporary vector needed for perturbative expansion"
    zout::Vector{Complex{T1}}

    "temporary vector need when multiplying vector by Γ"
    zΓ::Vector{Complex{T1}}

    function RightSingleSitePreconditioner(op::LeftSingleSitePreconditioner{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

        return new{T1,T2}(op.holstein,op.timefreqfft,op.η,op.expniΔτη,op.expniΔτϕ,op.G0,op.zin,op.zout,op.zΓ)
    end
end

"""
Transpose LeftSingleSitePreconditioner to construct and return a RightSingleSitePreconditioner
"""
function transpose(op::LeftSingleSitePreconditioner)

    return RightSingleSitePreconditioner(op)
end


"""
Calculate the diagonal matrix G₀[ω] in the single site limit.
"""
function setup!(op::SingleSitePreconditioner)

    expnΔτV = op.holstein.expnΔτV
    N  = op.holstein.nsites
    L  = op.holstein.Lτ
    NL = N*L

    @fastmath @inbounds for i in 1:N
        # calculating matrix element of diagonal matrix exp{-Δτ⋅V}_bar = (1/L)∑exp{-Δτ⋅V(τ)}
        expnΔτVi = 0.0
        for τ in 1:L
            expnΔτVi += expnΔτV[get_index(τ,i,L)]
        end
        expnΔτVi /= L
        # Calculating Green's function in single-site limit.
        # G₀[ω] = [1 - exp{-Δτ⋅( V + iϕ(ω) + iη )}]⁻¹
        for ω in 1:L
            indx = get_index(ω,i,L)
            op.G0[indx] = 1.0 / (1.0 - expnΔτVi * op.expniΔτϕ[ω] * op.expniΔτη[indx])
        end
    end

    return nothing
end

function setup!(op)

    return nothing
end


"""
Apply preconditioner.
"""
function ldiv!(vout::AbstractVector{T},op::SingleSitePreconditioner,vin::AbstractVector{T}) where {T<:Number}

    mul_invM0!(vout, op, vin)          # M₀⁻¹⋅v
    mul!(op.vtemp, op.holstein, vout)  # M⋅M₀⁻¹⋅v
    mul_invM0!(op.vtemp, op)           # M₀⁻¹⋅M⋅M₀⁻¹⋅v
    @. vout = 2*vout - op.vtemp        # (2⋅M₀⁻¹-M₀⁻¹⋅M⋅M₀⁻¹)⋅v

    return nothing
end

function ldiv!(op::SingleSitePreconditioner,v::AbstractVector{T}) where {T<:Number}

    mul_invM0!(v, op)              # M₀⁻¹⋅v
    mul!(op.vtemp, op.holstein, v) # M⋅M₀⁻¹⋅v
    mul_invM0!(op.vtemp, op)       # M₀⁻¹⋅M⋅M₀⁻¹⋅v
    @. v = 2*v - op.vtemp          # (2⋅M₀⁻¹-M₀⁻¹⋅M⋅M₀⁻¹)⋅v
    
    return nothing
end


"""
Multiply by M₀⁻¹[ω,ω]⋅v where M₀⁻¹[ω,ω] is an approximation of M⁻¹[ω,ω] arrived at by
perturbing away from the K=0 single-site limit.
"""
function mul_invM0!(vout::AbstractVector{T},op::LeftSingleSitePreconditioner,vin::AbstractVector{T}) where {T<:Number}

    τ_to_ω!(op.zin,op.timefreqfft,vin)

    # G₀ = [1 - exp{-Δτ⋅(V+iϕ+iη)}]⁻¹
    # Γ  = exp{+Δτ⋅(K-iη)}-1

    mul_G0!(op.zout, op, op.zin) # G₀⋅zin
    mul_Γ!(op.zin, op, op.zout)  # ΓG₀⋅zin
    mul_G0!(op.zin, op)          # G₀ΓG₀⋅zin
    axpy!(-1, op.zin, op.zout)   # (G₀-G₀ΓG₀)⋅zin
    mul_Γp1!(op.zout, op)        # zout = (Γ+1)⋅(G₀-G₀ΓG₀)⋅zin

    ω_to_τ!(vout,op.timefreqfft,op.zout)

    return nothing
end

"""
Multiply by M₀⁻ᵀ[ω,ω]⋅v where M₀⁻ᵀ[ω,ω] is an approximation of M⁻ᵀ[ω,ω] arrived at by
perturbing away from the K=0 single-site limit.
"""
function mul_invM0!(vout::AbstractVector{T},op::RightSingleSitePreconditioner,vin::AbstractVector{T}) where {T<:Number}

    τ_to_ω!(op.zin,op.timefreqfft,vin)

    # G₀ = [1 - exp{-Δτ⋅(V+iϕ+iη)}]⁻¹
    # Γ  = exp{+Δτ⋅(K-iη)}-1

    mul_Γp1!(op.zout, op, op.zin) # (Γ+1)ᵀ⋅zin
    mul_G0!(op.zout, op)          # G₀ᵀ⋅(Γ+1)ᵀ⋅zin
    mul_Γ!(op.zin, op, op.zout)   # ΓᵀG₀ᵀ⋅(Γ+1)ᵀ⋅zin
    mul_G0!(op.zin, op)           # G₀ᵀΓᵀG₀ᵀ⋅(Γ+1)ᵀ⋅zin
    axpy!(-1.0,op.zin,op.zout)    # zout = (G₀ᵀ-G₀ᵀΓᵀG₀ᵀ)⋅(Γ+1)ᵀ⋅zin

    ω_to_τ!(vout,op.timefreqfft,op.zout)

    return nothing
end

function mul_invM0!(v::AbstractVector{T},op::SingleSitePreconditioner) where {T<:Number}

    mul_invM0!(v,op,v)
    return nothing
end


"""
Multiply a vector by G₀[ω] diagonal matrix.
"""
function mul_G0!(vout::AbstractVector{T},op::LeftSingleSitePreconditioner,vin::AbstractVector{T}) where {T<:Complex}

    @. vout = op.G0 * vin
    return nothing
end

"""
Multiply a vector by G₀ᵀ[ω] diagonal matrix.
"""
function mul_G0!(vout::AbstractVector{T},op::RightSingleSitePreconditioner,vin::AbstractVector{T}) where {T<:Complex}

    @. vout = conj(op.G0) * vin
    return nothing
end

function mul_G0!(v::AbstractVector{T},op::SingleSitePreconditioner) where {T<:Complex}

    mul_G0!(v,op,v)
    return nothing
end


"""
Multiply vector by Γ+1=exp{+Δτ⋅(K-iη)}
"""
function mul_Γp1!(v::AbstractVector{T},op::LeftSingleSitePreconditioner) where {T<:Complex}

    neighbor_table_tij = op.holstein.neighbor_table_tij
    coshtij            = op.holstein.coshtij
    sinhtij            = op.holstein.sinhtij

    checkerboard_inverse_mul!(v, neighbor_table_tij, coshtij, sinhtij)
    @. v *= op.expniΔτη

    return nothing
end

"""
Multiply vector by (Γ+1)ᵀ=exp{+Δτ⋅(K-iη)}ᵀ
"""
function mul_Γp1!(v::AbstractVector{T},op::RightSingleSitePreconditioner) where {T<:Complex}

    neighbor_table_tij = op.holstein.neighbor_table_tij
    coshtij            = op.holstein.coshtij
    sinhtij            = op.holstein.sinhtij

    checkerboard_inverse_transpose_mul!(v, neighbor_table_tij, coshtij, sinhtij)
    @. v *= conj(op.expniΔτη)

    return nothing
end

function mul_Γp1!(vout::AbstractVector{T},op::SingleSitePreconditioner,vin::AbstractVector{T}) where {T<:Complex}

    copyto!(vout,vin)
    mul_Γp1!(vout,op)

    return nothing
end


"""
Multiply vector by Γ=exp{+Δτ⋅(K-iη)}-1
"""
function mul_Γ!(vout::AbstractVector{T},op::LeftSingleSitePreconditioner,vin::AbstractVector{T}) where {T<:Complex}

    neighbor_table_tij = op.holstein.neighbor_table_tij
    coshtij            = op.holstein.coshtij
    sinhtij            = op.holstein.sinhtij

    @. vout = op.expniΔτη * vin
    checkerboard_inverse_mul!(vout, neighbor_table_tij, coshtij, sinhtij)
    @. vout -= vin

    return nothing
end

"""
Multiply vector by Γᵀ=exp{+Δτ⋅(K-iη)}ᵀ-1
"""
function mul_Γ!(vout::AbstractVector{T},op::RightSingleSitePreconditioner,vin::AbstractVector{T}) where {T<:Complex}
    neighbor_table_tij = op.holstein.neighbor_table_tij
    coshtij            = op.holstein.coshtij
    sinhtij            = op.holstein.sinhtij

    @. vout = conj(op.expniΔτη) * vin
    checkerboard_inverse_transpose_mul!(vout, neighbor_table_tij, coshtij, sinhtij)
    @. vout -= vin

    return nothing
end

function mul_Γ!(v::AbstractVector{T},op::SingleSitePreconditioner) where {T<:Complex}

    copyto!(op.zΓ,v)
    mul_Γ!(v,op,op.zΓ)
    return nothing
end

end