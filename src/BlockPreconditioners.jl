module BlockPreconditioners

using  UnsafeArrays
using  LinearAlgebra
import LinearAlgebra: mul!, ldiv!, transpose

using ..HolsteinModels: HolsteinModel
using ..Checkerboard:   checkerboard_mul!, checkerboard_transpose_mul!
using ..TimeFreqFFTs:   TimeFreqFFT, τ_to_ω!, ω_to_τ!
using ..Utilities: get_index

using ..RestartedGMRES: GMRES
import ..RestartedGMRES

using ..ConjugateGradients: ConjugateGradient
import ..ConjugateGradients

export LeftBlockPreconditioner, RightBlockPreconditioner, SquaredBlockPreconditioner, setup!

abstract type BlockPreconditioner end

mutable struct LeftBlockPreconditioner{T1<:AbstractFloat,T2<:Number} <: BlockPreconditioner

    "holstein model."
    holstein::HolsteinModel{T1,T2}

    "TimeFreqFFT object for mapping between τ ⟷ ω"
    timefreqfft::TimeFreqFFT{T1}

    "GMRES type."
    gmres::GMRES{T1,Complex{T1}}

    "Represents (1/L)⋅∑[exp(-Δτ⋅λ⋅x(τ))]"
    expnΔτV_bar::Vector{T1}

    "Array of complex phase factor need to multiply by M[ω,ω] matrices."
    ω_phases::Vector{Complex{T1}}

    "Frequency of M[ω,ω] matrix that `mul!` will reference."
    ω::Int

    "Temporary storage vector of length NL."
    z1::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z2::Vector{Complex{T1}}

    function LeftBlockPreconditioner(holstein::HolsteinModel{T1,T2};
                                     tol::T1=1e-1, maxiter::Int=-1, restart::Int=-1) where {T1<:AbstractFloat,T2<:Number}

        N  = holstein.nsites
        L  = holstein.Lτ
        ω  = 1

        timefreqfft = TimeFreqFFT(holstein.lattice,L)
        ω_phases    = [exp(2*π*im*((ω-1)+1/2)/L) for ω = 1:L]
        expnΔτV_bar = zeros(T1,N)
        z1          = zeros(Complex{T1},N*L)
        z2          = zeros(Complex{T1},N*L)
        utemp       = zeros(Complex{T1},N)
        gmres       = GMRES(utemp,tol=tol,maxiter=maxiter,restart=restart)

        return new{T1,T2}(holstein,timefreqfft,gmres,expnΔτV_bar,ω_phases,ω,z1,z2)
    end
end


mutable struct RightBlockPreconditioner{T1<:AbstractFloat,T2<:Number} <: BlockPreconditioner

    "holstein model."
    holstein::HolsteinModel{T1,T2}

    "TimeFreqFFT object for mapping between τ ⟷ ω"
    timefreqfft::TimeFreqFFT{T1}

    "GMRES type."
    gmres::GMRES{T1,Complex{T1}}

    "Represents (1/L)⋅∑[exp(-Δτ⋅x(τ))]"
    expnΔτV_bar::Vector{T1}

    "Array of complex phase factor need to multiply by M[ω,ω] matrices."
    ω_phases::Vector{Complex{T1}}

    "Frequency of M[ω,ω] matrix that `mul!` will reference."
    ω::Int

    "Temporary storage vector of length NL."
    z1::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z2::Vector{Complex{T1}}

    function RightBlockPreconditioner(op::LeftBlockPreconditioner{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

        return new{T1,T2}(op.holstein,op.timefreqfft,op.gmres,op.expnΔτV_bar,op.ω_phases,op.ω,op.z1,op.z2)
    end
end


"""
Update expnΔτV_bar based on the current phonon field configuration.
"""
function setup!(op::BlockPreconditioner)

    # calucate expnΔτV_bar
    N = op.holstein.nsites
    L = op.holstein.Lτ
    @fastmath @inbounds for i in 1:N
        op.expnΔτV_bar[i] = 0.0
        for τ in 1:L
            op.expnΔτV_bar[i] += op.holstein.expnΔτV[get_index(τ,i,L)]
        end
        op.expnΔτV_bar[i] /= L
    end

    return nothing
end

function setup!(op)

    # this is an intert version of the setup! function so it can be called on
    # other preconditioners and not do anything.
    return nothing
end


"""
Apply Preconditioner.
"""
function ldiv!(vout::AbstractVector{T},op::BlockPreconditioner,vin::AbstractVector{T}) where {T<:AbstractFloat}

    N  = op.holstein.nsites
    L  = op.holstein.Lτ
    z1 = op.z1
    z2 = op.z2

    flag  = 0
    iters = 0
    Δ     = 0.0

    # 1. apply phase factor to go from (anit-periodic)⟶(periodic) in τ
    # 2. FFT from τ ⟶ ω
    τ_to_ω!(z2,op.timefreqfft,vin)

    @uviews z1 z2 begin

        a1  = reshape(z1,(L,N))
        a1T = reshape(z1,(N,L))
        a2  = reshape(z2,(L,N))
        a2T = reshape(z2,(N,L))

        transpose!(a1T,a2)
        fill!(z2,0.0)

        # iterating over half the range of frequencies
        for ω in 1:cld(L,2)

            # Solve M[ω,ω]⋅x  = b ==> x = M⁻¹[ω,ω]⋅b OR
            # Solve Mᵀ[ω,ω]⋅x = b ==> x = M⁻ᵀ[ω,ω]⋅b
            op.ω = ω
            b    = @view a1T[:,ω]
            x    = @view a2T[:,ω]
            flag, iters, Δ = RestartedGMRES.solve!(x,op,b,op.gmres,I)
            # println(ω," ",iters)

            # accounting for symmetry
            for i in 1:N
                a2T[i,L-ω+1] = conj(a2T[i,ω])
            end
        end

        transpose!(a1,a2T)
    end

    # 1. iFFT from ω ⟶ τ
    # 2. apply inverse phase factor to go from (periodic)⟶(anti-periodic) in τ
    ω_to_τ!(vout,op.timefreqfft,z1)

    return nothing
end

function ldiv!(op::BlockPreconditioner,v::AbstractVector)

    ldiv!(v,op,v)
    return nothing
end


"""
Multiply a length N vector v living in frequency ω space by the
M[ω,ω] NxN block diagonal matrix.
"""
function mul!(vout::AbstractVector{T},op::LeftBlockPreconditioner,vin::AbstractVector{T}) where {T<:Complex}

    # copy vector
    copyto!(vout,vin)

    # multiply by checkerboard matrix
    checkerboard_mul!(vout, op.holstein.neighbor_table_tij, op.holstein.coshtij, op.holstein.sinhtij)

    # multiply by average exponentiated interaction matrix
    # and a phase factor associated with current ω value
    @. vout = op.ω_phases[op.ω] * op.expnΔτV_bar * vout

    # get final result
    @. vout = vin - vout

    return nothing
end


"""
Multiply a length N vector v living in frequency ω space by the
Mᵀ[ω,ω] NxN block diagonal matrix.
"""
function mul!(vout::AbstractVector{T},op::RightBlockPreconditioner,vin::AbstractVector{T}) where {T<:Complex}

    # multiply by interaction matrix
    @. vout = conj(op.expnΔτV_bar) * vin

    # multiply by checkerboard matrix
    checkerboard_transpose_mul!(vout, op.holstein.neighbor_table_tij, op.holstein.coshtij, op.holstein.sinhtij)

    # multiply by phase
    @. vout = conj(op.ω_phases[op.ω]) * vout

    # get final result
    @. vout = vin - vout

    return nothing
end


"""
Transpose from LeftBlockPreconditioner to RightBlockPreconditioner.
"""
function transpose(op::LeftBlockPreconditioner)::RightBlockPreconditioner

    return RightBlockPreconditioner(op)
end

end