module BlockPreconditioners

using  IterativeSolvers
using  UnsafeArrays
using  LinearAlgebra
import LinearAlgebra: mul!, ldiv!

using ..HolsteinModels: HolsteinModel
using ..Checkerboard:   checkerboard_mul!
using ..TimeFreqFFTs:   TimeFreqFFT, τ_to_ω!, ω_to_τ!
using ..RestartedGMRES: GMRES, solve!
using ..Utilities: get_index

mutable struct BlockPreconditioner{T1<:AbstractFloat,T2<:Number}

    "holstein model."
    holstein::HolsteinModel{T1,T2}

    "TimeFreqFFT object for mapping between τ ⟷ ω"
    timefreqfft::TimeFreqFFT{T1}

    "GMRES type."
    rgmres::GMRES{Complex{T1},T1}

    "Represents (1/L)⋅∑[exp(-Δτ⋅ϕ(τ))]"
    expnΔτV_bar::Vector{T1}

    "Array of complex phase factor need to multiply by M[ω,ω] matrices."
    ω_phases :: Vector{Complex{T1}}

    "Frequency of M[ω,ω] matrix that `mul!` will reference."
    ω::Int

    "Temporary storage vector of length NL."
    z1::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z2::Vector{Complex{T1}}

    function BlockPreconditioner(holstein::HolsteinModel{T1,T2}; tol::T1=1e-3, maxiter::Int=-1, restart::Int=-1) where {T1<:AbstractFloat,T2<:Number}

        N             = holstein.nsites
        L             = holstein.Lτ
        ω             = 1
        timefreqfft   = TimeFreqFFT(holstein.lattice,L)
        expnΔτV_bar   = zeros(T1,N)
        ω_phases      = [exp(2*π*im*((ω-1)+1/2)/L) for ω = 1:L]
        z1            = zeros(Complex{T1},N*L)
        z2            = zeros(Complex{T1},N*L)
        utemp         = zeros(Complex{T1},N)
        rgmres        = GMRES(utemp,tol=tol,maxiter=maxiter,restart=restart)

        return new{T1,T2}(holstein,timefreqfft,rgmres,expnΔτV_bar,ω_phases,ω,z1,z2)
    end
end


"""
Update expnΔτV_bar based on the current phonon field configuration.
"""
function setup!(op::BlockPreconditioner)

    N       = op.holstein.nsites
    L       = op.holstein.Lτ
    expnΔτV = op.holstein.expnΔτV
    @fastmath @inbounds for i in 1:N
        op.expnΔτV_bar[i] = 0.0
        for τ in 1:L
            op.expnΔτV_bar[i] += expnΔτV[get_index(τ,i,L)]
        end
        op.expnΔτV_bar[i] /= L
    end
    return nothing
end


function ldiv!(vout::AbstractVector{T},op::BlockPreconditioner,vin::AbstractVector{T}) where {T<:AbstractFloat}

    N  = op.holstein.nsites
    L  = op.holstein.Lτ
    z1 = op.z1
    z2 = op.z2

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

            # solving M[ω,ω]⋅x = b ==> x = M⁻¹[ω,ω]⋅b
            op.ω = ω
            b    = @view a1T[:,ω]
            x    = @view a2T[:,ω]
            # flag, iters, Δ = solve!(x,op,b,op.rgmres,I)
            history = IterativeSolvers.gmres!(x, op, b, tol=1e-3, restart=5, maxiter=2*N, log=true)

            # accounting for symmetry
            for i in 1:N
                a2T[i,L-ω+1] = conj(a2T[i,ω])
            end
        end

        transpose!(a1,a2T)
    end

    # 1. iFFT from ω ⟶ τ
    # 2. apply inverse phase factor to go from (periodic)⟶(anti-periodic) in τ
    ω_to_τ!(z2,op.timefreqfft,z1)

    # making sure the imaginary part is small enough
    @. z1 = imag(z2)
    @assert norm(z1) < 1e-12 "Imaginary component norm(imag(z2))=$(norm(z1)) too large."

    # copying result to output vector
    @. vout = real(z2)

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
function mul!(vout::AbstractVector{T},op::BlockPreconditioner,vin::AbstractVector{T}) where {T<:Complex}

    neighbor_table_tij = op.holstein.neighbor_table_tij
    coshtij            = op.holstein.coshtij
    sinhtij            = op.holstein.sinhtij

    # copy vector
    copyto!(vout,vin)

    # multiply by checkerboard matrix
    checkerboard_mul!(vout,neighbor_table_tij,coshtij,sinhtij,1)

    # multiply by average exponentiated interaction matrix
    # and a phase factor associated with current ω value
    @. vout = op.ω_phases[op.ω] * op.expnΔτV_bar * vout

    # get final result
    @. vout = vin - vout

    return nothing
end


function Base.eltype(op::BlockPreconditioner{T}) where {T<:AbstractFloat}

    return T
end


function Base.size(op::BlockPreconditioner{T},d::Int) where {T<:AbstractFloat}

    return op.holstein.nsites
end

end