module TimeFreqFFTs

using LinearAlgebra
using FFTW

using ..Lattices:   Lattice
using ..Utilities: reshaped

struct TimeFreqFFT{T<:AbstractFloat,Tfft<:AbstractFFTs.Plan,Tifft<:AbstractFFTs.Plan}

    "Number of site in lattice."
    N::Int

    "Length of imaginary time axis."
    L::Int

    "Vector to represent unitary transformation to restore translational invariance in imaginary time direction."
    Θ::Vector{Complex{T}}

    "FFT plan."
    fftplan::Tfft

    "Inverse FFT plan."
    ifftplan::Tifft

    "Temporary storage vector."
    vtemp::Array{Complex{T},2}

    "Temporary storage vector."
    utemp::Vector{Complex{T}}

    function TimeFreqFFT(T::DataType,N::Int,L::Int)

        @assert T <: AbstractFloat
        vtemp    = zeros(Complex{T},L,N)
        utemp    = zeros(Complex{T},N*L)
        Θ        = [exp(-π*im*(τ-1)/L) for τ = 1:L]
        fftplan  = plan_fft(vtemp, (1,), flags=FFTW.PATIENT)
        ifftplan = plan_ifft(vtemp, (1,), flags=FFTW.PATIENT)
        Tfft     = typeof(fftplan)
        Tifft    = typeof(ifftplan)

        return new{T,Tfft,Tifft}(N,L,Θ,fftplan,ifftplan,vtemp,utemp)
    end
end

function TimeFreqFFT(lattice::Lattice{T},L::Int) where {T<:AbstractFloat}

    return TimeFreqFFT(T,lattice.nsites,L)
end

"""
Apply the transformation ν=[F⋅Θ]⋅v from imaginary time τ to frequency ω space.
"""
function τ_to_ω!(vout::AbstractVector{Complex{T1}},op::TimeFreqFFT{T1},vin::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Number}

    L  = op.L::Int
    N  = op.N::Int
    Θ  = op.Θ::Vector{Complex{T1}}
    vtemp = op.vtemp::Array{Complex{T1},2}
    # apply unitary transformation to restore translation invariance in the
    # imaginary time direciton.
    uin  = reshaped(vin,L,N)
    @fastmath @inbounds for i in 1:N
        for τ in 1:L
            vtemp[τ,i] = Θ[τ] * uin[τ,i]
        end
    end
    # apply (τ ⟶ ω) FFT
    uout = reshaped(vout,L,N)
    mul!(uout,op.fftplan,vtemp)
    return nothing
end

function τ_to_ω!(vout::AbstractVector{T1},op::TimeFreqFFT{T1},vin::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Complex}

    τ_to_ω!(op.utemp,op,vin)
    @. vout = real(op.utemp)
    return nothing
end

function τ_to_ω!(v::AbstractVector{Complex{T}},op::TimeFreqFFT{T}) where {T<:AbstractFloat}

    τ_to_ω!(v,op,v)
    return nothing
end


"""
Apply the transformation v=[Θᵀ⋅Fᵀ]⋅ν from frequency ω to imaginary time τ space.
"""
function ω_to_τ!(vout::AbstractVector{Complex{T}},op::TimeFreqFFT{T},vin::AbstractVector{Complex{T}}) where {T<:AbstractFloat}

    L     = op.L::Int
    N     = op.N::Int
    Θ     = op.Θ::Vector{Complex{T}}
    vtemp = op.vtemp::Array{Complex{T},2}
    # apply (ω ⟶ τ) FFT
    uin  = reshaped(vin, L,N)
    mul!(vtemp,op.ifftplan,uin)
    # apply unitary transformation to restore anitperiodic boundary conditions
    # imaginary time direciton.
    uout = reshaped(vout,L,N)
    @fastmath @inbounds for i in 1:N
        for τ in 1:L
            uout[τ,i] = conj(Θ[τ]) * vtemp[τ,i]
        end
    end
    return nothing
end

function ω_to_τ!(vout::AbstractVector{T},op::TimeFreqFFT{T},vin::AbstractVector{Complex{T}}) where {T<:AbstractFloat}

    L     = op.L::Int
    N     = op.N::Int
    Θ     = op.Θ::Vector{Complex{T}}
    vtemp = op.vtemp::Array{Complex{T},2}
    # apply (ω ⟶ τ) FFT
    uin  = reshaped(vin, L,N)
    mul!(vtemp,op.ifftplan,uin)
    # apply unitary transformation to restore anitperiodic boundary conditions
    # imaginary time direciton.
    uout = reshaped(vout,L,N)
    @fastmath @inbounds for i in 1:N
        for τ in 1:L
            uout[τ,i] = real( conj(Θ[τ]) * vtemp[τ,i] )
        end
    end
    return nothing
end

function ω_to_τ!(vout::AbstractVector{T2},op::TimeFreqFFT{T1},vin::AbstractVector{T1};thresh::T1=1e-12) where {T1<:AbstractFloat,T2<:Complex}

    copyto!(op.vtemp,vin)
    ω_to_τ!(vout,op,op.vtemp)
    return nothing
end

function ω_to_τ!(v::AbstractVector{Complex{T}},op::TimeFreqFFT{T}) where {T<:AbstractFloat}

    copyto!(op.vtemp,v)
    ω_to_τ!(v,op,op.vtemp)
    return nothing
end

end