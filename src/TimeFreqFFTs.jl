module TimeFreqFFTs

using FFTW
using UnsafeArrays
using LinearAlgebra

using ..Lattices:   Lattice

struct TimeFreqFFT{T<:AbstractFloat}

    "Number of site in lattice."
    N::Int

    "Length of imaginary time axis."
    Lτ::Int

    "Vector to represent unitary transformation to restore translational invariance in imaginary time direction."
    Θ::Vector{Complex{T}}

    "Vector to represent inverse unitary transformation to restore translational invariance in imaginary time direction."
    invΘ::Vector{Complex{T}}

    "FFT plan."
    fftplan::FFTW.cFFTWPlan{Complex{T},-1,false,2}

    "Inverse FFT plan."
    ifftplan::AbstractFFTs.ScaledPlan{Complex{T},FFTW.cFFTWPlan{Complex{T},1,false,2},T}

    "Temporary storage vector."
    vtemp::Array{Complex{T},2}

    "Temporary storage vector for dealing with a real-valued (not complex) input vector"
    vin::Vector{Complex{T}}
    
    "Temporary storage vector for dealing with a real-valued (not complex) output vector"
    vout::Vector{Complex{T}}

    function TimeFreqFFT(lattice::Lattice{T},Lτ::Int) where {T<:AbstractFloat}

        N        = lattice.nsites
        vin      = zeros(Complex{T},Lτ*N)
        vout     = zeros(Complex{T},Lτ*N)
        vtemp    = zeros(Complex{T},Lτ,N)
        Θ        = [exp(π*im*(τ-1)/Lτ) for τ = 1:Lτ]
        invΘ     = conj.(Θ)
        fftplan  = plan_fft(vtemp, 1, flags=FFTW.PATIENT)
        ifftplan = plan_ifft(vtemp, 1, flags=FFTW.PATIENT)

        new{T}(N,Lτ,Θ,invΘ,fftplan,ifftplan,vtemp,vin,vout)
    end
end


"""
Apply the transformation ν=[F⋅Θ]⋅v from imaginary time τ to frequency ω space.
"""
function τ_to_ω!(vout::AbstractVector{Complex{T1}},ωτfft::TimeFreqFFT{T1},vin::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Number}

    Lτ = ωτfft.Lτ::Int
    N  = ωτfft.N::Int
    Θ  = ωτfft.Θ::Vector{Complex{T1}}
    vtemp = ωτfft.vtemp::Array{Complex{T1},2}
    @uviews vin vtemp vout begin
        # apply unitary transformation to restore translation invariance in the
        # imaginary time direciton.
        uin  = reshape(vin,Lτ,N)
        @inbounds for i in 1:N
            @. @views vtemp[:,i] = Θ * uin[:,i]
        end
        # apply (τ ⟶ ω) FFT
        uout = reshape(vout,Lτ,N)
        mul!(uout,ωτfft.fftplan,vtemp)
    end
    return nothing
end


function τ_to_ω!(v::AbstractVector{Complex{T}},ωτfft::TimeFreqFFT{T}) where {T<:AbstractFloat}

    τ_to_ω!(v,ωτfft,v)
    return nothing
end


"""
Apply the transformation v=[Θ⁺⋅F⁺]⋅ν from frequency ω to imaginary time τ space.
"""
function ω_to_τ!(vout::AbstractVector{Complex{T1}},ωτfft::TimeFreqFFT{T1},vin::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Number}

    Lτ    = ωτfft.Lτ::Int
    N     = ωτfft.N::Int
    invΘ  = ωτfft.invΘ::Vector{Complex{T1}}
    vtemp = ωτfft.vtemp::Array{Complex{T1},2}
    @uviews vin vout begin
        copyto!(vtemp,vin)
        # apply (ω ⟶ τ) FFT
        uout = reshape(vout,Lτ,N)
        mul!(uout,ωτfft.ifftplan,vtemp)
        # apply unitary transformation to restore anitperiodic boundary conditions
        # imaginary time direciton.
        @inbounds for i in 1:N
            @. @views uout[:,i] *= invΘ
        end
    end
    return nothing
end

function ω_to_τ!(v::AbstractVector{Complex{T}},ωτfft::TimeFreqFFT{T}) where {T<:AbstractFloat}

    ω_to_τ!(v,ωτfft,v)
    return nothing
end

end