module TimeFreqFFTs

using FFTW
using UnsafeArrays
using LinearAlgebra

using ..Lattices:   Lattice

struct TimeFreqFFT{T<:AbstractFloat}

    "Number of site in lattice."
    N::Int

    "Length of imaginary time axis."
    L::Int

    "Vector to represent unitary transformation to restore translational invariance in imaginary time direction."
    Θ::Vector{Complex{T}}

    "FFT plan."
    fftplan::FFTW.cFFTWPlan{Complex{T},-1,false,2}

    "Inverse FFT plan."
    ifftplan::AbstractFFTs.ScaledPlan{Complex{T},FFTW.cFFTWPlan{Complex{T},1,false,2},T}

    "Temporary storage vector."
    vtemp::Array{Complex{T},2}

    "Temporary storage vector."
    utemp::Vector{Complex{T}}

    function TimeFreqFFT(lattice::Lattice{T},L::Int) where {T<:AbstractFloat}

        N        = lattice.nsites
        vtemp    = zeros(Complex{T},L,N)
        utemp    = zeros(Complex{T},L*N)
        Θ        = [exp(-π*im*(τ-1)/L) for τ = 1:L]
        fftplan  = plan_fft(vtemp, (1,), flags=FFTW.PATIENT)
        ifftplan = plan_ifft(vtemp, (1,), flags=FFTW.PATIENT)

        new{T}(N,L,Θ,fftplan,ifftplan,vtemp,utemp)
    end
end


"""
Apply the transformation ν=[F⋅Θ]⋅v from imaginary time τ to frequency ω space.
"""
function τ_to_ω!(vout::AbstractVector{Complex{T1}},op::TimeFreqFFT{T1},vin::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Number}

    L  = op.L::Int
    N  = op.N::Int
    Θ  = op.Θ::Vector{Complex{T1}}
    vtemp = op.vtemp::Array{Complex{T1},2}
    @uviews vin vtemp vout begin
        # apply unitary transformation to restore translation invariance in the
        # imaginary time direciton.
        uin  = reshape(vin,L,N)
        @fastmath @inbounds for i in 1:N
            for τ in 1:L
                vtemp[τ,i] = Θ[τ] * uin[τ,i]
            end
        end
        # apply (τ ⟶ ω) FFT
        uout = reshape(vout,L,N)
        mul!(uout,op.fftplan,vtemp)
    end
    return nothing
end

function τ_to_ω!(vout::AbstractVector{T1},op::TimeFreqFFT{T1},vin::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Complex}

    utemp = op.utemp
    τ_to_ω!(utemp,op,vin)
    @. vout = real(utemp)
    return nothing
end

function τ_to_ω!(v::AbstractVector{Complex{T}},op::TimeFreqFFT{T}) where {T<:AbstractFloat}

    τ_to_ω!(v,op,v)
    return nothing
end


"""
Apply the transformation v=[Θ⁺⋅F⁺]⋅ν from frequency ω to imaginary time τ space.
"""
function ω_to_τ!(vout::AbstractVector{Complex{T1}},op::TimeFreqFFT{T1},vin::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Number}

    L    = op.L::Int
    N     = op.N::Int
    Θ     = op.Θ::Vector{Complex{T1}}
    vtemp = op.vtemp::Array{Complex{T1},2}
    @uviews vin vout begin
        copyto!(vtemp,vin)
        # apply (ω ⟶ τ) FFT
        uout = reshape(vout,L,N)
        mul!(uout,op.ifftplan,vtemp)
        # apply unitary transformation to restore anitperiodic boundary conditions
        # imaginary time direciton.
        @fastmath @inbounds for i in 1:N
            for τ in 1:L
                uout[τ,i] *= conj(Θ[τ])
            end
        end
    end
    return nothing
end

function ω_to_τ!(vout::AbstractVector{T1},op::TimeFreqFFT{T1},vin::AbstractVector{T2};thresh::T1=1e-12) where {T1<:AbstractFloat,T2<:Complex}

    utemp = op.utemp
    ω_to_τ!(utemp,op,vin)
    @. vout = real(utemp)
    return nothing
end

function ω_to_τ!(v::AbstractVector{Complex{T}},op::TimeFreqFFT{T}) where {T<:AbstractFloat}

    ω_to_τ!(v,op,v)
    return nothing
end

end