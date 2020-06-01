module FourierAcceleration

using FFTW
using LinearAlgebra
using UnsafeArrays

using ..HolsteinModels: HolsteinModel
using ..Utilities: get_index

export FourierAccelerator, update_Q!, fourier_accelerate!

struct FourierAccelerator{T<:AbstractFloat}

    "Copy of input vector if input vector is real."
    vin::Vector{Complex{T}}

    "Copy of output vector if output vector is real."
    vout::Vector{Complex{T}}

    "Vector to store data associated with FFT of single time slice"
    u::Vector{Complex{T}}

    "Vector representing diagonal acceleration matrix."
    Q::Vector{T}

    "Performs forward fourier transformation"
    pfft::FFTW.cFFTWPlan{Complex{T},-1,false,2}

    "Performs forward fourier transformation"
    pifft::AbstractFFTs.ScaledPlan{Complex{T},FFTW.cFFTWPlan{Complex{T},1,false,2},T}

    "Number of sites in lattice getting acclerated."
    N::Int

    "Length of imagniary time axis"
    L::Int

    #######################
    ## INNER CONSTRUCTOR ##
    #######################

    """
    Constructor for FourierAccelerator type.
    """
    function FourierAccelerator(holstein::HolsteinModel{T1,T2}, mass::T1, Δt::T1) where {T1<:AbstractFloat,T2<:Number}

        # getting number of sites in lattice
        N = holstein.lattice.nsites

        # length of imaginary time axis
        L = holstein.Lτ

        # constructing Q and √(2Q) matrices
        Q = zeros(T1,N*L)
        update_Q!(Q,holstein,mass,Δt,-Inf,Inf)

        # declaring a temporary vector for storage purpose if input vectors are real
        vin = zeros(Complex{T1},N*L)

        # declaring a temporary vector for storage purpose if output vectors are real
        vout = zeros(Complex{T1},N*L)

        # declaring temporary storage vector to represent vectors in frequency space.
        u = zeros(Complex{T1},L)

        # vector for planning FFT
        v = zeros(Complex{T1},L,N)

        # planning forward FFT
        pfft  = plan_fft(v, (1,), flags=FFTW.PATIENT)

        # planning inverse FFT
        pifft = plan_ifft(v, (1,), flags=FFTW.PATIENT)

        new{T1}(vin,vout,u,Q,pfft,pifft,N,L)
    end

end

#######################################################
## FUNCITONS ASSOCIATED WITH FourierAccelerator TYPE ##
#######################################################

"""
Accelerate vector by multiplying with Q matrix.
"""
function fourier_accelerate!(v′::AbstractVector{Complex{T}}, fa::FourierAccelerator{T}, v::AbstractVector{Complex{T}}, power::T) where {T<:AbstractFloat}

    N = fa.N # number of sites in lattice
    L = fa.L # length of imaginary time axis
    u = fa.u::Vector{Complex{T}}

    @uviews u v v′ begin
        
        a  = reshape(v,L,N)
        a′ = reshape(v′,L,N)
        u′ = reshape(u,L,N)

        # τ → ω
        mul!(u′,fa.pfft,a)

        # apply diagonal acceleration matrix Q^power
        @. u *= fa.Q^power

        # ω → τ
        mul!(a′,fa.pifft,u′)
    end
    return nothing
end

function fourier_accelerate!(v′::AbstractVector{Complex{T}}, fa::FourierAccelerator{T}, v::AbstractVector{T}, power::T) where {T<:AbstractFloat}

    copyto!(fa.vin,v)
    fourier_accelerate!(v′,fa,fa.vin,power)
    return nothing
end

function fourier_accelerate!(v′::AbstractVector{T}, fa::FourierAccelerator{T}, v::AbstractVector{Complex{T}}, power::T) where {T<:AbstractFloat}

    fourier_accelerate!(fa.vout,fa,v,power)
    @. v′ = real(fa.vout)
    return nothing
end

function fourier_accelerate!(v′::AbstractVector{T}, fa::FourierAccelerator{T}, v::AbstractVector{T}, power::T) where {T<:AbstractFloat}

    copyto!(fa.vin,v)
    fourier_accelerate!(fa.vout,fa,fa.vin,power)
    @. v′ = real(fa.vout)
    return nothing
end

function fourier_accelerate!(v::AbstractVector, fa::FourierAccelerator{T}, power::T) where {T<:AbstractFloat}

    fourier_accelerate!(v,fa,v,power)
    return nothing
end


"""
Updates the fourier acceleration matrix for sites with phonon frequencies withing the specified range.
"""
function update_Q!(fa::FourierAccelerator{T1},holstein::HolsteinModel{T1,T2},mass::T1,Δt::T1,ω_min::T1,ω_max::T1) where {T1<:AbstractFloat,T2<:Number}

    # updating the acceleration matrix for sites with a phonon frequency withing the specified range
    update_Q!(fa.Q,holstein,mass,Δt,ω_min,ω_max)

    return nothing
end

#######################
## PRIVATE FUNCTIONS ##
#######################

"""
Updates the fourier acceleration matrix for sites with phonon frequencies withing the specified range.
"""
function update_Q!(Q::Vector{T1},holstein::HolsteinModel{T1,T2},mass::T1,Δt::T1,ω_min::T1,ω_max::T1) where {T1<:AbstractFloat,T2<:Number}

    N  = holstein.nsites::Int
    L  = holstein.Lτ
    Δτ = holstein.Δτ::T1
    ω  = holstein.ω
    λ  = holstein.λ
    μ  = holstein.μ
    # iterating over site in lattice
    for site in 1:N
        # if phonon frequncy on site falls withing specified range
        if ω_min < ω[site] < ω_max
            # get a view into Q matrix for current lattice site
            Qi = @view Q[get_index(1,site,L):get_index(L,site,L)]
            # define Q matrix just for current site
            construct_Qi!( Qi , ω[site] , λ[site] , μ[site] , Δτ , mass , Δt )
        end
    end
    return nothing
end


"""
Calculates acceleration matrix for specified phonon frequency `ω`, discretization `Δτ` and `mass`.
Obeys the FFTW convention for the ordering of the momentum values.
"""
function construct_Qi!(Qi::AbstractVector{T},ω::T,λ::T,μ::T,Δτ::T,mass::T,Δt::T) where {T<:AbstractFloat}

    L = length(Qi)
    for k in 0:L-1
        Qi[k+1] = element_Qi(k,ω,λ,μ,Δτ,mass,L,Δt)
    end
    return nothing
end


"""
Calculates a specified matrix element of the acceleration matrix for a given momentum k.
"""
function element_Qi(k::Int,ω::T,λ::T,μ::T,Δτ::T,mass::T,L::Int,Δt::T)::T where {T<:Number}

    val = (mass*mass + Δτ*ω*ω + 4.0/Δτ) / (mass*mass + Δτ*ω*ω + (2-2*cos(2*π*k/L))/Δτ)
    return val
end

end
