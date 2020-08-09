module FourierAcceleration

using FFTW
using LinearAlgebra
using UnsafeArrays

using ..Models: HolsteinModel
using ..Utilities: get_index

export FourierAccelerator, update_Q!, update_M!, fourier_accelerate!

struct FourierAccelerator{T<:AbstractFloat,Tfft<:AbstractFFTs.Plan,Tifft<:AbstractFFTs.Plan}

    "Copy of input vector if input vector is real."
    vin::Vector{Complex{T}}

    "Copy of output vector if output vector is real."
    vout::Vector{Complex{T}}

    "Vector to store data associated with FFT of single time slice"
    u::Vector{Complex{T}}

    "Vector representing diagonal acceleration matrix using old convention."
    Q::Vector{T}

    "Vector representing c1 matrix using new convetion."
    M::Vector{T}

    "Performs forward fourier transformation"
    pfft::Tfft

    "Performs forward fourier transformation"
    pifft::Tifft

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
    function FourierAccelerator(holstein::HolsteinModel{T1,T2,T3}) where {T1<:AbstractFloat,T2<:Number,T3}

        # getting number of sites in lattice
        N = holstein.lattice.nsites

        # length of imaginary time axis
        L = holstein.Lτ

        # constructing Q and M
        Q = zeros(T1,N*L)

        # constructing M
        M = zeros(T1,N*L)

        # declaring a temporary vector for storage purpose if input vectors are real
        vin = zeros(Complex{T1},N*L)

        # declaring a temporary vector for storage purpose if output vectors are real
        vout = zeros(Complex{T1},N*L)

        # declaring temporary storage vector to represent vectors in frequency space.
        u = zeros(Complex{T1},N*L)

        # vector for planning FFT
        v = zeros(Complex{T1},L,N)

        # planning forward FFT
        pfft  = plan_fft(v, (1,), flags=FFTW.PATIENT)

        # planning inverse FFT
        pifft = plan_ifft(v, (1,), flags=FFTW.PATIENT)

        new{T1,typeof(pfft),typeof(pifft)}(vin,vout,u,Q,M,pfft,pifft,N,L)
    end

end

#######################################################
## FUNCITONS ASSOCIATED WITH FourierAccelerator TYPE ##
#######################################################

"""
Accelerate vector by multiplying with Q matrix.
"""
function fourier_accelerate!(v′::AbstractVector{Complex{T}}, fa::FourierAccelerator{T}, v::AbstractVector{Complex{T}}, power::T; use_mass::Bool=false) where {T<:AbstractFloat}

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
        if use_mass
            @. u *= fa.M^power
        else
            @. u *= fa.Q^power
        end

        # ω → τ
        mul!(a′,fa.pifft,u′)
    end
    return nothing
end

function fourier_accelerate!(v′::AbstractVector{Complex{T}}, fa::FourierAccelerator{T}, v::AbstractVector{T}, power::T; use_mass::Bool=false) where {T<:AbstractFloat}

    copyto!(fa.vin,v)
    fourier_accelerate!(v′,fa,fa.vin,power,use_mass=use_mass)
    return nothing
end

function fourier_accelerate!(v′::AbstractVector{T}, fa::FourierAccelerator{T}, v::AbstractVector{Complex{T}}, power::T; use_mass::Bool=false) where {T<:AbstractFloat}

    fourier_accelerate!(fa.vout,fa,v,power,use_mass=use_mass)
    @. v′ = real(fa.vout)
    return nothing
end

function fourier_accelerate!(v′::AbstractVector{T}, fa::FourierAccelerator{T}, v::AbstractVector{T}, power::T; use_mass::Bool=false) where {T<:AbstractFloat}

    copyto!(fa.vin,v)
    fourier_accelerate!(fa.vout,fa,fa.vin,power,use_mass=use_mass)
    @. v′ = real(fa.vout)
    return nothing
end

function fourier_accelerate!(v::AbstractVector, fa::FourierAccelerator{T}, power::T; use_mass::Bool=false) where {T<:AbstractFloat}

    fourier_accelerate!(v,fa,v,power,use_mass=use_mass)
    return nothing
end


"""
Updates the fourier acceleration matrix for sites with phonon frequencies withing the specified range.
"""
function update_Q!(fa::FourierAccelerator{T1},holstein::HolsteinModel{T1,T2},ω_min::T1,ω_max::T1,m::T1) where {T1<:AbstractFloat,T2<:Number}

    # updating the acceleration matrix for sites with a phonon frequency withing the specified range
    update_Q!(fa.Q,holstein,ω_min,ω_max,m)

    return nothing
end


"""
Updates the c1 matrix for sites with phonon frequencies withing the specified range.
"""
function update_M!(fa::FourierAccelerator{T1},holstein::HolsteinModel{T1,T2},ω_min::T1,ω_max::T1,m0::T1,c::T1=0.0) where {T1<:AbstractFloat,T2<:Number}

    # updating the acceleration matrix for sites with a phonon frequency withing the specified range
    update_M!(fa.M,holstein,ω_min,ω_max,m0,c)

    return nothing
end

#######################
## PRIVATE FUNCTIONS ##
#######################

"""
Updates the fourier acceleration matrix for sites with phonon frequencies withing the specified range.
"""
function update_Q!(Q::Vector{T1},holstein::HolsteinModel{T1,T2},ω_min::T1,ω_max::T1,m::T1) where {T1<:AbstractFloat,T2<:Number}

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
            construct_Qi!( Qi , ω[site] , λ[site] , μ[site] , Δτ , m )
        end
    end
    return nothing
end


"""
Calculates acceleration matrix for specified phonon frequency `ω`, discretization `Δτ`.
Obeys the FFTW convention for the ordering of the momentum values.
"""
function construct_Qi!(Qi::AbstractVector{T},ω::T,λ::T,μ::T,Δτ::T,m::T) where {T<:AbstractFloat}

    L = length(Qi)
    for k in 0:L-1
        Qi[k+1] = element_Qi(k,ω,λ,μ,Δτ,m,L)
    end
    return nothing
end


"""
Calculates a specified matrix element of the acceleration matrix for a given mode k.
"""
function element_Qi(k::Int,ω::T,λ::T,μ::T,Δτ::T,m::T,L::Int)::T where {T<:Number}

    val = (m^2 + Δτ*ω*ω + 4.0/Δτ) / (m^2 + Δτ*ω*ω + (2-2*cos(2*π*k/L))/Δτ)
    return val
end


"""
Updates the c1 matrix for sites with phonon frequencies withing the specified range.
"""
function update_M!(M::Vector{T1},holstein::HolsteinModel{T1,T2},ω_min::T1,ω_max::T1,m0::T1,c::T1) where {T1<:AbstractFloat,T2<:Number}

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
            # get a view into M matrix for current lattice site
            Mi = @view M[get_index(1,site,L):get_index(L,site,L)]
            # define M matrix just for current site
            construct_Mi!( Mi , ω[site] , λ[site] , μ[site] , Δτ , m0 , c )
        end
    end
    return nothing
end


"""
Calculates matrix matrix for specified phonon frequency `ω`, discretization `Δτ`.
Obeys the FFTW convention for the ordering of the momentum values.
"""
function construct_Mi!(Mi::AbstractVector{T},ω::T,λ::T,μ::T,Δτ::T,m0::T,c::T) where {T<:AbstractFloat}

    L = length(Mi)
    for k in 0:L-1
        Mi[k+1] = element_Mi(k,ω,λ,μ,Δτ,m0,c,L)
    end
    return nothing
end


"""
Calculates a specified matrix element of the c1 matrix for a given mode k.
"""
function element_Mi(k::Int,ω::T,λ::T,μ::T,Δτ::T,m0::T,c::T,L::Int)::T where {T<:Number}

    k′  = min(k,L-k)
    m   = m0 * exp(-(c*k′/L)^2)
    val = Δτ * (m^2 + ω^2 + (2-2*cos(2*π*k′/L))/Δτ^2 ) / (m^2 + ω^2)
    return val
end

end
