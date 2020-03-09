module FourierAcceleration

using FFTW

using ..HolsteinModels: HolsteinModel
using ..Utilities: get_index

export FourierAccelerator
export update_Q!
export forward_fft!, inverse_fft!, accelerate!, accelerate_noise!

struct FourierAccelerator{T<:AbstractFloat}

    "Vector to store data associated with single time slice"
    vi::Vector{T}

    "Vector to store data associated with FFT of single time slice"
    νi::Vector{Complex{T}}

    "Vector representing diagonal acceleration matrix."
    Q::Vector{T}

    "Vector representing sqaure root of the diagonal acceleration matrix."
    sqrtQ::Vector{T}

    "Performs forward fourier transformation"
    pfft::FFTW.cFFTWPlan{Complex{T},-1,false,1}

    "Performs forward fourier transformation"
    pifft::AbstractFFTs.ScaledPlan{Complex{T},FFTW.cFFTWPlan{Complex{T},1,false,1},T}

    "Number of sites in lattice getting acclerated."
    nsites::Int

    "Length of imagniary time axis"
    Lτ::Int

    #######################
    ## INNER CONSTRUCTOR ##
    #######################

    """
    Constructor for FourierAccelerator type.
    """
    function FourierAccelerator(holstein::HolsteinModel{T1,T2}, mass::T1, Δt::T1) where {T1<:AbstractFloat,T2<:Number}

        # getting number of sites in lattice
        nsites = holstein.lattice.nsites

        # length of imaginary time axis
        Lτ = holstein.Lτ

        # constructing Q and √(2Q) matrices
        Q = zeros(T1,length(holstein))
        sqrtQ = zeros(T1,length(holstein))
        update_Q!(Q,sqrtQ,holstein,mass,Δt,-Inf,Inf)

        # declaring two full-length vectors for constructing FFT plans
        vi = zeros(T1,Lτ)
        νi = zeros(Complex{T1},Lτ)

        # planning forward FFT
        pfft  = plan_fft( vi )

        # planning inverse FFT
        pifft = plan_ifft( νi )

        new{T1}(vi,νi,Q,sqrtQ,pfft,pifft,nsites,Lτ)
    end

end

#######################################################
## FUNCITONS ASSOCIATED WITH FourierAccelerator TYPE ##
#######################################################

"""
FFT a vector.
"""
function forward_fft!(ν::AbstractVector,v::AbstractVector, fa::FourierAccelerator)

    # iterating over sites in lattice
    for i in 1:fa.nsites
        # copying data associated with current site
        for τ in 1:fa.Lτ
            fa.vi[τ] = real(v[get_index(τ,i,fa.Lτ)])
        end
        # performing FFT
        fa.νi .= fa.pfft * fa.vi
        # copying result for current site into destination vector
        for τ in 1:fa.Lτ
            ν[get_index(τ,i,fa.Lτ)] = fa.νi[τ]
        end
    end
    return nothing
end

"""
Inverse FFT a vector.
"""
function inverse_fft!(v::AbstractVector,ν::AbstractVector,fa::FourierAccelerator)

    # iterating over sites in lattice
    for i in 1:fa.nsites
        # copying data associated with current site
        for τ in 1:fa.Lτ
            fa.νi[τ] = ν[get_index(τ,i,fa.Lτ)]
        end
        # performing iFFT
        fa.vi .= real.(fa.pifft * fa.νi)
        # copying result for current site into destination vector
        for τ in 1:fa.Lτ
            v[get_index(τ,i,fa.Lτ)] = fa.vi[τ]
        end
    end
    return nothing
end


"""
Accelerate vector by multiplying with Q matrix.
"""
function accelerate!(ν::AbstractVector{Complex{T}},fa::FourierAccelerator{T}) where {T<:AbstractFloat}

    ν .*= fa.Q
    return nothing
end


"""
Accelerate noise vector multiplying with √Q matrix.
"""
function accelerate_noise!(η::AbstractVector{Complex{T}},fa::FourierAccelerator{T}) where {T<:AbstractFloat}

    η .*= fa.sqrtQ
    return nothing
end


"""
Updates the fourier acceleration matrix for sites with phonon frequencies withing the specified range.
"""
function update_Q!(fa::FourierAccelerator{T1},holstein::HolsteinModel{T1,T2},mass::T1,Δt::T1,ω_min::T1,ω_max::T1) where {T1<:AbstractFloat,T2<:Number}

    # updating the acceleration matrix for sites with a phonon frequency withing the specified range
    update_Q!(fa.Q,fa.sqrtQ,holstein,mass,Δt,ω_min,ω_max)

    return nothing
end

#######################
## PRIVATE FUNCTIONS ##
#######################

"""
Updates the fourier acceleration matrix for sites with phonon frequencies withing the specified range.
"""
function update_Q!(Q::Vector{T1},sqrtQ::Vector{T1},holstein::HolsteinModel{T1,T2},mass::T1,Δt::T1,ω_min::T1,ω_max::T1) where {T1<:AbstractFloat,T2<:Number}

    nsites = holstein.nsites::Int
    Δτ     = holstein.Δτ::T1
    Lτ     = holstein.Lτ
    ω      = holstein.ω
    λ      = holstein.λ
    μ      = holstein.μ
    # iterating over site in lattice
    for site in 1:nsites
        # if phonon frequncy on site falls withing specified range
        if ω_min < ω[site] < ω_max
            # get a view into Q matrix for current lattice site
            Qi = @view Q[get_index(1,site,Lτ):get_index(Lτ,site,Lτ)]
            # define Q matrix just for current site
            construct_Qi!( Qi , ω[site] , λ[site] , μ[site] , Δτ , mass , Δt )
        end
    end
    # udpating the square root of the acceleration matriz
    @. sqrtQ = sqrt(Q)
    return nothing
end


"""
Calculates acceleration matrix for specified phonon frequency `ω`, discretization `Δτ` and `mass`.
Obeys the FFTW convention for the ordering of the momentum values.
"""
function construct_Qi!(Qi::AbstractVector{T},ω::T,λ::T,μ::T,Δτ::T,mass::T,Δt::T) where {T<:AbstractFloat}

    Lτ = length(Qi)
    for k in 0:Lτ-1
        Qi[k+1] = element_Qi(k,ω,λ,μ,Δτ,mass,Lτ,Δt)/Lτ
    end
    return nothing
end


"""
Calculates a specified matrix element of the acceleration matrix for a given momentum k.
"""
function element_Qi(k::Int,ω::T,λ::T,μ::T,Δτ::T,mass::T,Lτ::Int,Δt::T)::T where {T<:Number}

    val = ( mass*mass + Δτ*ω*ω + 4.0/Δτ)/(mass*mass + Δτ*ω*ω + (2-2*cos(2*π*k/Lτ))/Δτ )
    return val
end

end
