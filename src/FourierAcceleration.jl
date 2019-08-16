module FourierAcceleration

using FFTW

using Langevin.HolsteinModels: HolsteinModel
using Langevin.HolsteinModels: view_by_site, view_by_τ

export FourierAccelerator
export update_Q!
export forward_fft!, inverse_fft!, accelerate!, accelerate_noise!

struct FourierAccelerator{T<:AbstractFloat}

    "Vector representing diagonal acceleration matrix."
    Q::Vector{T}

    "Vector representing sqaure root of the diagonal acceleration matrix."
    sqrt2Q::Vector{T}

    "Performs forward fourier transformation"
    pfft::FFTW.cFFTWPlan{Complex{T},-1,false,1}

    "Performs forward fourier transformation"
    pifft::AbstractFFTs.ScaledPlan{Complex{T},FFTW.cFFTWPlan{Complex{T},1,false,1},T}

    "Number of sites in lattice getting acclerated."
    nsites::Int

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

        # constructing Q matrix
        Q = zeros(T2,length(holstein))
        update_Q!(Q,holstein,mass,Δt,-Inf,Inf)

        # calculating the square root of the Q matrix
        sqrtQ = sqrt.(Q)

        # declaring two full-length vectors for constructing FFT plans
        v  = ones(T2,length(holstein))
        ν = ones(Complex{T1},length(holstein))

        # getting a view into each vector corresponding to site 1 in the lattice
        v1 = view_by_site(v,1,nsites)
        ν1 = view_by_site(ν,1,nsites)

        # planning forward FFT
        pfft  = plan_fft( v1 )

        # planning inverse FFT
        pifft = plan_ifft( ν1 )

        new{T1}(Q,sqrtQ,pfft,pifft,nsites)
    end

end

#######################################################
## FUNCITONS ASSOCIATED WITH FourierAccelerator TYPE ##
#######################################################

"""
FFT a vector.
"""
function forward_fft!(ν::AbstractVector{Complex{T1}},v::AbstractVector{T2},fa::FourierAccelerator{T1}) where {T1<:AbstractFloat,T2<:Number}

    for site in 1:fa.nsites
        νi = view_by_site(ν,site,fa.nsites)
        vi = view_by_site(v,site,fa.nsites)
        νi .= fa.pfft * vi
    end
    return nothing
end

"""
Inverse FFT a vector.
"""
function inverse_fft!(v::AbstractVector{T1},ν::AbstractVector{Complex{T2}},fa::FourierAccelerator{T2}) where {T1<:Number,T2<:AbstractFloat}

    for site in 1:fa.nsites
        νi = view_by_site(ν,site,fa.nsites)
        vi = view_by_site(v,site,fa.nsites)
        vi .= real.(fa.pifft * νi)
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

    η .*= fa.sqrt2Q
    return nothing
end


"""
Updates the fourier acceleration matrix for sites with phonon frequencies withing the specified range.
"""
function update_Q!(fa::FourierAccelerator{T1},holstein::HolsteinModel{T1,T2},mass::T1,Δt::T1,ω_min::T1,ω_max::T1) where {T1<:AbstractFloat,T2<:Number}

    # updating the acceleration matrix for sites with a phonon frequency withing the specified range
    update_Q!(fa.Q,holstein,mass,Δt,ω_min,ω_max)

    # udpating the square root of the acceleration matriz
    @. fa.sqrt2Q = sqrt(2.0*fa.Q)

    return nothing
end

#######################
## PRIVATE FUNCTIONS ##
#######################

"""
Updates the fourier acceleration matrix for sites with phonon frequencies withing the specified range.
"""
function update_Q!(Q::Vector{T1},holstein::HolsteinModel{T1,T2},mass::T1,Δt::T1,ω_min::T1,ω_max::T1) where {T1<:AbstractFloat,T2<:Number}

    nsites = holstein.nsites::Int
    Δτ     = holstein.Δτ::T1
    # iterating over site in lattice
    for site in 1:nsites
        # if phonon frequncy on site falls withing specified range
        if ω_min < holstein.ω[site] < ω_max
            # get a view into Q matrix for current lattice site
            Qi = view_by_site(Q,site,nsites)
            # define Q matrix just for current site
            construct_Qi!(Qi,holstein.ω[site],Δτ,mass,Δt)
        end
    end
    return nothing
end


"""
Calculates acceleration matrix for specified phonon frequency `ω`, discretization `Δτ` and `mass`.
Obeys the FFTW convention for the ordering of the momentum values.
"""
function construct_Qi!(Qi::AbstractVector{T},ω::T,Δτ::T,mass::T,Δt::T) where {T<:AbstractFloat}

    Lτ = length(Qi)
    k = 0 # momentum value
    for i in 1:div(Lτ,2)+1
        k = i-1 # converting index to momentum using FFTW convention
        Qi[i] = element_Qi(k,ω,Δτ,mass,Lτ,Δt)/Lτ
    end
    for i in div(Lτ,2)+2:Lτ
        k = i-Lτ-1 # converting index to momentum using FFTW convention
        Qi[i] = element_Qi(k,ω,Δτ,mass,Lτ,Δt)/Lτ
    end
    return nothing
end


"""
Calculates a specified matrix element of the acceleration matrix for a given momentum k.
"""
function element_Qi(k::Int,ω::T,Δτ::T,mass::T,Lτ::Int,Δt::T)::T where {T<:AbstractFloat}

    return Δt*(mass*mass + Δτ*ω*ω + 4.0/Δτ)/(mass*mass + Δτ*ω*ω + (2-2*cos(2*π*k/Lτ))/Δτ )
end

end
