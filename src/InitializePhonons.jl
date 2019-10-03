module InitializePhonons

using ..HolsteinModels: HolsteinModel, construct_expnΔτV!
using ..HolsteinModels: get_index

export init_phonons_single_site!


"""
Intializes phonon fields in `HolsteinModel` instance using the
single-site limit where the hopping between sites is zero.
Uses the Levy Construction to sample phonon fields in the imaginary
time direction.
"""
function init_phonons_single_site!(holstein::HolsteinModel{T1,T2}) where {T1<:AbstractFloat,T2<:Number}
    
    # info about temperature of system
    β  = holstein.β::T1
    Δτ = holstein.Δτ::T1
    Lτ = holstein.Lτ::Int
    
    # number of site in physical lattice
    nsites = holstein.nsites::Int
    
    # phonon frequency of site
    ω = 0.0
    # el-ph coupling on site
    λ = 0.0
    # chemical potential of site
    μ = 0.0
    
    # statistical weight associated with 1 or 2 electrons on site.
    # The statistical weight of 0 electrons is just 1.
    Z1 = 0.0
    Z2 = 0.0
    # normalization
    Z = 0.0
    
    # holds random number between [0.0,1.0)
    r = 0.0
    
    # iterating over sites in lattice
    for site in 1:nsites
        
        # parameters associated with site
        μ = holstein.μ[site]
        ω = holstein.ω[site]
        λ = holstein.λ[site]
        
        # if non-zero phonon frequency
        if abs(ω)>0.0
            
            # constructing levy harmonic path for site along τ-axis
            path = @view holstein.ϕ[(site-1)*Lτ+1:site*Lτ]
            levy_path!(path,ω,β,Δτ,Lτ)
        
            # if non-zero el-phonon coupling
            if abs(λ)>0.0

                # chemical potential for half-filling
                μhalf = -(λ*λ)/(ω*ω)
                # weight associated with 1 electron on site
                Z1 = 2.0*exp(β*(λ^2/ω^2/2.0+μ))
                # weight associated with 2 electron on site
                Z2 = exp(2*β*(λ^2/ω^2+μ))
                # normalization
                Z = 1.0 + Z1 + Z2
                
                # sample random number between [0.0,1.0)
                r = rand()
                # if 1 electrons on site
                if r < abs(Z1/Z)
                    # shifting mean position of phonon fields on site
                    path .-= λ/ω^2
                # if 2 electrons on site
                elseif r < abs((Z1+Z2)/Z)
                    # shifting mean position of phonon fields on site
                    path .-= 2.0*λ/ω^2
                end
            end
        end
    end
    # construct the exponentiated interaction matrix based on intialized phonon fields
    construct_expnΔτV!(holstein)

    return nothing
end


"""
Directly samples the phonon fields along the imaginary time axis
for a QHO using the Levy construction.
"""
function levy_path!(path::AbstractVector,ω::Number,β::AbstractFloat,Δτ::AbstractFloat,Lτ::Int)
    
    x1 = sample_qho(ω,β)
    γ1 = 0.0
    γ2 = 0.0
    μi = 0.0
    σi = 0.0
    path[1] = x1
    for i = 2:Lτ
        γ1 = coth(ω*Δτ) + coth((Lτ+1-i)*ω*Δτ)
        γ2 = path[i-1]/sinh(ω*Δτ) + x1/sinh((Lτ+1-i)*ω*Δτ)
        μi = γ2/γ1
        σi = 1.0/sqrt(γ1)
        path[i] = μi + σi*randn()
    end
    return nothing
end


"""
Samples the position distribution of a QHO with frequency `ω`
at inverse temperature `β`.
"""
function sample_qho(ω::T,β::AbstractFloat)::T where {T<:Number}
    
    σ = sqrt( 1.0/(2.0*ω*tanh(ω*β/2.0)) )
    return σ*randn()
end

end