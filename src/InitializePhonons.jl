module InitializePhonons

using Random

using ..Models: HolsteinModel, SSHModel, update_model!
using ..Utilities: get_index, reshaped

export init_phonons_single_site!, init_phonons_half_filled!, sample_qho


function init_phonons_half_filled!(ssh::SSHModel{T1,T2}) where {T1,T2}
    
    # info about temperature of system
    β  = ssh.β::T1
    Δτ = ssh.Δτ::T1
    Lτ = ssh.Lτ::Int

    # number of phonons
    Nph = ssh.Nph

    # get phonon fields
    x = ssh.x

    # keeps track of the fields
    field = 0

    # iterate over phonons
    for phonon in 1:Nph

        # calculate average phonon position
        bond = ssh.phonon_to_bond[phonon]
        t    = ssh.t[bond]
        α    = ssh.α[phonon]
        ω    = ssh.ω[phonon]
        if iszero(α) || iszero(t)
            x0 = sample_qho(ω,β,ssh.rng)
        else
            x0 = (2.0*rand(ssh.rng)-1.0)/5.0 * (t/α)
        end

        # iterate over imaginary time slices
        for τ in 1:Lτ

            # increment field count
            field += 1

            # set phonon value
            x[field] = x0
        end
    end

    # account for equivalent fields
    @views @. x = x[ssh.primary_field]

    # udpate exponentiated interaction matrix
    update_model!(ssh)

    return nothing
end

function init_phonons_half_filled!(holstein::HolsteinModel{T1,T2}) where {T1,T2}

    # info about temperature of system
    β  = holstein.β::T1
    Δτ = holstein.Δτ::T1
    Lτ = holstein.Lτ::Int

    # iterate over sites in lattice
    for site in 1:holstein.Nsites

        # get parameters for site in lattice
        ω = holstein.ω[site]
        λ = holstein.λ[site]

        # get all phonon fields corresponding to current
        # site in lattice.
        i_start = get_index(1,site,Lτ)
        i_end   = get_index(Lτ,site,Lτ)
        path    = @view holstein.x[i_start:i_end]

        # shift levy path by ammount corresponding having either
        # a density of 0, 1 or 2 on the site
        x0 = -λ/ω^2 * rand(holstein.rng,0:2)
        xr = x0 + sample_qho(ω,β,holstein.rng)
        @. path = xr
    end

    # udpate exponentiated interaction matrix
    update_model!(holstein)

    return nothing
end

"""
Samples the position distribution of a QHO with frequency `ω` at inverse temperature `β`.
"""
function sample_qho(ω::T,β::T,rng::AbstractRNG)::T where {T<:Number}
    
    if ω>0
        σ = 1/sqrt(2*ω*tanh(β*ω/2))
    else
        σ = 1.0
    end
    return σ*randn(rng)
end

end