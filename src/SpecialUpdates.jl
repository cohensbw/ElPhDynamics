module SpecialUpdates

using Random
using Statistics
using Parameters
using Printf
using LinearAlgebra
using SparseArrays
using StatsBase

using ..Utilities: get_index, reshaped, swap!
using ..Models: AbstractModel, HolsteinModel, SSHModel, update_model!, mulM!, muldMdx!, mulMᵀ!, construct_M
using ..PhononAction: calc_Sb
using ..HMC: HybridMonteCarlo, refresh_ϕ!, calc_S, calc_Sf, calc_O⁻¹Λϕ!, update_Λ!

export SpecialUpdate, NullUpdate, ReflectionUpdate, SwapUpdate, special_update!

"""
Abstract type to represent special updates.
"""
abstract type SpecialUpdate end


"""
Represent Null Special Update, meant to act as a placeholder in the code
when no special update is being used.
"""
struct NullUpdate <: SpecialUpdate

    """
    Whether the updater is active (always false).
    """
    active::Bool

    """
    Frequency of updates (set to 1 because never active).
    """
    freq::Int

    function NullUpdate()

        return new(false,1)
    end
end

"""
NullUpdate behavior.
"""
function special_update!(model::AbstractModel{T},hmc::HybridMonteCarlo{T},nu::NullUpdate,preconditioner)::T where {T<:AbstractFloat}

    return 0.0
end

"""
Performs Reflection update on Holstein model where (xᵢ)⟶(-xᵢ) on some set of sites i.
On all other models it is a null operation.
"""
mutable struct ReflectionUpdate <: SpecialUpdate

    """
    Whether the updater is turned on.
    """
    active::Bool

    """
    Frequency of special updates.
    """
    freq::Int
    
    """
    Number of sites the reflection will be applied to.
    """
    nsites::Int

    """
    Which sites the reflection will be applied to.
    """
    sites::Vector{Int}
end

function ReflectionUpdate(model::HolsteinModel,freq::Int,nsites::Int)

    nsites = min(model.Nph,nsites)
    sites = zeros(Int,nsites)
    return ReflectionUpdate(true,freq,nsites,sites)
end

function ReflectionUpdate(model::AbstractModel,freq::Int,nsites::Int)

    sites = Vector{Int}(undef,0)
    return ReflectionUpdate(false,freq,0,sites)
end

"""
Apply reflection updates to Holstein model.
"""
function special_update!(model::HolsteinModel{T},hmc::HybridMonteCarlo{T},ru::ReflectionUpdate,preconditioner)::T where {T<:AbstractFloat}

    @unpack Nph, Lτ = model
    @unpack nsites, sites = ru

    # counts number of accepted reflections
    accepted = 0.0

    # if updater is active
    if ru.active

        # get all phonon fields
        x = reshaped(model.x,Lτ,Nph)

        # randomly sample sites
        sample!(model.rng,1:Nph,sites,replace=false)

        # update exp{-Δτ⋅V[x]}
        update_model!(model)

        # iterate over sites to apply reflection operation to
        for i in sites

            # resample ϕ
            refresh_ϕ!(hmc,model)

            # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
            iters = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)

            # get initial action
            S₀ = calc_S(hmc,model)

            # get phonon fields associated with site
            xᵢ = @view x[:,i]

            # reflect phonon fields
            @. xᵢ = -xᵢ

            # update exp{-Δτ⋅V[x]}
            update_model!(model)

            # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
            iters = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)

            # get final action
            S₁ = calc_S(hmc,model)

            # accept/reject decision
            if rand(model.rng) < exp(-(S₁-S₀))

                accepted += 1.0
            else

                # reflect phonon fields
                @. xᵢ = -xᵢ

                # update exp{-Δτ⋅V[x]}
                update_model!(model)
            end
        end
    end

    return accepted/nsites
end

function special_update!(model::AbstractModel{T},hmc::HybridMonteCarlo{T},ru::ReflectionUpdate,preconditioner)::T where {T<:AbstractFloat}

    return 0.0
end


"""
Swap update in Holstein model where adjacent sites swap phonon worldlines.
"""
mutable struct SwapUpdate <: SpecialUpdate

    """
    Whether is turned on.
    """
    active::Bool

    """
    Frequency of special update.
    """
    freq::Int

    """
    Number of pairs of sites to apply swap to.
    """
    nsites::Int

    """
    The sites to apply the swap updates to.
    """
    sites::Vector{Int}

    """
    Neighbors of each site.
    """
    neighbors::Vector{Vector{Int}}
end

function SwapUpdate(model::HolsteinModel,freq::Int,nsites::Int)

    @unpack Nsites, Nbonds, neighbor_table = model

    if Nbonds > 0
        # get neighbors associated with each site in the lattice
        active    = true
        neighbors = [Vector{Int}(undef,0) for n in 1:Nsites]
        for b in 1:Nbonds
            i = neighbor_table[1, b]
            j = neighbor_table[2, b]
            append!(neighbors[i],j)
            append!(neighbors[j],i)
        end
    else
        active    = false
        neighbors = Vector{Vector{Int}}(undef,Int)
    end

    nsites = min(model.Nph,nsites)
    sites  = zeros(Int,nsites)
    return SwapUpdate(active,freq,nsites,sites,neighbors)
end

function SwapUpdate(model::AbstractModel,freq::Int,nsites::Int)

    neighbors = Vector{Vector{Int}}(undef,Int)
    sites     = Vector{Int}(undef,0)
    return SwapUpdate(false,freq,0,sites,neighbors)
end

"""
Apply swap updates to Holstein model.
"""
function special_update!(model::HolsteinModel{T},hmc::HybridMonteCarlo{T},su::SwapUpdate,preconditioner)::T where {T<:AbstractFloat}

    @unpack Nsites, Lτ = model
    @unpack nsites, sites, neighbors = su

    # counts number of accepted reflections
    accepted = 0.0

    # if updater is active
    if su.active

        # get all phonon fields
        x = reshaped(model.x,Lτ,Nsites)

        # randomly sample sites
        sample!(model.rng,1:Nsites,sites,replace=false)

        # iterate over sites to apply reflection operation to
        for i in sites

            # site has more than zero neighbors
            if length(neighbors[i]) > 0

                # get a randomly selected neighbor sie
                j = sample(model.rng,neighbors[i])

                # get phonon fields associated with each sites
                xᵢ = @view x[:,i]
                xⱼ = @view x[:,j]

                # resample ϕ
                refresh_ϕ!(hmc,model)

                # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
                iters = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)

                # get initial action
                S₀ = calc_S(hmc,model)

                # swap phonon fields
                swap!(xᵢ,xⱼ)

                # update exp{-Δτ⋅V[x]}
                update_model!(model)

                # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
                iters = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)

                # get final action
                S₁ = calc_S(hmc,model)

                # accept/reject decision
                if rand(model.rng) < exp(-(S₁-S₀))

                    accepted += 1.0
                else

                    # swap phonon fields
                    swap!(xᵢ,xⱼ)

                    # update exp{-Δτ⋅V[x]}
                    update_model!(model)
                end
            end
        end
    end

    return accepted/nsites
end

function special_update!(model::AbstractModel{T},hmc::HybridMonteCarlo{T},ru::SwapUpdate,preconditioner)::T where {T<:AbstractFloat}

    return 0.0
end

end