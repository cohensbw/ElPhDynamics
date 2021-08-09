module SpecialUpdates

using Random
using Statistics
using Parameters
using Printf
using LinearAlgebra
using SparseArrays
using StatsBase

using ..Utilities: get_index, reshaped
using ..Models: AbstractModel, HolsteinModel, SSHModel, update_model!, mulM!, muldMdx!, mulMᵀ!, construct_M
using ..PhononAction: calc_Sb
using ..HMC: HybridMonteCarlo, refresh_ϕ!, calc_S, calc_Sf, calc_O⁻¹Λϕ!, update_Λ!

export SpecialUpdate, NullUpdate, ReflectionUpdate, special_update!

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
mutable struct ReflectionUpdate{T<:AbstractFloat} <: SpecialUpdate

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

    function ReflectionUpdate(x::AbstractVector{T},Nph::Int,active::Bool,freq::Int,nsites::Int) where {T<:AbstractFloat}

        if nsites <= 0 || freq <= 0
            active = false
            sites  = zeros(Int,0)
        else
            sites = zeros(Int,nsites)
        end
        return new{T}(active,freq,nsites,sites)
    end
end

function ReflectionUpdate(model::HolsteinModel{T},freq::Int,nsites::Int) where {T}

    nsites = min(model.Nph,nsites)
    return ReflectionUpdate(model.x,model.Nph,true,freq,nsites)
end

function ReflectionUpdate(model::AbstractModel{T},freq::Int,nsites::Int) where {T}

    return ReflectionUpdate(model.x,model.Nph,false,freq,nsites)
end

"""
Apply reflection updates to Holstein model.
"""
function special_update!(model::HolsteinModel{T},hmc::HybridMonteCarlo{T},ru::ReflectionUpdate{T},preconditioner)::T where {T<:AbstractFloat}

    @unpack Nph, Lτ = model
    @unpack nsites, sites = ru

    # get all phonon fields
    x = reshaped(model.x,Lτ,Nph)

    # randomly sample sites
    sample!(model.rng,1:Nph,sites,replace=false)

    # counts number of accepted reflections
    accepted = 0.0

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

    return accepted/nsites
end

function special_update!(model::AbstractModel{T},hmc::HybridMonteCarlo{T},ru::ReflectionUpdate{T},preconditioner)::T where {T<:AbstractFloat}

    return 0.0
end

end