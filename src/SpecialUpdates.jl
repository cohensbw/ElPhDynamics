module SpecialUpdates

using Random
using Statistics
using Parameters
using Printf
using LinearAlgebra
using SparseArrays
using Distributions
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

    """
    Geometric distribution to determine number of Phi vectors to calucate backward transition probability.
    """
    g::Geometric{T}
end

function ReflectionUpdate(model::HolsteinModel{T},freq::Int,nsites::Int,nₚ::Int=4) where {T<:AbstractFloat}

    nsites = min(model.Nph,nsites)
    sites  = zeros(Int,nsites)
    g      = Geometric(1/nₚ)
    return ReflectionUpdate{T}(true,freq,nsites,sites,g)
end

function ReflectionUpdate(model::AbstractModel{T},freq::Int,nsites::Int,nₚ::Int=0) where {T<:AbstractFloat}

    Pf     = zeros(T,nₚ)
    Pb     = zeros(T,nₚ)
    sites = Vector{Int}(undef,0)
    g      = Geometric(0.5)
    return ReflectionUpdate{T}(false,freq,0,sites,g)
end

"""
Apply reflection updates to Holstein model.
"""
function special_update!(model::HolsteinModel{T},hmc::HybridMonteCarlo{T},ru::ReflectionUpdate,preconditioner)::T where {T<:AbstractFloat}

    @unpack Nph, Lτ = model
    @unpack nsites, sites, g = ru
    @unpack ϕ₊, ϕ₋ = hmc

    # counts number of accepted reflections
    accepted = 0.0

    # initialize flag
    flag = 0

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

            # get phonon fields associated with site
            xᵢ = @view x[:,i]

            # get number of Phi vectors to use to make estimate
            N = rand(g) + 1

            # calcualte probability of sampling N
            qₙ = pdf(g,N-1)

            # estimation constant
            ω = 0.5

            # initialize forward tranition probability
            pf = 0.0

            # initialize log of backward tranition probability
            logpb⁻¹ = log(ω/qₙ)

            # iterate over sample
            for n in 1:N

                # CALCULATE FORWARD TRANSITION PROBABILITY

                # resample ϕ and get initial energy
                S₀ = refresh_ϕ!(hmc,model,sample_R=true)

                # reflect phonon fields
                @. xᵢ = -xᵢ

                # update exp{-Δτ⋅V[x]}
                update_model!(model)

                # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
                if iszero(flag)
                    iters, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)
                end

                # get final action
                S₁ = calc_S(hmc,model)

                # forward transition probability
                pf += min( 1.0 , exp(-(S₁-S₀)) ) / N

                # CALCULATE INVERSE OF BACKWARD TRANSITION PROBABILITY

                # resample ϕ and get initial energy
                S₁′ = refresh_ϕ!(hmc,model,sample_R=true)

                # reflect phonon fields
                @. xᵢ = -xᵢ

                # update exp{-Δτ⋅V[x]}
                update_model!(model)

                # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
                if iszero(flag)
                    iters, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)
                end

                # get final action
                S₀′ = calc_S(hmc,model)

                # forward transition probability
                pb = min( 1.0 , exp(-(S₀′-S₁′)) )

                # update estimator
                logpb⁻¹ += log(1.0 - ω*pb)
            end

            # ACCEPT/REJECT DECISICION

            # calculate inverse backward transition probability
            pb⁻¹ = exp(logpb⁻¹)

            # acceptance probability
            P = min( 1.0 , pf*pb⁻¹)

            # accept/reject decision
            if rand(model.rng) < P && iszero(flag)

                accepted += 1.0

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
mutable struct SwapUpdate{T<:AbstractFloat} <: SpecialUpdate

    """
    Whether is turned on.
    """
    active::Bool

    """
    Frequency of special update.
    """
    freq::Int

    """
    Number bonds to swap phonon position across.
    """
    nbonds::Int

    """
    The sites to apply the swap updates to.
    """
    bonds::Vector{Int}

    """
    Geometric distribution to determine number of Phi vectors to calucate backward transition probability.
    """
    g::Geometric{T}
end

function SwapUpdate(model::HolsteinModel{T},freq::Int,nbonds::Int,nₚ::Int=4) where {T<:AbstractFloat}

    @unpack Nsites, Nbonds, neighbor_table = model

    if Nbonds==0
        active = false
    else
        active = true
    end
    nbonds = min(model.Nbonds,nbonds)
    bonds  = zeros(Int,nbonds)
    g      = Geometric(1/nₚ)
    return SwapUpdate{T}(active,freq,nbonds,bonds,g)
end

function SwapUpdate(model::AbstractModel{T},freq::Int,nbonds::Int,nₚ::Int=0) where {T<:AbstractFloat}

    nbonds = 0
    bonds  = Vector{Int}(undef,nbonds)
    g      = Geometric(1/nₚ)
    return SwapUpdate{T}(active,freq,nbonds,bonds,g)
end

"""
Apply swap updates to Holstein model.
"""
function special_update!(model::HolsteinModel{T},hmc::HybridMonteCarlo{T},su::SwapUpdate,preconditioner)::T where {T<:AbstractFloat}

    @unpack Nbonds, Nsites, Lτ, neighbor_table = model
    @unpack nbonds, bonds, g = su

    # counts number of accepted reflections
    accepted = 0.0

    # initialize flag
    flag = 0

    # if updater is active
    if su.active

        # get all phonon fields
        x = reshaped(model.x,Lτ,Nsites)

        # randomly sample sites
        sample!(model.rng,1:Nbonds,bonds,replace=false)

        # update exp{-Δτ⋅V[x]}
        update_model!(model)

        # iterate over sites to apply reflection operation to
        for b in bonds

            # get a randomly selected neighbor table
            i = neighbor_table[1,b]
            j = neighbor_table[2,b]

            # get phonon fields associated with each sites
            xᵢ = @view x[:,i]
            xⱼ = @view x[:,j]

            # get number of Phi vectors to use to make estimate
            N = rand(g) + 1

            # calcualte probability of sampling N
            qₙ = pdf(g,N-1)

            # estimation constant
            ω = 0.5

            # initialize forward transition probability
            pf = 0.0

            # initialize log of backward tranition probability
            logpb⁻¹ = log(ω/qₙ)

            # iterate over sample
            for n in 1:N

                # CALCULATE FORWARD TRANSITION PROBABILITY

                # resample ϕ and get initial action
                S₀ = refresh_ϕ!(hmc,model,sample_R=true)

                # swap mean phonon positions
                swap!(xᵢ,xⱼ)

                # update exp{-Δτ⋅V[x]}
                update_model!(model)

                # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
                if iszero(flag)
                    iters, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)
                end

                # get final action
                S₁ = calc_S(hmc,model)

                # forward transition proability
                pf += min( 1.0 , exp(-(S₁-S₀)) ) / N

                # CALCULATE INVERSE OF BACKWARD TRANSITION PROBABILITY

                # resample ϕ and get initial action
                S₁′ = refresh_ϕ!(hmc,model,sample_R=true)

                # reflect phonon fields
                swap!(xᵢ,xⱼ)

                # update exp{-Δτ⋅V[x]}
                update_model!(model)

                # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
                if iszero(flag)
                    iters, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)
                end

                # get final action
                S₀′ = calc_S(hmc,model)

                # forward transition probability
                pb = min( 1.0 , exp(-(S₀′-S₁′)) )

                # update estimator
                logpb⁻¹ += log(1.0 - ω*pb)
            end

            # ACCEPT/REJECT DECISICION

            # calculate inverse backward transition probability
            pb⁻¹ = exp(logpb⁻¹)

            # acceptance probability
            P = min( 1.0 , pf*pb⁻¹ )

            # accept/reject decision
            if rand(model.rng) < P && iszero(flag)

                accepted += 1.0

                # swap phonon positions
                swap!(xᵢ,xⱼ)

                # update exp{-Δτ⋅V[x]}
                update_model!(model)
            end
        end
    end

    return accepted/nbonds
end

function special_update!(model::AbstractModel{T},hmc::HybridMonteCarlo{T},ru::SwapUpdate,preconditioner)::T where {T<:AbstractFloat}

    return 0.0
end

end