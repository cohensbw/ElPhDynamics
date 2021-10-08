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
    Number of pairs of Phi vectors to use when estimating transition rate.
    """
    nₚ::Int

    """
    Store forward transition probability for each ϕ.
    """
    Pf::Vector{T}

    """
    Store backward transition probability for each ϕ.
    """
    Pb::Vector{T}
end

function ReflectionUpdate(model::HolsteinModel{T},freq::Int,nsites::Int,nₚ::Int=4) where {T<:AbstractFloat}

    nsites = min(model.Nph,nsites)
    sites  = zeros(Int,nsites)
    Pf     = zeros(T,nₚ)
    Pb     = zeros(T,nₚ)
    return ReflectionUpdate{T}(true,freq,nsites,sites,nₚ,Pf,Pb)
end

function ReflectionUpdate(model::AbstractModel{T},freq::Int,nsites::Int,nₚ::Int=0) where {T<:AbstractFloat}

    Pf     = zeros(T,nₚ)
    Pb     = zeros(T,nₚ)
    sites = Vector{Int}(undef,0)
    return ReflectionUpdate{T}(false,freq,0,sites,nₚ,Pf,Pb)
end

"""
Apply reflection updates to Holstein model.
"""
function special_update!(model::HolsteinModel{T},hmc::HybridMonteCarlo{T},ru::ReflectionUpdate,preconditioner)::T where {T<:AbstractFloat}

    @unpack Nph, Lτ = model
    @unpack nsites, sites, Pf, Pb, nₚ = ru
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

            # iterate over number of ϕ pairs to use
            for p in 1:nₚ

                # CALCULATE FORWARD TRANSITION PROBABILITY

                # resample ϕ
                refresh_ϕ!(hmc,model,sample_R=true)

                # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
                iters, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)

                # get initial action
                S₀ = calc_S(hmc,model)

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
                Pf[p] = min( 1.0 , exp(-(S₁-S₀)) )

                # CALCULATE BACKWARD TRANSITION PROBABILITY

                # resample ϕ
                refresh_ϕ!(hmc,model,sample_R=true)

                # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
                iters, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)

                # get initial action
                S₁′ = calc_S(hmc,model)

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
                Pb[p] = min( 1.0 , exp(-(S₀′-S₁′)) )
            end

            # ACCEPT/REJECT DECISION

            # acceptance probability

            # P̄f = mean(Pf)
            # P̄b = mean(Pb)
            # σf = stdm(Pf,P̄f)
            # σb = stdm(Pb,P̄b)
            # ρ  = cor(Pf,Pb)
            # P  = P̄f/P̄b
            # P  = P + (1/nₚ)*(P*σb^2-ρ*σf*σb)/P̄b^2

            P̄f  = mean(Pf)
            P̄b  = mean(Pb)
            P   = P̄f/P̄b
            θ   = 1/nₚ
            σ²f = varm(Pf,P̄f)
            σ²b = varm(Pb,P̄b)
            σfb = cov(Pb,Pf)
            c²b = σ²b/P̄b^2
            cfb = σfb/(P̄f*P̄b)
            # P   = P + (P̄f-P*P̄b)/(nₚ-1)
            P   = P*(1+θ*cfb)*(1-θ*c²b)

            # accept/reject decision
            if rand(model.rng) < P && iszero(flag)

                # reflect phonon fields
                @. xᵢ = -xᵢ

                # update exp{-Δτ⋅V[x]}
                update_model!(model)

                accepted += 1.0
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
    Number of pairs of Phi vectors to use when estimating transition rate.
    """
    nₚ::Int

    """
    Store forward transition probability for each ϕ.
    """
    Pf::Vector{T}

    """
    Store backward transition probability for each ϕ.
    """
    Pb::Vector{T}
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
    Pf     = zeros(T,nₚ)
    Pb     = zeros(T,nₚ)
    return SwapUpdate{T}(active,freq,nbonds,bonds,nₚ,Pf,Pb)
end

function SwapUpdate(model::AbstractModel{T},freq::Int,nbonds::Int,nₚ::Int=0) where {T<:AbstractFloat}

    nbonds    = 0
    bonds     = Vector{Int}(undef,nbonds)
    Pf        = zeros(T,0)
    Pb        = zeros(T,0)
    return SwapUpdate{T}(active,freq,nbonds,bonds,nₚ,Pf,Pb)
end

"""
Apply swap updates to Holstein model.
"""
function special_update!(model::HolsteinModel{T},hmc::HybridMonteCarlo{T},su::SwapUpdate,preconditioner)::T where {T<:AbstractFloat}

    @unpack Nbonds, Nsites, Lτ, neighbor_table = model
    @unpack nbonds, bonds, Pf, Pb, nₚ = su

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

            # iterate over number of ϕ pairs to use
            for p in 1:nₚ

                # FORWARD TRANSITION PROBABILITY

                # resample ϕ
                refresh_ϕ!(hmc,model,sample_R=true)

                # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
                iters, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)

                # get initial action
                S₀ = calc_S(hmc,model)

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
                Pf[p] = min( 1.0 , exp(-(S₁-S₀)) )

                # BACKWARD TRANSITION PROBABILITY

                # resample ϕ
                refresh_ϕ!(hmc,model,sample_R=true)

                # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
                iters, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)

                # get initial action
                S₁′ = calc_S(hmc,model)

                # swap mean phonon positions
                swap!(xᵢ,xⱼ)

                # update exp{-Δτ⋅V[x]}
                update_model!(model)

                # calculate O⁻¹⋅Λ⋅ϕ₊ and O⁻¹⋅Λ⋅ϕ₋
                if iszero(flag)
                    iters, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)
                end

                # get final action
                S₀′ = calc_S(hmc,model)

                # forward transition proability
                Pb[p] = min( 1.0 , exp(-(S₀′-S₁′)))
            end

            # ACCEPT/REJECT DECISICION

            # acceptance probability

            # P̄f = mean(Pf)
            # P̄b = mean(Pb)
            # σf = stdm(Pf,P̄f)
            # σb = stdm(Pb,P̄b)
            # ρ  = cor(Pf,Pb)
            # P  = P̄f/P̄b
            # P  = P + (1/nₚ)*(P*σb^2-ρ*σf*σb)/P̄b^2

            P̄f  = mean(Pf)
            P̄b  = mean(Pb)
            P   = P̄f/P̄b
            θ   = 1/nₚ
            σ²f = varm(Pf,P̄f)
            σ²b = varm(Pb,P̄b)
            σfb = cov(Pb,Pf)
            c²b = σ²b/P̄b^2
            cfb = σfb/(P̄f*P̄b)
            # P   = P + (P̄f-P*P̄b)/(nₚ-1)
            P   = P*(1+θ*cfb)*(1-θ*c²b)

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