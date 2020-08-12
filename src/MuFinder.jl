module MuFinder

using Parameters
using Statistics
using LinearAlgebra

export MuTuner, update_μ!

mutable struct MuTuner{T<:AbstractFloat}

    active      :: Bool
    μ_traj      :: Vector{T}
    N_traj      :: Vector{T}
    κ_traj      :: Vector{T}
    forgetful_c :: T
    μ           :: T
    β           :: T
    Δτ          :: T
    L           :: Int
    target_N    :: T
    μ̄           :: T
    N̄           :: T
    κ̄           :: T
    κ_var       :: T

    function MuTuner(active::Bool, init_μ::T, target_N::T, β::T, Δτ::T, κ₀::T=1.0, forgetful_c::T=0.5, ninit::Int=10) where {T<:AbstractFloat}

        L      = round(Int,β/Δτ)
        μ_traj = fill(init_μ, ninit)
        N_traj = fill(target_N, ninit)
        κ_traj = fill(κ₀, ninit)

        return new{T}(active,μ_traj,N_traj,κ_traj,forgetful_c,init_μ,β,Δτ,L,target_N,init_μ,target_N,1.0,0.0)
    end
end

""" 
Update array of μ values.
"""  
function update_μ!(μ::AbstractVector{T}, tuner::MuTuner{T},
                   R₁::AbstractVector{T}, M⁻¹R₁::AbstractVector{T},
                   R₂::AbstractVector{T}, M⁻¹R₂::AbstractVector{T})::T where {T<:AbstractFloat}

    μ₀ = mean(μ)
    @assert isapprox(μ₀,tuner.μ,atol=1e-6)

    if tuner.active

        # length of imaginary time axis
        L  = tuner.L

        # dimension of matrix
        NL = length(R₁)

        # measure ⟨N⟩= (2/L)Tr[I-M⁻¹]
        Nup = (NL-dot(R₁,M⁻¹R₁))/L
        Ndn = (NL-dot(R₂,M⁻¹R₂))/L
        N   = Nup + Ndn

        # measure ⟨N²⟩ = ((2/L)Tr[I-M⁻¹])² = (4/L²)Tr[I-M⁻¹]²
        N² = 4/L^2 * (NL-dot(R₁,M⁻¹R₁)) * (NL-dot(R₂,M⁻¹R₂))

        # update μ
        μ₁ = update_μ!(tuner,N,N²)
        Δμ = μ₁-μ₀
        @. μ += Δμ

        return μ₁
    else
        return μ₀
    end
end

""" 
Given a MuTuner, and a new set of measurements for N, N², updates the MuTuner and returns the new value of μ.
"""
function update_μ!(tuner::MuTuner, N::T, N²::T)::T where {T<:AbstractFloat}

    if tuner.active
        @unpack μ_traj, N_traj, κ_traj, forgetful_c, β, target_N = tuner

        κ = β * (N² - 2 * N * tuner.N̄ + tuner.N̄^2)
        push!(N_traj, N)
        push!(κ_traj, κ)

        tuner.μ̄ = forgetful_mean(μ_traj, forgetful_c, tuner.μ̄)
        tuner.N̄ = forgetful_mean(N_traj, forgetful_c, tuner.N̄)
        (tuner.κ̄, tuner.κ_var) = forgetful_mean_var(κ_traj, forgetful_c, tuner.κ̄, tuner.κ_var)

        new_μ = tuner.μ̄ + (target_N - tuner.N̄) / max(tuner.κ̄, sqrt(tuner.κ_var) / sqrt(length(κ_traj)))
        tuner.μ = new_μ
        push!(μ_traj, new_μ)

        return new_μ
    else
        return tuner.μ
    end
end

############################
## PRIVATE MODULE METHODS ##
############################

function forgetful_mean(data::Vector{T}, c::T) where {T<:AbstractFloat}

    cutoff = ceil(Int, (1.0 - c) * length(data))
    return mean(data[cutoff:end])
end

#= Constant-time "forgetful" mean and variance =# 
#= These assume that only one update has happened since the last call =#

function forgetful_mean(data::Vector{T}, c::T, prev_mean::T) where {T<:AbstractFloat}

    cutoff = ceil(Int, (1.0 - c) * length(data))
    prev_cutoff = ceil(Int, (1.0 - c) * (length(data) - 1))

    new_mean = (length(data) - prev_cutoff) * prev_mean
    if prev_cutoff != cutoff
        new_mean -= data[prev_cutoff]
    end
    new_mean += data[end]

    return new_mean / (length(data) - cutoff + 1)
end

function forgetful_var(data::Vector{T}, c::T) where {T<:AbstractFloat}

    cutoff = ceil(Int, (1.0 - c) * length(data))
    return var(data[cutoff:end]; corrected=true)
end

# Welford's online algorithm
function forgetful_mean_var(data::Vector{T}, c::T, prev_mean::T, prev_var::T) where {T<:AbstractFloat}

    cutoff = ceil(Int, (1.0 - c) * length(data))
    prev_cutoff = ceil(Int, (1.0 - c) * (length(data) - 1))

    new_pt = data[end]

    # Add the new point, update mean and sample variance
    new_length = length(data) - prev_cutoff + 1
    new_mean = prev_mean + (new_pt - prev_mean) / new_length
    new_var = prev_var * (new_length - 2) + (new_pt - prev_mean) * (new_pt - new_mean)
    new_var /= new_length - 1

    # If we need to drop a point off the back, update mean and sample variance again
    if prev_cutoff != cutoff
        new_length = length(data) - cutoff + 1
        drop_pt = data[prev_cutoff]
        prev_mean = new_mean
        prev_var = new_var

        new_mean = prev_mean - (drop_pt - prev_mean) / new_length
        new_var = prev_var * new_length - (drop_pt - prev_mean) * (drop_pt - new_mean)
        new_var /= new_length - 1
    end

    return (new_mean, new_var)
end

end