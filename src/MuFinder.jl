module MuFinder

using Parameters
using Statistics
using LinearAlgebra

using ..Models: AbstractModel, HolsteinModel, SSHModel, update_model!
using ..GreensFunctions: EstimateGreensFunction
using ..Measurements: measure_N², measure_density

export MuTuner, update_μ!

mutable struct MuTuner{T<:AbstractFloat}

    active      :: Bool
    μ_traj      :: Vector{T}
    N_traj      :: Vector{T}
    κ_traj      :: Vector{T}
    forgetful_c :: T
    μ           :: T
    N           :: Int
    β           :: T
    Δτ          :: T
    L           :: Int
    target_N    :: T
    μ_bar       :: T
    N_bar       :: T
    κ_bar       :: T
    κ_min       :: T

    function MuTuner(active::Bool, init_μ::T, target_N::T, N::Int, β::T, Δτ::T, forgetful_c::T, κ_min::T=0.1) where{T<:AbstractFloat}

        L      = round(Int,β/Δτ)
        μ_traj = [init_μ]
        N_traj = Vector{T}()
        κ_traj = Vector{T}()

        return new{T}(active, μ_traj, N_traj, κ_traj, forgetful_c, init_μ, N, β, Δτ, L, target_N, init_μ, -1.0, 0.0, κ_min)
    end
end

"""
Update μ values in model.
"""
function update_μ!(model::AbstractModel{T},  tuner::MuTuner{T}, estimator::EstimateGreensFunction{T})::T where {T}

    μ_new = update_μ!(model.μ, tuner, model, estimator)
    update_model!(model)

    return μ_new
end


""" 
Update array of μ values.
"""  
function update_μ!(μ::AbstractVector{T},  tuner::MuTuner{T}, model::AbstractModel{T}, estimator::EstimateGreensFunction{T})::T where {T}
    
    μ₀ = mean(μ)

    if tuner.active

        # measure ⟨N⟩
        n = measure_density(model,estimator)
        N = n * model.Nsites

        # measure ⟨N²⟩
        N² = real(measure_N²(model,estimator))

        # update μ
        μ₁ = update_μ!(tuner,N,N²)
        Δμ = μ₁-μ₀
        @. μ += Δμ

        return μ₁
    else

        tuner.μ = μ₀

        return μ₀
    end
end

""" 
Given a MuTuner, and a new set of measurements for N, N², updates the MuTuner and returns the new value of μ.
"""
function update_μ!(tuner::MuTuner, N::T, N²::T)::T where {T<:AbstractFloat}

    @unpack μ_traj, N_traj, κ_traj, forgetful_c, β, target_N, κ_min = tuner

    tuner.μ_bar = forgetful_mean(μ_traj, forgetful_c, tuner.μ_bar)

    push!(N_traj, N)
    tuner.N_bar = forgetful_mean(N_traj, forgetful_c, tuner.N_bar)

    κ = β * (N² - 2 * N * tuner.N_bar + tuner.N_bar^2)
    push!(κ_traj, κ)
    tuner.κ_bar = forgetful_mean(κ_traj, forgetful_c, tuner.κ_bar )
    κ_update = max(tuner.κ_bar, κ_min / sqrt(length(κ_traj)))

    new_μ = tuner.μ_bar + (target_N - tuner.N_bar) / κ_update
    tuner.μ = new_μ
    push!(μ_traj, new_μ)

    return new_μ
end

"""
Given a MuTuner, returns the best guess for (μ, err_μ) from
previous trajectory.
"""
function estimate_μ(tuner::MuTuner{T})::Tuple{T,T} where {T<:AbstractFloat}
    
    if tuner.active

        μ_bar = tuner.μ_bar

        # Run through and reconstruct the N̄ and κ̄ trajectories
        N_bar = tuner.N_traj[1]
        κ_bar = tuner.κ_traj[1]
        N_bar_traj = Vector{Float64}()
        κ_bar_traj = Vector{Float64}()
        sizehint!(N_bar_traj, length(tuner.N_traj))
        sizehint!(κ_bar_traj, length(tuner.κ_traj))
        for i in 1:length(tuner.N_traj)
            N_bar = forgetful_mean((@view tuner.N_traj[1:i]), tuner.forgetful_c, N_bar)
            κ_bar = forgetful_mean((@view tuner.κ_traj[1:i]), tuner.forgetful_c, κ_bar)
            push!(N_bar_traj, N_bar)
            push!(κ_bar_traj, κ_bar)
        end

        μ_corrections = (tuner.target_N .- N_bar_traj) ./ κ_bar_traj
        forgetful_idx = convert(Int, tuner.forgetful_c * length(μ_corrections))
        err_μ = sqrt(mean(μ_corrections[forgetful_idx:end] .^ 2))

        return (μ_bar, err_μ)
    else

        return (tuner.μ, 0.0)
    end
end

############################
## PRIVATE MODULE METHODS ##
############################

function forgetful_mean(data::AbstractVector{T}, c::T, prev_mean::T)::T where {T<:AbstractFloat}

    if length(data) == 1
        return data[1]
    end

    cutoff = ceil(Int, (1.0 - c) * length(data))
    prev_cutoff = ceil(Int, (1.0 - c) * (length(data) - 1))

    new_mean = (length(data) - prev_cutoff) * prev_mean
    if prev_cutoff != cutoff
        new_mean -= data[prev_cutoff]
    end
    new_mean += data[end]

    return new_mean / (length(data) - cutoff + 1)
end

end