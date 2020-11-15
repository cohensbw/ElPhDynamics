module MuFinder

using Parameters
using Statistics
using LinearAlgebra
using Printf
using Statistics

using ..Models: AbstractModel, HolsteinModel, SSHModel, update_model!
using ..GreensFunctions: EstimateGreensFunction
using ..Measurements: measure_N², measure_density

export MuTuner, update_μ!

mutable struct MuTuner{T<:AbstractFloat}

    active      :: Bool
    μ_traj      :: Vector{T}
    N_traj      :: Vector{T}
    N²_traj     :: Vector{T}
    forgetful_c :: T
    μ           :: T
    N           :: Int
    β           :: T
    Δτ          :: T
    L           :: Int
    target_N    :: T
    μ_bar       :: T
    κ_bar       :: T
    N_bar       :: T
    N²_bar      :: T
    κ_min       :: T
    μ_avg       :: T
    μ_err       :: T
    log         :: Bool
    logfile     :: IOStream

    function MuTuner(active::Bool, init_μ::T, target_N::T, N::Int, β::T, Δτ::T, forgetful_c::T, κ_min::T, log::Bool, logfilename::String) where{T<:AbstractFloat}

        L       = round(Int,β/Δτ)
        μ_traj  = [init_μ]
        N_traj  = Vector{T}()
        N²_traj = Vector{T}()

        logfile = open(logfilename,"w")
        write(logfile,"mu_bar kappa_bar N_bar Nsqr_bar mu N Nsqr\n")

        return new{T}(active, μ_traj, N_traj, N²_traj, forgetful_c, init_μ, N, β, Δτ, L, target_N, init_μ, κ_min, -1.0, -1.0, κ_min, init_μ, 0.0, log, logfile)
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

    @unpack μ_traj, N_traj, N²_traj, forgetful_c, β, target_N, κ_min = tuner

    tuner.μ_bar = forgetful_mean(μ_traj, forgetful_c, tuner.μ_bar)

    push!(N_traj, N)
    tuner.N_bar = forgetful_mean(N_traj, forgetful_c, tuner.N_bar)

    push!(N²_traj, N²)
    tuner.N²_bar = forgetful_mean(N²_traj, forgetful_c, tuner.N²_bar)

    κ_bar = β * (tuner.N²_bar - tuner.N_bar^2)
    κ_update = max(κ_bar, κ_min / sqrt(length(N_traj)))
    tuner.κ_bar = κ_update

    new_μ = tuner.μ_bar + (target_N - tuner.N_bar) / κ_update
    tuner.μ = new_μ
    push!(μ_traj, new_μ)

    if tuner.active && tuner.log
        line = @sprintf("%5.f %.5f %.5f %.5f %.5f %.5f %.5f\n",tuner.μ_bar, tuner.κ_bar, tuner.N_bar, tuner.N²_bar, tuner.μ, N, N²)
        write(tuner.logfile, line)
    end

    return new_μ
end

"""
Given a MuTuner, returns the best guess for (μ, err_μ) from
previous trajectory.
"""
function estimate_μ(tuner::MuTuner{T}) where {T<:AbstractFloat}
    
    if tuner.active

        μ_bar = tuner.μ_bar

        # Run through and reconstruct the N_bar and N²_bar trajectories
        N_bar = tuner.N_traj[1]
        N²_bar = tuner.N²_traj[1]
        N_bar_traj = Vector{Float64}()
        N²_bar_traj = Vector{Float64}()
        μ_bar_traj = Vector{Float64}()
        sizehint!(N_bar_traj, length(tuner.N_traj))
        sizehint!(N²_bar_traj, length(tuner.N²_traj))
        sizehint!(μ_bar_traj, length(tuner.μ_traj))
        for i in 1:length(tuner.N_traj)
            N_bar = forgetful_mean((@view tuner.N_traj[1:i]), tuner.forgetful_c, N_bar)
            N²_bar = forgetful_mean((@view tuner.N²_traj[1:i]), tuner.forgetful_c, N²_bar)
            μ_bar = forgetful_mean((@view tuner.μ_traj[1:i]), tuner.forgetful_c, μ_bar)
            push!(N_bar_traj, N_bar)
            push!(N²_bar_traj, N²_bar)
            push!(μ_bar_traj, μ_bar)
        end

        κ_bar_traj    = @. tuner.β * (N²_bar_traj - N_bar_traj^2)
        μ_corrections = @. (tuner.target_N - N_bar_traj) / κ_bar_traj
        forgetful_idx = convert(Int, tuner.forgetful_c * length(μ_corrections))
        err_μ = stdm( view(μ_corrections,forgetful_idx:length(μ_corrections)), 0.0 )
        # err_μ = sqrt(mean(μ_corrections[forgetful_idx:end].^ 2))
        # err_μ = std(tuner.μ_traj[forgetful_idx:end])

        tuner.μ_avg = μ_bar
        tuner.μ_err = err_μ

        # write trajectories to logfile if not already written
        if !tuner.log
            # iterate over trajectory
            for i in 1:length(μ_bar_traj)
                # write data to log file
                line = @sprintf("%5.f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                                μ_bar_traj[i], κ_bar_traj[i], N_bar_traj[i], N²_bar_traj[i],
                                tuner.μ_traj[i], tuner.N_traj[i], tuner.N²_traj[i])
                write(tuner.logfile,line)
            end
        end
        close(tuner.logfile)

        return nothing
    else

        tuner.μ_avg = tuner.μ
        tuner.μ_err = 0.0
        return nothing
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