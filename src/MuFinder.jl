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
        if active
            write(logfile,"mu_bar kappa_bar N_bar Nsqr_bar mu N Nsqr\n")
        else
            close(logfile)
        end

        return new{T}(active, μ_traj, N_traj, N²_traj, forgetful_c, init_μ, N, β, Δτ, L, target_N, init_μ, κ_min, -1.0, -1.0, κ_min, init_μ, 0.0, log, logfile)
    end
end

"""
Update μ values in model.
""" 
function update_μ!(model::AbstractModel{T},  tuner::MuTuner{T}, estimator::EstimateGreensFunction{T})::T where {T<:AbstractFloat}
    
    μ  = model.μ::Vector{T}
    μ₀ = mean(μ)

    if tuner.active

        # measure ⟨N⟩
        n = measure_density(model,estimator)
        N = n * model.Nsites

        # measure ⟨N²⟩
        N² = real(measure_N²(model,estimator))

        # update μ
        μ₁      = update_μ!(tuner,N,N²)
        Δμ      = μ₁-μ₀
        @. μ   += Δμ
        tuner.μ = μ₁

        return μ₁
    else

        tuner.μ = μ₀

        return μ₀
    end
end

""" 
Given new measurements for N and N², updates the MuTuner and returns the new value of μ.
"""
function update_μ!(tuner::MuTuner, N::T, N²::T)::T where {T<:AbstractFloat}

    @unpack μ_traj, N_traj, N²_traj, forgetful_c, β, target_N, κ_min = tuner

    push!(N_traj,  N)
    push!(N²_traj, N²)
    tuner.μ_bar  = forgetful_mean(μ_traj,  forgetful_c, tuner.μ_bar)
    tuner.N_bar  = forgetful_mean(N_traj,  forgetful_c, tuner.N_bar)
    tuner.N²_bar = forgetful_mean(N²_traj, forgetful_c, tuner.N²_bar)
    tuner.κ_bar  = max( β*(tuner.N²_bar - tuner.N_bar^2) , κ_min/sqrt(length(N_traj)) )

    if tuner.active && tuner.log
        line = @sprintf("%5.f %.5f %.5f %.5f %.5f %.5f %.5f\n", tuner.μ_bar, tuner.κ_bar, tuner.N_bar, tuner.N²_bar, tuner.μ, N, N²)
        write(tuner.logfile, line)
        flush(tuner.logfile)
    end

    tuner.μ = tuner.μ_bar + (target_N - tuner.N_bar) / tuner.κ_bar 
    push!(μ_traj, tuner.μ)

    return tuner.μ
end

"""
Given a MuTuner, returns the best guess for (μ, err_μ) from
previous trajectory.
"""
function estimate_μ(tuner::MuTuner{T}) where {T<:AbstractFloat}
    
    if tuner.active

        forgetful_idx = ceil(Int, tuner.forgetful_c * length(tuner.μ_traj))
        μ_traj        = @view tuner.μ_traj[forgetful_idx:length(tuner.μ_traj)]
        μ_err         = std( μ_traj )
        tuner.μ_avg   = tuner.μ_bar
        tuner.μ_err   = μ_err

        # write trajectories to logfile if not already written
        if !tuner.log

            # Run through and reconstruct the N_bar, N²_bar, μ_bar, and κ_bar trajectories
            N_bar       = tuner.N_traj[1]
            N²_bar      = tuner.N²_traj[1]
            μ_bar       = tuner.μ_traj[1]
            N_bar_traj  = Vector{T}()
            N²_bar_traj = Vector{T}()
            μ_bar_traj  = Vector{T}()
            κ_bar_traj  = Vector{T}()
            sizehint!(N_bar_traj,  length(tuner.N_traj))
            sizehint!(N²_bar_traj, length(tuner.N_traj))
            sizehint!(μ_bar_traj,  length(tuner.N_traj))
            sizehint!(κ_bar_traj,  length(tuner.N_traj))
            for i in 1:length(tuner.N_traj)
                N_bar  = forgetful_mean( view(tuner.N_traj, 1:i), tuner.forgetful_c, N_bar)
                N²_bar = forgetful_mean( view(tuner.N²_traj,1:i), tuner.forgetful_c, N²_bar)
                μ_bar  = forgetful_mean( view(tuner.μ_traj, 1:i), tuner.forgetful_c, μ_bar)
                κ_bar  = max( tuner.β*(N²_bar - N_bar^2) , tuner.κ_min/sqrt(i) )
                push!(N_bar_traj,  N_bar)
                push!(N²_bar_traj, N²_bar)
                push!(μ_bar_traj,  μ_bar)
                push!(κ_bar_traj,  κ_bar)
            end

            # iterate over trajectory
            for i in 1:length(tuner.N_traj)
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