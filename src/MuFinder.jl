module MuFinder

using Parameters
using Statistics
using LinearAlgebra
using Printf
using Statistics

using ..Models: AbstractModel, HolsteinModel, SSHModel, update_model!
using ..GreensFunctions: EstimateGreensFunction, setup!
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
    μ_std       :: T
    κ_bar       :: T
    N_bar       :: T
    N_std       :: T
    N²_bar      :: T
    μ_bar_traj  :: Vector{T}
    κ_bar_traj  :: Vector{T}
    N_bar_traj  :: Vector{T}
    N²_bar_traj :: Vector{T}
    κ_min       :: T
    μ_avg       :: T
    μ_err       :: T
    logfile     :: String

    function MuTuner(active::Bool, init_μ::T, target_N::T, N::Int, β::T, Δτ::T, forgetful_c::T, κ_min::T, logfile::String) where{T<:AbstractFloat}

        L           = round(Int,β/Δτ)
        μ_traj      = [init_μ]
        N_traj      = Vector{T}()
        N²_traj     = Vector{T}()
        μ_bar_traj  = Vector{T}()
        κ_bar_traj  = Vector{T}()
        N_bar_traj  = Vector{T}()
        N²_bar_traj = Vector{T}()

        
        if !isfile(logfile) && active
            open(logfile,"w") do file
                write(file,"mu_bar kappa_bar n_bar Nsqr_bar mu n Nsqr\n")
            end
        end

        return new{T}(active, μ_traj, N_traj, N²_traj, forgetful_c, init_μ, N, β, Δτ, L, target_N, init_μ, 0.0, κ_min, -1.0, 0.0, -1.0,
                      μ_bar_traj, κ_bar_traj, N_bar_traj, N²_bar_traj, κ_min, init_μ, 0.0, logfile)
    end
end

"""
Update μ values in model.
"""
function update_μ!(model::AbstractModel{T},  tuner::MuTuner{T}, estimator::EstimateGreensFunction{T})::T where {T<:AbstractFloat}
    
    μ  = model.μ::Vector{T}
    μ₀ = mean(μ)

    if tuner.active

        # ⟨N⟩ and ⟨N²⟩ value
        N  = 0.0
        N² = 0.0

        # iterate over all possible pairs of random vectors
        for i in 1:(estimator.nᵥ-1)
            for j in (i+1):estimator.nᵥ

                # set up estimates
                setup!(estimator,i,j)

                # measure ⟨N⟩
                N  += model.Nsites * measure_density(model,estimator)

                # measure ⟨N²⟩
                N² += real(measure_N²(model,estimator))
            end
        end

        # normalize measurement
        N  /= binomial(estimator.nᵥ,2)
        N² /= binomial(estimator.nᵥ,2)

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

    @unpack μ_traj, N_traj, N²_traj, μ_bar_traj, N_bar_traj, N²_bar_traj, κ_bar_traj, forgetful_c, β, target_N, κ_min = tuner

    # record new ⟨N⟩ and ⟨N²⟩ values
    push!(N_traj,  N)
    push!(N²_traj, N²)

    # calculate new averages
    tuner.μ_bar, tuner.μ_std = forgetful_welfords(μ_traj, tuner.μ_bar, tuner.μ_std, forgetful_c)
    tuner.N_bar  = forgetful_mean(N_traj, tuner.N_bar, forgetful_c)
    tuner.N²_bar = forgetful_mean(N²_traj, tuner.N²_bar, forgetful_c)
    push!(μ_bar_traj,tuner.μ_bar)
    push!(N_bar_traj,tuner.N_bar)
    push!(N²_bar_traj,tuner.N²_bar)

    # length of trajectory
    n = length(N_traj)

    # variance of N
    varN = tuner.N²_bar - tuner.N_bar^2

    # calculate lower bound for κ
    κ_lo = κ_min/sqrt(n)

    # calculate upper bound for κ
    if n==1 || varN < 0.0 || tuner.μ_std <= 0.0
        κ_hi = κ_lo
    else
        κ_hi = sqrt(varN)/tuner.μ_std
    end

    # calculate κ
    tuner.κ_bar  = β * varN

    # apply bounds to κ value
    tuner.κ_bar  = min( tuner.κ_bar , κ_hi )
    tuner.κ_bar  = max( tuner.κ_bar , κ_lo )
    push!(κ_bar_traj,tuner.κ_bar)

    # write to log file
    if tuner.active
        open(tuner.logfile,"a") do file
            @printf file "%.8f %.8f %.8f %.8f %.8f %.8f %.8f\n" tuner.μ_bar (tuner.κ_bar/tuner.N) (tuner.N_bar/tuner.N) tuner.N²_bar tuner.μ (N/tuner.N) N²
        end
    end

    # calculate new μ value
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

        # forgetfulnes parameter
        forgetful_c = tuner.forgetful_c

        # if entire history is used in mu-tuning, then only use
        # half the history to estimate μ
        if forgetful_c == 1.0
            forgetful_c = 0.5
        end

        # calculate best estimate for true μ value
        forgetful_idx = ceil(Int, forgetful_c * length(tuner.μ_traj))
        μ_traj        = @view tuner.μ_traj[forgetful_idx:length(tuner.μ_traj)]
        μ_err         = stdm( μ_traj , median(μ_traj) )
        tuner.μ_avg   = tuner.μ_bar
        tuner.μ_err   = μ_err

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

"""
Update the mean of the vector `x` using only the most recent `c` fraction of the history.
"""
function forgetful_mean(x::AbstractVector{T}, x̄ₙ₋₁::T, c::T)::T where {T<:AbstractFloat}

    N = length(x)
    if N  == 1
        return x[1]
    end
    i′   = 1 + floor(Int, (1.0 - c) * (N - 1))
    n    = N - i′ + 1
    xₙ   = x[N]
    x̄ₙ   = x̄ₙ₋₁ + (xₙ-x̄ₙ₋₁)/n
    i    = 1 + floor(Int, (1.0 - c) * N)
    if i != i′
        n  = N - i + 1
        x₀ = x[i′]
        x̄ₙ = x̄ₙ - (x₀-x̄ₙ)/n
    end
    return x̄ₙ
end

"""
Update the mean and standard deviation of the vector `x` using only the most recent `c` fraction of the history.
Uses the numerically stable Welford's algorithm to update the standard deviation.
"""
function forgetful_welfords(x::AbstractVector{T},x̄ₙ₋₁::T,sₙ₋₁::T,c::T)::Tuple{T,T} where {T<:AbstractFloat}

    N = length(x)
    if N == 1
        return (x[1], 0.0)
    end
    i′   = 1 + floor(Int, (1.0 - c) * (N - 1))
    n    = N - i′ + 1
    xₙ   = x[N]
    Mₙ₋₁ = (n-2)*sₙ₋₁^2
    x̄ₙ   = x̄ₙ₋₁ + (xₙ-x̄ₙ₋₁)/n
    Mₙ   = Mₙ₋₁ + (xₙ-x̄ₙ)*(xₙ-x̄ₙ₋₁)
    i    = 1 + floor(Int, (1.0 - c) * N)
    if i != i′
        n  = N - i + 1
        x₀ = x[i′]
        x̄′ = x̄ₙ
        x̄ₙ = x̄′ - (x₀-x̄′)/n
        Mₙ = Mₙ - (x₀-x̄ₙ)*(x₀-x̄′)
    end
    if n>1
        sₙ = sqrt(Mₙ/(n-1))
    else
        sₙ = 0.0
    end

    return (x̄ₙ, sₙ)
end

end