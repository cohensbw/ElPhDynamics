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
    κ_min       :: T

    function MuTuner(active::Bool, init_μ::T, target_N::T, β::T, Δτ::T, forgetful_c::T, κ_min::T=0.1) where{T<:AbstractFloat}

        L      = round(Int,β/Δτ)
        μ_traj = [init_μ]
        N_traj = Vector{T}()
        κ_traj = Vector{T}()

        # println("init Mu = ", init_μ)

        return new{T}(active, μ_traj, N_traj, κ_traj, forgetful_c, init_μ, β, Δτ, L, target_N, init_μ, -1.0, 0.0, κ_min)
    end
end

""" 
Update array of μ values.
"""  
function update_μ!(μ::AbstractVector{T}, tuner::MuTuner{T},
                   R₁::AbstractVector{T}, M⁻¹R₁::AbstractVector{T},
                   R₂::AbstractVector{T}, M⁻¹R₂::AbstractVector{T})::T where {T<:AbstractFloat}
    if tuner.active

        # get mean chemical potential
        μ₀ = mean(μ)
        @assert isapprox(μ₀,tuner.μ,atol=1e-6)

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
function update_μ!(tuner::MuTuner, N::T, N²::T)::T  where{T<:AbstractFloat}

    @unpack μ_traj, N_traj, κ_traj, forgetful_c, β, target_N, κ_min = tuner

    tuner.μ̄ = forgetful_mean(μ_traj, forgetful_c, tuner.μ̄ )

    push!(N_traj, N)
    tuner.N̄ = forgetful_mean(N_traj, forgetful_c, tuner.N̄)

    κ = β * (N² - 2 * N * tuner.N̄ + tuner.N̄^2)
    push!(κ_traj, κ)
    tuner.κ̄ = forgetful_mean(κ_traj, forgetful_c, tuner.κ̄ )
    κ_update = max(tuner.κ̄, κ_min / sqrt(length(κ_traj)))

    new_μ = tuner.μ̄  + (target_N - tuner.N̄) / κ_update
    tuner.μ = new_μ
    push!(μ_traj, new_μ)

    # println("new mu = ",new_μ)

    return new_μ
end

############################
## PRIVATE MODULE METHODS ##
############################

function forgetful_mean(data::Vector{T}, c::T, prev_mean::T)::T  where{T<:AbstractFloat}

    if length(data) == 1
        return data[1]
    end

    cutoff = ceil(Int64, (1.0 - c) * length(data))
    prev_cutoff = ceil(Int64, (1.0 - c) * (length(data) - 1))

    new_mean = (length(data) - prev_cutoff) * prev_mean
    if prev_cutoff != cutoff
        new_mean -= data[prev_cutoff]
    end
    new_mean += data[end]

    return new_mean / (length(data) - cutoff + 1)
end

end