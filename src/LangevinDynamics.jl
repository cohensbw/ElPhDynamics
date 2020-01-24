module LangevinDynamics

using IterativeSolvers
using Random
import LinearAlgebra: mul!, ldiv!

using ..Utilities: get_index
using ..HolsteinModels: HolsteinModel, construct_expnΔτV!, mulMᵀ!, muldMdϕ!
using ..PhononAction: calc_dSbosedϕ!
using ..FourierAcceleration: FourierAccelerator, forward_fft!, inverse_fft!, accelerate!, accelerate_noise!

export update_euler!, update_rk!, update_euler_fa!, update_rk_fa!

"""
Update phonon fields using Runge-Kutta/Heun's equation and fourier acceleration.
"""
function update_rk_fa!(holstein::HolsteinModel{T1,T2}, fa::FourierAccelerator{T1},
                       dϕ::AbstractVector{T1}, fft_dϕ::AbstractVector{Complex{T1}},
                       dSdϕ2::AbstractVector{T2}, dSdϕ1::AbstractVector{T2}, fft_dSdϕ::AbstractVector{Complex{T1}},
                       g::AbstractVector{T2}, M⁻¹g::AbstractVector{T2},
                       η::AbstractVector{T1}, fft_η::AbstractVector{Complex{T1}},
                       Δt::T1, preconditioner=Identity())::Int where {T1<:AbstractFloat,T2<:Number}
    
    # itialize η as vector of gaussian random number
    randn!(η)

    # calculate dSdϕ = [∂S/∂ϕ₁(1),...,∂S/∂ϕₙ(1),...,∂S/∂ϕ₁(τ),...,∂S/∂ϕₙ(τ),...,∂S/∂ϕ₁(Lτ),...,∂S/∂ϕₙ(Lτ)]
    iters = calc_dSdϕ!(dSdϕ1, g, M⁻¹g, holstein, preconditioner)

    # get the update for the fields using euler method
    @. dϕ = sqrt(2*Δt)*η - Δt*real(dSdϕ1)

    # update phonon fields
    @. holstein.ϕ += dϕ
 
    # update the exponentiated interaction matrix so that it reflects the current
    # phonon field configuration.
    construct_expnΔτV!(holstein)

    # calculate dSdϕ = [∂S/∂ϕ₁(1),...,∂S/∂ϕₙ(1),...,∂S/∂ϕ₁(τ),...,∂S/∂ϕₙ(τ),...,∂S/∂ϕ₁(Lτ),...,∂S/∂ϕₙ(Lτ)]
    iters = calc_dSdϕ!(dSdϕ2, g, M⁻¹g, holstein, preconditioner)

    # revert back to original phonon fields
    @. holstein.ϕ -= dϕ

    # update the exponentiated interaction matrix so that it reflects the current
    # phonon field configuration.
    construct_expnΔτV!(holstein)

    # get the partial derivative for the RK step
    @. dSdϕ2 = (dSdϕ2+dSdϕ1)/2.0

    # fourier transform dSdϕ2
    forward_fft!( fft_dSdϕ , dSdϕ2 , fa)

    # accelerate fft_dSdϕ ==> Q⋅fft_dSdϕ
    accelerate!( fft_dSdϕ , fa )

    # fourier transform η
    forward_fft!( fft_η , η , fa )

    # accelerate noise vector fft_η ==> √(2Q)⋅fft_η
    accelerate_noise!( fft_η , fa )

    # calculate fft_dϕ
    @. fft_dϕ = fft_η - fft_dSdϕ

    # perform inverse fourier transform to get dϕ
    inverse_fft!( dϕ , fft_dϕ , fa )

    # update phonon fields
    @. holstein.ϕ += dϕ

    # update the exponentiated interaction matrix so that it reflects the current
    # phonon field configuration.
    construct_expnΔτV!(holstein)

    return iters
end


"""
Update phonon fields using Euler equation and fourier acceleration.
"""
function update_euler_fa!(holstein::HolsteinModel{T1,T2}, fa::FourierAccelerator{T1},
                          dϕ::AbstractVector{T1}, fft_dϕ::AbstractVector{Complex{T1}},
                          dSdϕ::AbstractVector{T2}, fft_dSdϕ::AbstractVector{Complex{T1}},
                          g::AbstractVector{T2}, M⁻¹g::AbstractVector{T2},
                          η::AbstractVector{T1}, fft_η::AbstractVector{Complex{T1}},
                          Δt::T1, preconditioner=Identity())::Int where {T1<:AbstractFloat,T2<:Number}
    
    # itialize η as vector of gaussian random number
    randn!(η)

    # calculate dSdϕ = [∂S/∂ϕ₁(1),...,∂S/∂ϕₙ(1),...,∂S/∂ϕ₁(τ),...,∂S/∂ϕₙ(τ),...,∂S/∂ϕ₁(Lτ),...,∂S/∂ϕₙ(Lτ)]
    iters = calc_dSdϕ!(dSdϕ, g, M⁻¹g, holstein, preconditioner)

    # fourier transform dSdϕ
    forward_fft!( fft_dSdϕ , dSdϕ , fa)

    # accelerate fft_dSdϕ ==> Q⋅fft_dSdϕ
    accelerate!( fft_dSdϕ , fa )

    # fourier transform η
    forward_fft!( fft_η , η , fa )

    # accelerate fft_η ==> √(2Q)⋅fft_η
    accelerate_noise!( fft_η , fa )

    # calculate fft_dϕ
    @. fft_dϕ = fft_η - fft_dSdϕ

    # perform inverse fourier transform to get dϕ
    inverse_fft!( dϕ , fft_dϕ , fa )

    # update phonon fields
    @. holstein.ϕ += dϕ

    # update the exponentiated interaction matrix so that it reflects the current
    # phonon field configuration.
    construct_expnΔτV!(holstein)

    return iters
end

################################################
## PRIVATE FUNCTIONS JUST USED IN THIS SCRIPT ##
################################################

"""
    function calc_dSdϕ!(dSdϕ::AbstractVector{T1},g::AbstractVector{T1},Mᵀg::AbstractVector{T1},M⁻¹g::AbstractVector{T1},holstein::HolsteinModel{T1,T2},tol::AbstractFloat)::Int where {T1<:AbstractFloat,T2<:Number}

Calculates all of the partial derivatives ∂S/∂ϕᵢ(τ) and stores each partial derivative in a vector dSdϕ.
The expression we are evaluating is `∂S/∂ϕᵢ(τ) = ∂Sbose/∂ϕᵢ(τ) - 2gᵀ[∂M/∂ϕᵢ(τ)]⋅M⁻¹g}`.
# Arguments
- `dSdϕ::AbstractVector{T1}`: vector the will be modified to contain all the partial derivatives ∂S/∂ϕᵢ(τ)
- `g::AbstractVector{T1}`: A random vector.
- `Mᵀg::AbstractVector{T1}`: Vector containing the product Mᵀg
- `M⁻¹g::AbstractVector{T1}`: A vector the will contain the solution of the linear equation M⋅v=g when computed.
- `holstein::HolsteinModel{T1,T2}`: Type represent holstein model being simulated.
- `tol::AbstractFloat`: The tolerance used when solving the linear equation M⋅v=g in order to get M⁻¹g.
# Returns
- `iters::Int`: Number of iterations used to solve for M⁻¹g.
"""
function calc_dSdϕ!(dSdϕ::AbstractVector{T2},g::AbstractVector{T2},M⁻¹g::AbstractVector{T2},
                    holstein::HolsteinModel{T1,T2}, preconditioner=Identity())::Int where {T1<:AbstractFloat,T2<:Number}

    # NOTE: The method I use below for calculating all of the partial derivatives {∂S/∂ϕᵢ(τ)} works only
    # because I am assuming a strictly local form of the electron-phonon interaction given by ∑(λᵢ⋅ϕᵢ⋅nᵢ).
    # If a longer-range electron-phonon interaction were added of the form ∑(λᵢⱼ⋅ϕᵢ⋅nⱼ), then this way of doing
    # things would no longer work, and a loop over each individual phonon field ϕᵢ(τ) would need to be added.

    # intialize random vector g.
    @inbounds @fastmath for i in 1:length(g)
        g[i] = randn()
    end

    # solve MᵀM⋅v = Mᵀg ==> v = M⁻¹g.
    iters = ldiv!(M⁻¹g,holstein,g,preconditioner)

    # ∂S/∂ϕᵢ(τ) = ∂M/∂ϕᵢ(τ)⋅M⁻¹g
    muldMdϕ!( dSdϕ , holstein , M⁻¹g )

    # ∂S/∂ϕᵢ(τ) = -2gᵀ⋅∂M/∂ϕᵢ(τ)⋅M⁻¹g
    @. g *= -2.0 * dSdϕ
    # iterate over sites
    @inbounds @fastmath for site in 1:holstein.nsites
        # iterate over time slices
        for τ in 1:holstein.Lτ
            idx_τ   = get_index(τ,               site, holstein.Lτ)
            idx_τp1 = get_index(τ%holstein.Lτ+1, site, holstein.Lτ)
            # shifting values one time slice forward
            dSdϕ[idx_τp1] = real(g[idx_τ])
        end
    end
    # In the lines of code above there is a subtle detail that is addressed.
    # After doing the element-wise multiplication, the expectation value for the partial
    # derivatives corresponding to the τ time slice live in the array indices corresponding to τ-1.
    # Therefore, the values need to be shifted one time slice forward. This is done by first calculating
    # the and storing the ∂S/∂ϕᵢ(τ) derivative values in the vector g, and then copying a proper shifted
    # version into the vector dSdϕ.

    # ∂S/∂ϕᵢ(τ) = ∂Sbose/∂ϕᵢ(τ) - 2gᵀ⋅[∂M/∂ϕᵢ(τ)]⋅M⁻¹g ==> All Done!
    calc_dSbosedϕ!( dSdϕ , holstein )

    # returning number of iterations in solving for M⁻¹g
    return iters
end

end