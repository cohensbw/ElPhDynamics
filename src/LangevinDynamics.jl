module LangevinDynamics

using IterativeSolvers
using Random
using LinearAlgebra: mul!
using Langevin.HolsteinModels: HolsteinModel, construct_expnΔτV!, mulMᵀ!, muldMdϕ!
using Langevin.PhononAction: calc_dSbosedϕ!
using Langevin.FourierAcceleration: FourierAccelerator, forward_fft!, inverse_fft!, accelerate!, accelerate_noise!

export update_euler!, update_rk!, update_euler_fa!, update_rk_fa!

"""
Update phonon fields using Runge-Kutta/Heun's equation and fourier acceleration.
"""
function update_rk_fa!(holstein::HolsteinModel{T1,T2}, fa::FourierAccelerator{T1},
                       dϕdt::AbstractVector{T1}, fft_dϕdt::AbstractVector{Complex{T1}},
                       dSdϕ2::AbstractVector{T1}, dSdϕ1::AbstractVector{T1}, fft_dSdϕ::AbstractVector{Complex{T1}},
                       g::AbstractVector{T1}, Mᵀg::AbstractVector{T1}, M⁻¹g::AbstractVector{T1},
                       η::AbstractVector{T1}, fft_η::AbstractVector{Complex{T1}},
                       Δt::AbstractFloat, tol::AbstractFloat)::Int where {T1<:AbstractFloat,T2<:Number}
    
    # itialize η as vector of gaussian random number
    randn!(η)

    # calculate dSdϕ = [∂S/∂ϕ₁(1),...,∂S/∂ϕₙ(1),...,∂S/∂ϕ₁(τ),...,∂S/∂ϕₙ(τ),...,∂S/∂ϕ₁(Lτ),...,∂S/∂ϕₙ(Lτ)]
    iters = calc_dSdϕ!(dSdϕ1, g, Mᵀg, M⁻¹g, holstein, tol)

    # tentatively update fields using euler method
    @. holstein.ϕ -= sqrt(2.0*Δt)*η + Δt*dSdϕ1

    # calculate dSdϕ = [∂S/∂ϕ₁(1),...,∂S/∂ϕₙ(1),...,∂S/∂ϕ₁(τ),...,∂S/∂ϕₙ(τ),...,∂S/∂ϕ₁(Lτ),...,∂S/∂ϕₙ(Lτ)]
    iters = calc_dSdϕ!(dSdϕ2, g, Mᵀg, M⁻¹g, holstein, tol)
    @. dSdϕ2 = ( dSdϕ1 - dSdϕ2 ) / 2.0

    # fourier transform dSdϕ
    forward_fft!( fft_dSdϕ , dSdϕ2 , fa)

    # accelerate fft_dSdϕ ==> Q⋅fft_dSdϕ
    accelerate!( fft_dSdϕ , fa )

    # fourier transform η
    forward_fft!( fft_η , η , fa )

    # accelerate fft_η ==> √(2Q)⋅fft_η
    accelerate_noise!( fft_η , fa )

    # calculate fft_dϕdt
    @. fft_dϕdt = 2.0*fft_η + fft_dSdϕ

    # perform inverse fourier transform to get dϕdt
    inverse_fft!( dϕdt , fft_dϕdt , fa )

    # update phonon fields
    @. holstein.ϕ += dϕdt

    return iters
end


"""
Update phonon fields using Euler equation and fourier acceleration.
"""
function update_euler_fa!(holstein::HolsteinModel{T1,T2}, fa::FourierAccelerator{T1},
                          dϕdt::AbstractVector{T1}, fft_dϕdt::AbstractVector{Complex{T1}},
                          dSdϕ::AbstractVector{T1}, fft_dSdϕ::AbstractVector{Complex{T1}},
                          g::AbstractVector{T1}, Mᵀg::AbstractVector{T1}, M⁻¹g::AbstractVector{T1},
                          η::AbstractVector{T1}, fft_η::AbstractVector{Complex{T1}},
                          Δt::AbstractFloat, tol::AbstractFloat)::Int where {T1<:AbstractFloat,T2<:Number}
    
    # itialize η as vector of gaussian random number
    randn!(η)

    # calculate dSdϕ = [∂S/∂ϕ₁(1),...,∂S/∂ϕₙ(1),...,∂S/∂ϕ₁(τ),...,∂S/∂ϕₙ(τ),...,∂S/∂ϕ₁(Lτ),...,∂S/∂ϕₙ(Lτ)]
    iters = calc_dSdϕ!(dSdϕ, g, Mᵀg, M⁻¹g, holstein, tol)

    # fourier transform dSdϕ
    forward_fft!( fft_dSdϕ , dSdϕ , fa)

    # accelerate fft_dSdϕ ==> Q⋅fft_dSdϕ
    accelerate!( fft_dSdϕ , fa )

    # fourier transform η
    forward_fft!( fft_η , η , fa )

    # accelerate fft_η ==> √(2Q)⋅fft_η
    accelerate_noise!( fft_η , fa )

    # calculate fft_dϕdt
    @. fft_dϕdt = fft_η - fft_dSdϕ

    # perform inverse fourier transform to get dϕdt
    inverse_fft!( dϕdt , fft_dϕdt , fa )

    # update phonon fields
    @. holstein.ϕ += dϕdt

    return iters
end


"""
Update phonon fields using Runge-Kutta/Heun's equation and fourier acceleration.
"""
function update_rk!(holstein::HolsteinModel{T1,T2}, dSdϕ2::AbstractVector{T1}, dSdϕ1::AbstractVector{T1},
                    g::AbstractVector{T1}, Mᵀg::AbstractVector{T1}, M⁻¹g::AbstractVector{T1},
                    η::AbstractVector{T1}, Δt::AbstractFloat, tol::AbstractFloat)::Int where {T1<:AbstractFloat,T2<:Number}
    
    # itialize η as vector of gaussian random number
    randn!(η)

    # calculate dSdϕ1 = [∂S/∂ϕ₁(1),...,∂S/∂ϕₙ(1),...,∂S/∂ϕ₁(τ),...,∂S/∂ϕₙ(τ),...,∂S/∂ϕ₁(Lτ),...,∂S/∂ϕₙ(Lτ)]
    iters = calc_dSdϕ!(dSdϕ1, g, Mᵀg, M⁻¹g, holstein, tol)

    # tentatively update fields using euler method
    @. holstein.ϕ += -sqrt(2.0*Δt)*η - Δt*dSdϕ1

    # calculate dSdϕ2 = [∂S/∂ϕ₁(1),...,∂S/∂ϕₙ(1),...,∂S/∂ϕ₁(τ),...,∂S/∂ϕₙ(τ),...,∂S/∂ϕ₁(Lτ),...,∂S/∂ϕₙ(Lτ)]
    iters = calc_dSdϕ!(dSdϕ2, g, Mᵀg, M⁻¹g, holstein, tol)

    # final update of phonon field
    @. holstein.ϕ += 2*sqrt(2*Δt)*η + Δt/2*(dSdϕ1-dSdϕ2)

    return iters
end


"""
Update phonon fields using Euler equation.
"""
function update_euler!(holstein::HolsteinModel{T1,T2}, dSdϕ::AbstractVector{T1},
                       g::AbstractVector{T1}, Mᵀg::AbstractVector{T1}, M⁻¹g::AbstractVector{T1},
                       η::AbstractVector{T1}, Δt::AbstractFloat, tol::AbstractFloat)::Int where {T1<:AbstractFloat,T2<:Number}
    
    # itialize η as vector of gaussian random number
    randn!(η)

    # calculate dSdϕ = [∂S/∂ϕ₁(1),...,∂S/∂ϕₙ(1),...,∂S/∂ϕ₁(τ),...,∂S/∂ϕₙ(τ),...,∂S/∂ϕ₁(Lτ),...,∂S/∂ϕₙ(Lτ)]
    iters = calc_dSdϕ!(dSdϕ, g, Mᵀg, M⁻¹g, holstein, tol)

    # update phonon fields without fourier acceleration
    @. holstein.ϕ += sqrt(2.0*Δt)*η - Δt*dSdϕ

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
function calc_dSdϕ!(dSdϕ::AbstractVector{T1},g::AbstractVector{T1},Mᵀg::AbstractVector{T1},M⁻¹g::AbstractVector{T1},
                    holstein::HolsteinModel{T1,T2},tol::AbstractFloat)::Int where {T1<:AbstractFloat,T2<:Number}

    # NOTE: The method I use below for calculating all of the partial derivatives {∂S/∂ϕᵢ(τ)} works only
    # because I am assuming a strictly local form of the electron-phonon interaction given by ∑(λᵢ⋅ϕᵢ⋅nᵢ).
    # If a longer-range electron-phonon interaction were added of the form ∑(λᵢⱼ⋅ϕᵢ⋅nⱼ), then this way of doing
    # things would no longer work, and a loop over each individual phonon field ϕᵢ(τ) would need to be added.

    # first update the exponentiated interaction matrix so that it reflects the current
    # phonon field configuration.
    construct_expnΔτV!(holstein)

    # intialize random vector g.
    randn!(g)
    #rand!(g,-1:2:1)

    # getting Mᵀg.
    mulMᵀ!( Mᵀg , holstein , g )

    # intialize M⁻¹g to vector of zeros.
    M⁻¹g .= 0.0

    # solve MᵀM⋅v = Mᵀg ==> v = M⁻¹g.
    info = cg!( M⁻¹g , holstein , Mᵀg , tol=tol , log=true , statevars=holstein.cg_state_vars , initially_zero=true )[2]
    # info = minres!( M⁻¹g , holstein , Mᵀg , tol=tol , log=true , initially_zero=true )[2]

    # ∂S/∂ϕᵢ(τ) = ∂M/∂ϕᵢ(τ)⋅M⁻¹g
    muldMdϕ!( dSdϕ , holstein , M⁻¹g )

    # ∂S/∂ϕᵢ(τ) = -2gᵀ⋅∂M/∂ϕᵢ(τ)⋅M⁻¹g
    @. g *= -2.0 * dSdϕ
    circshift!( dSdϕ , g , holstein.nsites )
    # In the line above there is a subtle detail that is addressed.
    # After doing the element-wise multiplication, the expectation value for the partial
    # derivatives corresponding to the τ time slice lives in the array indices corresponding to τ-1.
    # Therefore, all the values need to be circularly shifted nsites in the array.

    # ∂S/∂ϕᵢ(τ) = ∂Sbose/∂ϕᵢ(τ) - 2gᵀ⋅[∂M/∂ϕᵢ(τ)]⋅M⁻¹g ==> All Done!
    calc_dSbosedϕ!( dSdϕ , holstein )

    # returning number of iterations in solving for M⁻¹g
    return info.iters
end

end