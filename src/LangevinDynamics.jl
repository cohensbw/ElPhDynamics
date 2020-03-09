module LangevinDynamics

using IterativeSolvers
using Random
import LinearAlgebra: mul!, ldiv!, dot

using ..Utilities: get_index
using ..HolsteinModels: HolsteinModel, construct_expnΔτV!, mulMᵀ!, muldMdx!
using ..PhononAction: calc_dSbosedx!
using ..FourierAcceleration: FourierAccelerator, forward_fft!, inverse_fft!, accelerate!, accelerate_noise!

export update_leapfrog_fa!, update_euler_fa!, update_rk_fa!, calc_dSdx!, calc_dSfdx!

"""
Update phonon field using the symplectic Leapfrog integrator with fourier acceleration applied.
This method introduces a conjugate momentum `p` that is refreshed every 1/Δt time steps.
"""
function update_leapfrog_fa!(timestep::Int, holstein::HolsteinModel{T1,T2}, fa::FourierAccelerator{T1},
                             dx::AbstractVector{T1}, dSdx::AbstractVector{T2}, fft_dSdx::AbstractVector{Complex{T1}},
                             g::AbstractVector{T2}, M⁻¹g::AbstractVector{T2},
                             p::AbstractVector{T1}, fft_p::AbstractVector{Complex{T1}},
                             Δt::T1, preconditioner)::Int where {T1<:AbstractFloat,T2<:Number}

    # frequency with which to refresh the momentum
    refresh_freq = Int(1/Δt)

    iters = update_leapfrog_fa!(timestep, holstein, fa, dx, dSdx, fft_dSdx, g, M⁻¹g, p, fft_p, Δt, preconditioner, refresh_freq)

    return iters
end

function update_leapfrog_fa!(timestep::Int, holstein::HolsteinModel{T1,T2}, fa::FourierAccelerator{T1},
                             dx::AbstractVector{T1}, dSdx::AbstractVector{T2}, fft_dSdx::AbstractVector{Complex{T1}},
                             g::AbstractVector{T2}, M⁻¹g::AbstractVector{T2},
                             p::AbstractVector{T1}, fft_p::AbstractVector{Complex{T1}},
                             Δt::T1, preconditioner, refresh_freq::Int)::Int where {T1<:AbstractFloat,T2<:Number}
    
    # refresh noise vector
    randn!(g)
    
    # update the exponentiated interaction matrix for current phonon config
    construct_expnΔτV!(holstein)
    
    # calculate force
    iters = calc_dSdx!(dSdx, g, M⁻¹g, holstein, preconditioner)
        
    # fourier accelerate the force.
    forward_fft!(fft_dSdx, dSdx, fa)
    accelerate_noise!(fft_dSdx, fa )
    inverse_fft!(dSdx, fft_dSdx, fa)
    
    if (timestep%refresh_freq-1)==0
        # refresh the momentum from a gaussian
        randn!(p)
        # update momentum a half-time step
        @. p -= 0.5 * dSdx * Δt
    else
        # update the momentum a full time step
        @. p -= dSdx * Δt
    end

    # fourier accelerate the momentum
    # note that the fourier acceleration carries with it a factor of Δt.
    forward_fft!(fft_p, p, fa)
    accelerate_noise!(fft_p, fa )
    inverse_fft!(dx, fft_p, fa)
    
    # udpate the phonon field
    holstein.x += dx * Δt
    
    return iters
end


"""
Update phonon fields using Runge-Kutta/Heun's equation and fourier acceleration.
"""
function update_rk_fa!(holstein::HolsteinModel{T1,T2}, fa::FourierAccelerator{T1},
                       dx::AbstractVector{T1}, fft_dx::AbstractVector{Complex{T1}},
                       dSdx2::AbstractVector{T2}, dSdx1::AbstractVector{T2}, fft_dSdx::AbstractVector{Complex{T1}},
                       g::AbstractVector{T2}, M⁻¹g::AbstractVector{T2},
                       η::AbstractVector{T1}, fft_η::AbstractVector{Complex{T1}},
                       Δt::T1, preconditioner=Identity())::Int where {T1<:AbstractFloat,T2<:Number}
    
    # itialize η as vector of gaussian random number
    randn!(η)

    # calculate dSdx = [∂S/∂x₁(1),...,∂S/∂xₙ(1),...,∂S/∂x₁(τ),...,∂S/∂xₙ(τ),...,∂S/∂x₁(Lτ),...,∂S/∂xₙ(Lτ)]
    randn!(g)
    iters = calc_dSdx!(dSdx1, g, M⁻¹g, holstein, preconditioner)

    # get the update for the fields using euler method
    @. dx = sqrt(2*Δt)*η - Δt*real(dSdx1)

    # update phonon fields
    @. holstein.x += dx
 
    # update the exponentiated interaction matrix so that it reflects the current
    # phonon field configuration.
    construct_expnΔτV!(holstein)

    # calculate dSdx = [∂S/∂x₁(1),...,∂S/∂xₙ(1),...,∂S/∂x₁(τ),...,∂S/∂xₙ(τ),...,∂S/∂x₁(Lτ),...,∂S/∂xₙ(Lτ)]
    randn!(g)
    iters = calc_dSdx!(dSdx2, g, M⁻¹g, holstein, preconditioner)

    # revert back to original phonon fields
    @. holstein.x -= dx

    # update the exponentiated interaction matrix so that it reflects the current
    # phonon field configuration.
    construct_expnΔτV!(holstein)

    # get the partial derivative for the RK step
    @. dSdx2 = (dSdx2+dSdx1)/2.0

    # fourier transform dSdx2
    forward_fft!( fft_dSdx , dSdx2 , fa)

    # accelerate fft_dSdx ==> Q⋅fft_dSdx
    accelerate!( fft_dSdx , fa )

    # fourier transform η
    forward_fft!( fft_η , η , fa )

    # accelerate noise vector fft_η ==> √(Q)⋅fft_η
    accelerate_noise!( fft_η , fa )

    # calculate fft_dx
    @. fft_dx = sqrt(2*Δt)*fft_η - Δt*fft_dSdx

    # perform inverse fourier transform to get dx
    inverse_fft!( dx , fft_dx , fa )

    # update phonon fields
    @. holstein.x += dx

    # update the exponentiated interaction matrix so that it reflects the current
    # phonon field configuration.
    construct_expnΔτV!(holstein)

    return iters
end


"""
Update phonon fields using Euler equation and fourier acceleration.
"""
function update_euler_fa!(holstein::HolsteinModel{T1,T2}, fa::FourierAccelerator{T1},
                          dx::AbstractVector{T1}, fft_dx::AbstractVector{Complex{T1}},
                          dSdx::AbstractVector{T2}, fft_dSdx::AbstractVector{Complex{T1}},
                          g::AbstractVector{T2}, M⁻¹g::AbstractVector{T2},
                          η::AbstractVector{T1}, fft_η::AbstractVector{Complex{T1}},
                          Δt::T1, preconditioner=Identity())::Int where {T1<:AbstractFloat,T2<:Number}
    
    # itialize η as vector of gaussian random number
    randn!(η)

    # calculate dSdx = [∂S/∂x₁(1),...,∂S/∂xₙ(1),...,∂S/∂x₁(τ),...,∂S/∂xₙ(τ),...,∂S/∂x₁(Lτ),...,∂S/∂xₙ(Lτ)]
    randn!(g)
    iters = calc_dSdx!(dSdx, g, M⁻¹g, holstein, preconditioner)

    # fourier transform dSdx
    forward_fft!( fft_dSdx , dSdx , fa)

    # accelerate fft_dSdx ==> Q⋅fft_dSdx
    accelerate!( fft_dSdx , fa )

    # fourier transform η
    forward_fft!( fft_η , η , fa )

    # accelerate fft_η ==> √(Q)⋅fft_η
    accelerate_noise!( fft_η , fa )

    # calculate fft_dx
    @. fft_dx = sqrt(2.0*Δt)*fft_η - Δt*fft_dSdx

    # perform inverse fourier transform to get dx
    inverse_fft!( dx , fft_dx , fa )

    # update phonon fields
    @. holstein.x += dx

    # update the exponentiated interaction matrix so that it reflects the current
    # phonon field configuration.
    construct_expnΔτV!(holstein)

    return iters
end

#################################################
## FUNCTION FOR CALCULATING FORCE IN DYNAMICS ##
#################################################

"""
    function calc_dSdx!(dSdx::AbstractVector{T1},g::AbstractVector{T1},Mᵀg::AbstractVector{T1},M⁻¹g::AbstractVector{T1},holstein::HolsteinModel{T1,T2},tol::AbstractFloat)::Int where {T1<:AbstractFloat,T2<:Number}

Calculates all of the partial derivatives ∂S/∂xᵢ(τ) and stores each partial derivative in a vector dSdx.
The expression we are evaluating is `∂S/∂xᵢ(τ) = ∂Sbose/∂xᵢ(τ) - 2gᵀ[∂M/∂xᵢ(τ)]⋅M⁻¹g`.
# Arguments
- `dSdx::AbstractVector{T1}`: vector the will be modified to contain all the partial derivatives ∂S/∂xᵢ(τ)
- `g::AbstractVector{T1}`: A random vector.
- `Mᵀg::AbstractVector{T1}`: Vector containing the product Mᵀg
- `M⁻¹g::AbstractVector{T1}`: A vector the will contain the solution of the linear equation M⋅v=g when computed.
- `holstein::HolsteinModel{T1,T2}`: Type represent holstein model being simulated.
- `tol::AbstractFloat`: The tolerance used when solving the linear equation M⋅v=g in order to get M⁻¹g.
# Returns
- `iters::Int`: Number of iterations used to solve for M⁻¹g.
"""
function calc_dSdx!(dSdx::AbstractVector{T2},g::AbstractVector{T2},M⁻¹g::AbstractVector{T2},
                    holstein::HolsteinModel{T1,T2}, preconditioner=Identity())::Int where {T1<:AbstractFloat,T2<:Number}
    
    # ∂S/∂xᵢ(τ) = -2gᵀ⋅[∂M/∂xᵢ(τ)]⋅M⁻¹g
    iters = calc_dSfdx!(dSdx, g, M⁻¹g, holstein, preconditioner)

    # ∂S/∂xᵢ(τ) = ∂Sbose/∂xᵢ(τ) - 2gᵀ⋅[∂M/∂xᵢ(τ)]⋅M⁻¹g ==> All Done!
    calc_dSbosedx!( dSdx , holstein )

    # returning number of iterations in solving for M⁻¹g
    return iters
end

"""
Calculate just the force associated with the fermionic part of the action.
"""
function calc_dSfdx!(dSfdx::AbstractVector{T2},g::AbstractVector{T2},M⁻¹g::AbstractVector{T2},
                     holstein::HolsteinModel{T1,T2}, preconditioner=Identity())::Int where {T1<:AbstractFloat,T2<:Number}


    # NOTE: The method I use below for calculating all of the partial derivatives {∂S/∂xᵢ(τ)} works only
    # because I am assuming a strictly local form of the electron-phonon interaction given by ∑(λᵢ⋅xᵢ⋅nᵢ).
    # If a longer-range electron-phonon interaction were added of the form ∑(λᵢⱼ⋅xᵢ⋅nⱼ), then this way of doing
    # things would no longer work, and a loop over each individual phonon field xᵢ(τ) would need to be added.

    # # intialize random vector g.
    # randn!(g)

    # solve M⋅v = g ==> v = M⁻¹⋅g
    iters = ldiv!(M⁻¹g,holstein,g,preconditioner)

    # ∂Sf/∂xᵢ(τ) = ∂M/∂xᵢ(τ)⋅M⁻¹g
    muldMdx!( dSfdx , holstein , M⁻¹g )

    # ∂Sf/∂xᵢ(τ) = -2gᵀ⋅∂M/∂xᵢ(τ)⋅M⁻¹g
    @. g *= -2.0 * dSfdx
    # iterate over sites
    @inbounds @fastmath for site in 1:holstein.nsites
        # iterate over time slices
        for τ in 1:holstein.Lτ
            idx_τ   = get_index(τ,               site, holstein.Lτ)
            idx_τp1 = get_index(τ%holstein.Lτ+1, site, holstein.Lτ)
            # shifting values one time slice forward
            dSfdx[idx_τp1] = real(g[idx_τ])
        end
    end
    # In the lines of code above there is a subtle detail that is addressed.
    # After doing the element-wise multiplication, the expectation value for the partial
    # derivatives corresponding to the τ time slice live in the array indices corresponding to τ-1.
    # Therefore, the values need to be shifted one time slice forward. This is done by first calculating
    # and storing the ∂Sf/∂xᵢ(τ) derivative values in the vector g, and then copying a proper shifted
    # version into the vector dSdx.

    return iters
end


end