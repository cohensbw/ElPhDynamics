module LangevinDynamics

using Random
using UnsafeArrays
using LinearAlgebra

using ..Utilities: get_index
using ..Models: AbstractModel, HolsteinModel, SSHModel, update_model!, mulMᵀ!, muldMdx!
using ..PhononAction: calc_dSbosedx!
using ..FourierAcceleration: FourierAccelerator, fourier_accelerate!

using ..KPMPreconditioners: setup!

export evolve!, Dynamics, EulerDynamics, RungeKuttaDynamics, HeunsDynamics
export calc_dSdx!, calc_dSfdx!

"""
Abstract type for representing different way of evolving Langevin Dynamics.
"""
abstract type Dynamics end

##################
## EULER UPDATE ##
##################

struct EulerDynamics{T<:AbstractFloat} <: Dynamics

    """
    Number of degrees of freedom (DOF) to update.
    """
    Ndof::Int

    """
    Dimension of M matrix.
    """
    Ndim::Int

    """
    Timestep.
    """
    Δt::T

    """
    Derivative of action with respect to each DOF.
    """
    dSdx::Vector{T}

    """
    Noise vector in Langevin dynamics.
    """
    η::Vector{T}

    """
    Change in phonon fields.
    """
    Δx::Vector{T}

    """
    Random noise vector.
    """
    R::Vector{T}

    """
    M⁻¹⋅R
    """
    M⁻¹R::Vector{T}

    function EulerDynamics(model::AbstractModel, Δt::T) where {T<:AbstractFloat}

        Ndof     = model.Ndof
        Ndim     = model.Ndim
        dSdx     = zeros(T,Ndof)
        η        = zeros(T,Ndof)
        Δx       = zeros(T,Ndof)
        R        = zeros(T,Ndim)
        M⁻¹R     = zeros(T,Ndim)

        return new{T}(Ndof,Ndim,Δt,dSdx,η,Δx,R,M⁻¹R)
    end
end

function evolve!(model::AbstractModel{T1,T2,T3}, dyn::EulerDynamics{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Int  where {T1,T2,T3}

    Δt       = dyn.Δt
    dSdx     = dyn.dSdx
    QdSdx    = dyn.dSdx
    η        = dyn.η
    sqrtQη   = dyn.η
    Δx       = dyn.Δx
    R        = dyn.R
    M⁻¹R     = dyn.M⁻¹R

    # update the exponentiated interaction matrix so that it reflects the current
    # phonon field configuration.
    update_model!(model)

    # itialize η as vector of gaussian random number
    randn!(η)

    # calculate dSdx = [∂S/∂x₁(1),...,∂S/∂xₙ(1),...,∂S/∂x₁(τ),...,∂S/∂xₙ(τ),...,∂S/∂x₁(Lτ),...,∂S/∂xₙ(Lτ)]
    randn!(R)
    iters = calc_dSdx!(dSdx, R, M⁻¹R, model, preconditioner)

    # dSdx ==> Q⋅dSdx
    fourier_accelerate!( QdSdx, fa , dSdx, 1.0 )

    # η = √Q⋅η
    fourier_accelerate!( sqrtQη, fa , η, 0.5 )

    # Δx = √(2⋅Δt⋅Q)⋅η - Δt⋅Q⋅dS/dx
    @. Δx = sqrt(2.0*Δt)*sqrtQη - Δt*QdSdx

    # update phonon fields
    @. model.x += Δx

    # update the exponentiated interaction matrix to reflect current phonon field configuration.
    update_model!(model)

    return iters
end

########################
## RUNGE-KUTTA UPDATE ##
########################

# Implemented Definition of Runge-Kutta update with Fourier Acceleration:
# 1. initialize η
# 2. calcualte dS/dx
# 3. x′ = x - Δt⋅dS/dx + √(2Δt)⋅η
# 4. calculate dS/dx′
# 5. x″ = x - Δt⋅[F⁻¹⋅Q⋅F]⋅(dS/dx′+dS/dx)/2 + √(2Δt)⋅[F⁻¹⋅√Q⋅F]⋅η
# Note: Fourier Acceleration Only Applied in Step 5
# Note: F is fourier transform from τ ⟶ ω
# Note: Q is the diagonal acceleration matrix

struct RungeKuttaDynamics{T<:AbstractFloat} <: Dynamics

    Ndof::Int
    Ndim::Int
    Δt::T
    dSdx::Vector{T}
    dSdx′::Vector{T}
    η::Vector{T}
    Δx::Vector{T}
    R::Vector{T}
    M⁻¹R::Vector{T}

    function RungeKuttaDynamics(model::AbstractModel, Δt::T) where {T<:AbstractFloat}

        Ndof     = model.Ndof
        Ndim     = model.Ndim
        dSdx     = zeros(T,Ndof)
        dSdx′    = zeros(T,Ndof)
        η        = zeros(T,Ndof)
        Δx       = zeros(T,Ndof)
        R        = zeros(T,Ndim)
        M⁻¹R     = zeros(T,Ndim)

        return new{T}(N,Δt,dSdx,dSdx′,η,Δx,R,M⁻¹R)
    end
end

function evolve!(model::AbstractModel{T1,T2,T3}, dyn::RungeKuttaDynamics{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Int  where {T1,T2,T3}

    Δt       = dyn.Δt
    dSdx     = dyn.dSdx
    QdSdx    = dyn.dSdx
    dSdx′    = dyn.dSdx′
    η        = dyn.η
    sqrtQη   = dyn.η
    Δx       = dyn.Δx
    R        = dyn.R
    M⁻¹R     = dyn.M⁻¹R
    x        = model.x
    x′       = model.x
    x″       = model.x

    # update the exponentiated interaction matrix to reflect current phonon field configuration.
    update_model!(model)

    # itialize η as vector of gaussian random number
    randn!(η)

    # calculate dSdx = [∂S/∂x₁(1),...,∂S/∂xₙ(1),...,∂S/∂x₁(τ),...,∂S/∂xₙ(τ),...,∂S/∂x₁(Lτ),...,∂S/∂xₙ(Lτ)]
    randn!(R)
    iters = calc_dSdx!(dSdx, R, M⁻¹R, model, preconditioner)

    # get the update for the fields using euler method
    @. Δx = sqrt(2*Δt)*η - Δt*dSdx

    # update phonon fields
    @. x′ = x + Δx
 
    # update the exponentiated interaction matrix so that it reflects the current phonon field configuration.
    update_model!(model)

    # calculate dSdx = [∂S/∂x₁(1),...,∂S/∂xₙ(1),...,∂S/∂x₁(τ),...,∂S/∂xₙ(τ),...,∂S/∂x₁(Lτ),...,∂S/∂xₙ(Lτ)]
    randn!(R)
    iters = calc_dSdx!(dSdx′, R, M⁻¹R, model, preconditioner)

    # revert back to original phonon fields
    @. x = x′ - Δx

    # update the exponentiated interaction matrix so that it reflects the current phonon field configuration.
    update_model!(model)

    # get the partial derivative for the second RK step
    @. dSdx = (dSdx′+dSdx)/2.0

    # dS/dx ==> Q⋅dS/dx
    fourier_accelerate!(QdSdx,fa,dSdx,1.0)

    # η ==> √Q⋅η
    fourier_accelerate!(sqrtQη,fa,η,0.5)

    # Δx = √(2⋅Δt⋅Q)⋅η - Δt⋅Q⋅dS/dx
    @. Δx = sqrt(2.0*Δt)*sqrtQη - Δt*QdSdx 

    # update phonon fields
    @. x″ = x + Δx

    # update the exponentiated interaction matrix so that it reflects the current phonon field configuration.
    update_model!(model)

    return iters
end

#####################
## HEUN'S DYNAMICS ##
#####################

# Implemented Definition of Heun's update with Fourier Acceleration:
# 1.  initialize η
# 2.  ξ      = [F⁻¹⋅√Q⋅F]⋅η
# 3.  calcualte dS/dx
# 4.  dΓ/dx  = [F⁻¹⋅Q⋅F]⋅dS/dx
# 5.  Δx     = √(2Δt)⋅ξ - Δt⋅dΓ/dx
# 6.  x′     = x + Δx
# 7.  calculate dS/dx′
# 8.  dΓ/dx′ = [F⁻¹⋅Q⋅F]⋅dS/dx′
# 9.  x      = x′- Δx
# 10. x″     = x + √(2Δt)⋅ξ - Δt⋅(dΓ/dx+dΓ/dx′)/2
# Note: F is fourier transform from τ ⟶ ω
# Note: Q is the diagonal acceleration matrix

struct HeunsDynamics{T<:AbstractFloat} <: Dynamics

    Ndof::Int
    Ndim::Int
    Δt::T
    η::Vector{T}
    dSdx::Vector{T}
    dSdx′::Vector{T}
    Δx::Vector{T}
    R::Vector{T}
    M⁻¹R::Vector{T}

    function HeunsDynamics(model::AbstractModel,Δt::T) where {T<:AbstractFloat}

        Ndof  = model.Ndof
        Ndim  = model.Ndim
        η     = zeros(T,Ndof)
        dSdx  = zeros(T,Ndof)
        dSdx′ = zeros(T,Ndof)
        Δx    = zeros(T,Ndof)
        R     = zeros(T,Ndim)
        M⁻¹R  = zeros(T,Ndim)

        return new{T}(Ndof,Ndim,Δt,η,dSdx,dSdx′,Δx,R,M⁻¹R)
    end
end

function evolve!(model::AbstractModel{T1,T2,T3}, dyn::HeunsDynamics{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Int  where {T1,T2,T3}

    x        = model.x
    x′       = model.x
    x″       = model.x
    Δx       = dyn.Δx
    Δt       = dyn.Δt
    η        = dyn.η
    dSdx     = dyn.dSdx
    dΓdx     = dyn.dSdx
    dSdx′    = dyn.dSdx′
    dΓdx′    = dyn.dSdx′
    R        = dyn.R
    M⁻¹R     = dyn.M⁻¹R

    # 1. intialize η
    randn!(η)

    # 2. ξ = [F⁻¹⋅√Q⋅F]⋅η
    ξ = η
    fourier_accelerate!(ξ,fa,η,0.5)

    # 3. calcualte dS/dx
    update_model!(model)
    randn!(R)
    iters1 = calc_dSdx!(dSdx, R, M⁻¹R, model, preconditioner)

    # 4. dΓ/dx  = [F⁻¹⋅Q⋅F]⋅dS/dx
    fourier_accelerate!(dΓdx,fa,dSdx,1.0)

    # 5. Δx = √(2Δt)⋅ξ - Δt⋅dΓ/dx
    @. Δx = sqrt(2*Δt)*ξ - Δt*dΓdx

    # 6. x′ = x + Δx
    @. x′ = x + Δx
    update_model!(model)

    # 7. calculate dS/dx′
    randn!(R)
    iters2 = calc_dSdx!(dSdx′, R, M⁻¹R, model, preconditioner)

    # 8. dΓ/dx′ = [F⁻¹⋅Q⋅F]⋅dS/dx′
    fourier_accelerate!(dΓdx′,fa,dSdx′,1.0)

    # 9. x = x′- Δx
    @. x = x′ - Δx

    # 10. x″ = x + √(2Δt)⋅ξ - Δt⋅(dΓ/dx+dΓ/dx′)/2 
    @. x″ = x + sqrt(2*Δt)*ξ - Δt*(dΓdx+dΓdx′)/2
    update_model!(model)

    return div(iters1+iters2,2)
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
                    model::AbstractModel{T1,T2,T3}, preconditioner=I)::Int where {T1,T2,T3}
    
    # ∂S/∂xᵢ(τ) = -2gᵀ⋅[∂M/∂xᵢ(τ)]⋅M⁻¹g
    iters = calc_dSfdx!(dSdx, g, M⁻¹g, model, preconditioner)

    # ∂S/∂xᵢ(τ) = ∂Sbose/∂xᵢ(τ) - 2gᵀ⋅[∂M/∂xᵢ(τ)]⋅M⁻¹g ==> All Done!
    calc_dSbosedx!( dSdx , model )

    # returning number of iterations in solving for M⁻¹g
    return iters
end

"""
Calculate just the force associated with the fermionic part of the action.
"""
function calc_dSfdx!(dSfdx::AbstractVector{T2},g::AbstractVector{T2},M⁻¹g::AbstractVector{T2},
                     model::AbstractModel{T1,T2,T3}, preconditioner=I)::Int where {T1,T2,T3}


    # NOTE: The method I use below for calculating all of the partial derivatives {∂S/∂xᵢ(τ)} works only
    # because I am assuming a strictly local form of the electron-phonon interaction given by ∑(λᵢ⋅xᵢ⋅nᵢ).
    # If a longer-range electron-phonon interaction were added of the form ∑(λᵢⱼ⋅xᵢ⋅nⱼ), then this way of doing
    # things would no longer work, and a loop over each individual phonon field xᵢ(τ) would need to be added.

    # # intialize random vector g.
    # randn!(g)

    # solve linear system to get M⁻¹⋅g
    iters = 0
    setup!(preconditioner) # setup block preconditioner
    fill!(M⁻¹g,0.0)
    if model.mul_by_M
        # solve M⋅x=g ==> x=M⁻¹⋅g
        model.transposed = false
        iters = ldiv!(M⁻¹g,model,g,preconditioner)
    else
        # solve MᵀM⋅x=Mᵀg ==> x=[MᵀM]⁻¹⋅Mᵀg=M⁻¹⋅g
        mulMᵀ!(model.Mᵀg,model,g)
        iters = ldiv!(M⁻¹g,model,model.Mᵀg,preconditioner)
    end

    # ⟨∂M/∂xᵢ(τ)⟩=gᵀ⋅∂M/∂xᵢ(τ)⋅M⁻¹g
    muldMdx!( dSfdx , g, model , M⁻¹g )

    # ∂Sf/∂xᵢ(τ) = -2⟨∂M/∂xᵢ(τ)⟩ = -2gᵀ⋅∂M/∂xᵢ(τ)⋅M⁻¹g
    @. dSfdx = -2.0 * dSfdx

    return iters
end

end