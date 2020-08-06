module LangevinDynamics

using Random
using UnsafeArrays
using LinearAlgebra

using ..Utilities: get_index
using ..Models: HolsteinModel, construct_expnΔτV!, mulMᵀ!, muldMdx!
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

    N::Int
    Δt::T
    dSdx::Vector{T}
    η::Vector{T}
    Δx::Vector{T}
    R::Vector{T}
    M⁻¹R::Vector{T}

    function EulerDynamics(N::Int, Δt::T) where {T<:AbstractFloat}

        dSdx     = zeros(T,N)
        η        = zeros(T,N)
        Δx       = zeros(T,N)
        R        = zeros(T,N)
        M⁻¹R     = zeros(T,N)

        return new{T}(N,Δt,dSdx,η,Δx,R,M⁻¹R)
    end
end

function evolve!(holstein::HolsteinModel{T1,T2}, dyn::EulerDynamics{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Int  where {T1<:AbstractFloat,T2<:Number}

    N        = dyn.N
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
    construct_expnΔτV!(holstein)

    # itialize η as vector of gaussian random number
    randn!(η)

    # calculate dSdx = [∂S/∂x₁(1),...,∂S/∂xₙ(1),...,∂S/∂x₁(τ),...,∂S/∂xₙ(τ),...,∂S/∂x₁(Lτ),...,∂S/∂xₙ(Lτ)]
    randn!(R)
    iters = calc_dSdx!(dSdx, R, M⁻¹R, holstein, preconditioner)

    # dSdx ==> Q⋅dSdx
    fourier_accelerate!( QdSdx, fa , dSdx, 1.0 )

    # η = √Q⋅η
    fourier_accelerate!( sqrtQη, fa , η, 0.5 )

    # Δx = √(2⋅Δt⋅Q)⋅η - Δt⋅Q⋅dS/dx
    @. Δx = sqrt(2.0*Δt)*sqrtQη - Δt*QdSdx

    # update phonon fields
    @. holstein.x += Δx

    # update the exponentiated interaction matrix to reflect current phonon field configuration.
    construct_expnΔτV!(holstein)

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

    N::Int
    Δt::T
    dSdx::Vector{T}
    dSdx′::Vector{T}
    η::Vector{T}
    Δx::Vector{T}
    R::Vector{T}
    M⁻¹R::Vector{T}

    function RungeKuttaDynamics(N::Int, Δt::T) where {T<:AbstractFloat}

        dSdx     = zeros(T,N)
        dSdx′    = zeros(T,N)
        η        = zeros(T,N)
        Δx       = zeros(T,N)
        R        = zeros(T,N)
        M⁻¹R     = zeros(T,N)

        return new{T}(N,Δt,dSdx,dSdx′,η,Δx,R,M⁻¹R)
    end
end

function evolve!(holstein::HolsteinModel{T1,T2}, dyn::RungeKuttaDynamics{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Int  where {T1<:AbstractFloat,T2<:Number}

    N        = dyn.N
    Δt       = dyn.Δt
    dSdx     = dyn.dSdx
    QdSdx    = dyn.dSdx
    dSdx′    = dyn.dSdx′
    η        = dyn.η
    sqrtQη   = dyn.η
    Δx       = dyn.Δx
    R        = dyn.R
    M⁻¹R     = dyn.M⁻¹R
    x        = holstein.x
    x′       = holstein.x
    x″       = holstein.x

    # update the exponentiated interaction matrix to reflect current phonon field configuration.
    construct_expnΔτV!(holstein)

    # itialize η as vector of gaussian random number
    randn!(η)

    # calculate dSdx = [∂S/∂x₁(1),...,∂S/∂xₙ(1),...,∂S/∂x₁(τ),...,∂S/∂xₙ(τ),...,∂S/∂x₁(Lτ),...,∂S/∂xₙ(Lτ)]
    randn!(R)
    iters = calc_dSdx!(dSdx, R, M⁻¹R, holstein, preconditioner)

    # get the update for the fields using euler method
    @. Δx = sqrt(2*Δt)*η - Δt*dSdx

    # update phonon fields
    @. x′ = x + Δx
 
    # update the exponentiated interaction matrix so that it reflects the current phonon field configuration.
    construct_expnΔτV!(holstein)

    # calculate dSdx = [∂S/∂x₁(1),...,∂S/∂xₙ(1),...,∂S/∂x₁(τ),...,∂S/∂xₙ(τ),...,∂S/∂x₁(Lτ),...,∂S/∂xₙ(Lτ)]
    randn!(R)
    iters = calc_dSdx!(dSdx′, R, M⁻¹R, holstein, preconditioner)

    # revert back to original phonon fields
    @. x = x′ - Δx

    # update the exponentiated interaction matrix so that it reflects the current phonon field configuration.
    construct_expnΔτV!(holstein)

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
    construct_expnΔτV!(holstein)

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

    N::Int
    Δt::T
    η::Vector{T}
    dSdx::Vector{T}
    dSdx′::Vector{T}
    Δx::Vector{T}
    R::Vector{T}
    M⁻¹R::Vector{T}

    function HeunsDynamics(N::Int,Δt::T) where {T<:AbstractFloat}

        η     = zeros(T,N)
        dSdx  = zeros(T,N)
        dSdx′ = zeros(T,N)
        Δx    = zeros(T,N)
        R     = zeros(T,N)
        M⁻¹R  = zeros(T,N)

        return new{T}(N,Δt,η,dSdx,dSdx′,Δx,R,M⁻¹R)
    end
end

function evolve!(holstein::HolsteinModel{T1,T2}, dyn::HeunsDynamics{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Int  where {T1<:AbstractFloat,T2<:Number}

    x        = holstein.x
    x′       = holstein.x
    x″       = holstein.x
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
    construct_expnΔτV!(holstein)
    randn!(R)
    iters1 = calc_dSdx!(dSdx, R, M⁻¹R, holstein, preconditioner)

    # 4. dΓ/dx  = [F⁻¹⋅Q⋅F]⋅dS/dx
    fourier_accelerate!(dΓdx,fa,dSdx,1.0)

    # 5. Δx = √(2Δt)⋅ξ - Δt⋅dΓ/dx
    @. Δx = sqrt(2*Δt)*ξ - Δt*dΓdx

    # 6. x′ = x + Δx
    @. x′ = x + Δx
    construct_expnΔτV!(holstein)

    # 7. calculate dS/dx′
    randn!(R)
    iters2 = calc_dSdx!(dSdx′, R, M⁻¹R, holstein, preconditioner)

    # 8. dΓ/dx′ = [F⁻¹⋅Q⋅F]⋅dS/dx′
    fourier_accelerate!(dΓdx′,fa,dSdx′,1.0)

    # 9. x = x′- Δx
    @. x = x′ - Δx

    # 10. x″ = x + √(2Δt)⋅ξ - Δt⋅(dΓ/dx+dΓ/dx′)/2 
    @. x″ = x + sqrt(2*Δt)*ξ - Δt*(dΓdx+dΓdx′)/2
    construct_expnΔτV!(holstein)

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
                    holstein::HolsteinModel{T1,T2}, preconditioner=I)::Int where {T1<:AbstractFloat,T2<:Number}
    
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
function calc_dSfdx!(dSfdx::AbstractVector{T2},g::AbstractVector{T3},M⁻¹g::AbstractVector{T3},
                     holstein::HolsteinModel{T1,T2}, preconditioner=I)::Int where {T1<:AbstractFloat,T2<:Number,T3<:Number}


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
    if holstein.mul_by_M
        # solve M⋅x=g ==> x=M⁻¹⋅g
        holstein.transposed = false
        iters = ldiv!(M⁻¹g,holstein,g,preconditioner)
    else
        # solve MᵀM⋅x=Mᵀg ==> x=[MᵀM]⁻¹⋅Mᵀg=M⁻¹⋅g
        mulMᵀ!(holstein.Mᵀg,holstein,g)
        iters = ldiv!(M⁻¹g,holstein,holstein.Mᵀg,preconditioner)
    end

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