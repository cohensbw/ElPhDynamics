module HMC

using Random
using UnsafeArrays
using LinearAlgebra
using Printf
using Parameters

using ..Utilities: get_index
using ..Models: HolsteinModel, construct_expnΔτV!, mulM!, muldMdx!, mulMᵀ!
using ..PhononAction: calc_dSbosedx!, calc_Sbose
using ..FourierAcceleration: FourierAccelerator, fourier_accelerate!
import ..KPMPreconditioners

export HybridMonteCarlo, update!

mutable struct HybridMonteCarlo{T<:AbstractFloat}

    """
    Number of degrees of freedom to update.
    """
    Ndof::Int

    """
    Store initial phonon fields.
    """
    x0::Vector{T}

    """
    Time between refreshing the momentum p and auxialiary fields ϕ.
    """
    tr::T

    """
    BDP thermostat timescale
    """
    τ::T

    """
    Timestep.
    """
    Δt::T

    """
    Number of time steps.
    """
    Nt::Int

    """
    Smaller timestep used to evolve Sb for multi-timestep algorithm.
    """
    Δt′::T

    """
    Number of steps for Sb per one timestep of Sf for multi-timestep algorithm.
    """
    Nb::Int

    """
    Partial momentum refresh parameter.
    """
    α::T

    """
    The energy H = S + K where K is energy of conjugate momentum and S is the action.
    """
    H::T

    """
    Derivative of the action.
    """
    dSdx::Vector{T}

    """
    Velocity.
    """
    v::Vector{T}

    """
    Stores initial velocity vector.
    """
    v0::Vector{T}

    """
    Random gaussian vector.
    """
    R::Vector{T}

    """
    Auxiliary fields for spin up electrons.
    """
    ϕ₊::Vector{T}

    """
    Auxiliary fields for spin down electrons.
    """
    ϕ₋::Vector{T}

    """
    M⁻ᵀ⋅ϕ₊.
    """
    M⁻ᵀϕ₊::Vector{T}

    """
    M⁻ᵀ⋅ϕ₊ one time step back.
    """
    M⁻ᵀϕ₊′::Vector{T}

    """
    M⁻ᵀ⋅ϕ₋.
    """
    M⁻ᵀϕ₋::Vector{T}

    """
    M⁻ᵀ⋅ϕ₋ one time step back.
    """
    M⁻ᵀϕ₋′::Vector{T}

    """
    O⁻¹⋅ϕ₊ where O=MᵀM
    """
    O⁻¹ϕ₊::Vector{T}

    """
    O⁻¹⋅ϕ₊ where O=MᵀM one time step back.
    """
    O⁻¹ϕ₊′::Vector{T}

    """
    O⁻¹⋅ϕ₋ where O=MᵀM
    """
    O⁻¹ϕ₋::Vector{T}

    """
    O⁻¹⋅ϕ₋ where O=MᵀM one time step back.
    """
    O⁻¹ϕ₋′::Vector{T}

    """
    Construct initial guess when solving linear system.
    """
    construct_guess::Bool

    """
    Temporary storage vector.
    """
    u::Vector{T}

    function HybridMonteCarlo(Ndof::Int,Δt::T,tr::T,τ::T,α::T,Nb::Int,construct_guess::Bool=true) where {T<:AbstractFloat}

        # partial momentum refresh parameter
        @assert 0.0 <= α < 1.0

        x0     = zeros(T,Ndof)
        dSdx   = zeros(T,Ndof)
        v      = randn(T,Ndof)
        v0     = zeros(T,Ndof)
        R      = zeros(T,Ndof)
        ϕ₊     = zeros(T,Ndof)
        M⁻ᵀϕ₊  = zeros(T,Ndof)
        O⁻¹ϕ₊  = zeros(T,Ndof)
        ϕ₋     = zeros(T,Ndof)
        M⁻ᵀϕ₋  = zeros(T,Ndof)
        O⁻¹ϕ₋  = zeros(T,Ndof)

        M⁻ᵀϕ₊′ = zeros(T,Ndof)
        O⁻¹ϕ₊′ = zeros(T,Ndof)

        M⁻ᵀϕ₋′ = zeros(T,Ndof)
        O⁻¹ϕ₋′ = zeros(T,Ndof)

        u      = zeros(T,Ndof)

        # the action
        H = 0.0::T

        # number of timesteps
        Nt = round(Int,tr/Δt)

        # size of smaller timestep for Sb
        Δt′ = Δt/Nb

        # checking conditions on parameters
        @assert τ >= 0.0
        @assert 0.0 <= α < 1.0
        @assert !((α>0)&(isfinite(τ)))

        return new{T}(Ndof, x0, tr, τ, Δt, Nt, Δt′, Nb, α, H, dSdx, v, v0, R, ϕ₊, ϕ₋, M⁻ᵀϕ₊, M⁻ᵀϕ₊′, M⁻ᵀϕ₋, M⁻ᵀϕ₋′, O⁻¹ϕ₊, O⁻¹ϕ₊′, O⁻¹ϕ₋, O⁻¹ϕ₋′, construct_guess, u)
    end

    function HybridMonteCarlo(hmc::HybridMonteCarlo{T};Δt,tr,τ,α,Nb,construct_guess) where {T<:AbstractFloat}

        @unpack Ndof, x0, α, H, dSdx, v, v0, R, ϕ₊, ϕ₋, M⁻ᵀϕ₊, M⁻ᵀϕ₊′, M⁻ᵀϕ₋, M⁻ᵀϕ₋′, O⁻¹ϕ₊, O⁻¹ϕ₊′, O⁻¹ϕ₋, O⁻¹ϕ₋′, u = hmc
        Nt  = round(Int,tr/Δt)
        Δt′ = Δt/Nb
    
        return new{T}(Ndof, x0, tr, τ, Δt, Nt, Δt′, Nb, α, H, dSdx, v, v0, R, ϕ₊, ϕ₋, M⁻ᵀϕ₊, M⁻ᵀϕ₊′, M⁻ᵀϕ₋, M⁻ᵀϕ₋′, O⁻¹ϕ₊, O⁻¹ϕ₊′, O⁻¹ϕ₋, O⁻¹ϕ₋′, construct_guess, u)
    end
end


"""
Do a Hybrid Monte Carlo update to the phonon fields.
"""
function update!(holstein::HolsteinModel{T1,T2}, hmc::HybridMonteCarlo{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Tuple{Bool,T1}  where {T1<:AbstractFloat,T2<:Number}
    
    if hmc.Nb==1
        accepted, iters = standard_update!(holstein,hmc,fa,preconditioner)
    else
        accepted, iters = multitimestep_update!(holstein,hmc,fa,preconditioner)
    end
    return accepted, iters
end

"""
Standard HMC update.
"""
function standard_update!(holstein::HolsteinModel{T1,T2}, hmc::HybridMonteCarlo{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Tuple{Bool,T1}  where {T1<:AbstractFloat,T2<:Number}

    x     = holstein.x
    x0    = hmc.x0
    v0    = hmc.v0
    dSdx  = hmc.dSdx
    QdSdx = hmc.dSdx
    v     = hmc.v
    Nt    = hmc.Nt
    Δt    = hmc.Δt

    # update exp{-Δτ⋅V[x]}
    construct_expnΔτV!(holstein)

    # refresh the velocity v
    refresh_v!(hmc,fa)

    # refresh ϕ
    refresh_ϕ!(hmc,holstein,fa)

    # calculate the initial dS/dx value
    iter_t = calc_dSdx!(hmc, holstein, preconditioner)
    iters  = iter_t

    # dS/dx(t+Δt) ==> Q⋅dS/dx(t+Δt)
    fourier_accelerate!(QdSdx,fa,dSdx,-1.0,use_mass=true)

    # keeps track of change in effective energy
    ΔH̃ = 0.0

    # record intial state
    copyto!(x0,x)
    copyto!(v0,v)

    # iterate over time steps, doing leapfrog updates to the phonon fields
    iters = 0
    for t in 1:Nt

        # calculate energy
        H₀, S, K = calc_H(hmc, holstein, fa)

        # v(t+Δt/2) = v(t) - Δt/2⋅Q⋅dS/dx(t)
        @. v = v - Δt/2*QdSdx

        # x(t+Δt) = x(t) + Δt⋅v(t+Δt/2)
        @. x = x + Δt*v

        # update exp{-Δτ⋅V[x]}
        construct_expnΔτV!(holstein)

        # calculate dS/dx(t+Δt) value
        iter_t = calc_dSdx!(hmc, holstein, preconditioner)
        iters += iter_t

        # dS/dx(t+Δt) ==> Q⋅dS/dx(t+Δt)
        fourier_accelerate!(QdSdx,fa,dSdx,-1.0,use_mass=true)

        # v(t+Δt) = v(t+Δt/2) - Δt/2⋅Q⋅dS/dx(t+Δt)
        @. v = v - Δt/2*QdSdx

        # calculate energy
        H₁, S, K = calc_H(hmc, holstein, fa)

        # update change in energy
        ΔH̃ += H₁-H₀

        # apply BDP Thermostat
        bdp_thermostat!(v,hmc.u,K,hmc.τ,hmc.Δt)
    end

    # calculate probability of acceptance
    P = min(1.0, exp(-ΔH̃))

    # get the number of iterations
    iters = cld(iters,Nt)

    # Metropolis-Hasting Accept/Reject Step
    if rand() < P # if accepted
        
        return true, T1(iters)

    else # if rejected

        # reset to original phonon field
        copyto!(x,x0)

        # reset to reflected original velocities.
        # note: this does not do anything unless doing partial momentum refreshes.
        @. v = -v0

        # update exp{-Δτ⋅V[x]}
        construct_expnΔτV!(holstein)

        return false, T1(iters)
    end
end


"""
Multi-timestepping HMC update.
"""
function multitimestep_update!(holstein::HolsteinModel{T1,T2}, hmc::HybridMonteCarlo{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Tuple{Bool,T1}  where {T1<:AbstractFloat,T2<:Number}

    x      = holstein.x
    x0     = hmc.x0
    v0     = hmc.v0
    dSbdx  = hmc.dSdx
    QdSbdx = hmc.dSdx
    dSfdx  = hmc.dSdx
    QdSfdx = hmc.dSdx
    v      = hmc.v
    Nt     = hmc.Nt
    Δt     = hmc.Δt
    Nb     = hmc.Nb
    Δt′    = hmc.Δt′

    # update exp{-Δτ⋅V[x]}
    construct_expnΔτV!(holstein)

    # refresh the velocity v
    refresh_v!(hmc,fa)

    # refresh ϕ
    refresh_ϕ!(hmc,holstein,fa)

    # calculate the initial dSf/dx value
    iter_t = calc_dSfdx!(hmc, holstein, preconditioner)
    iters  = iter_t
    println("Iters = ",iter_t)

    # dSf/dx(t+Δt) ==> Q⋅dSf/dx(t+Δt)
    fourier_accelerate!(QdSfdx,fa,dSfdx,-1.0,use_mass=true)

    # keeps track of change in effective energy
    ΔH̃ = 0.0

    # record intial phonon configuration
    copyto!(x0,x)
    copyto!(v0,v)

    # iterate over timesteps
    for t in 1:Nt
        println("t = ",t)

        # calculate energy
        H₀, S, K = calc_H(hmc, holstein, fa)
        println("S = ",S)

        # v(t+Δt/2) = v(t) - Δt/2⋅Q⋅dSf/dx(t)
        @. v = v - Δt/2*QdSfdx

        # calculate the initial dSb/dx value
        fill!(dSbdx,0.0)
        calc_dSbosedx!(dSbdx,holstein)

        # dSb/dx(t+Δt) ==> Q⋅dSb/dx(t+Δt)
        fourier_accelerate!(QdSbdx,fa,dSbdx,-1.0,use_mass=true)

        # evolve Sb using Nb smaller timesteps of size Δt′=Δt/Nb
        for t′ in 1:Nb

            # v(t+Δt/2) = v(t) - Δt′/2⋅Q⋅dSb/dx(t)
            @. v = v - Δt′/2*QdSbdx

            # x(t+Δt) = x(t) + Δt⋅v(t+Δt/2)
            @. x = x + Δt′*v

            # calculate dSb/dx(t+Δt) value
            fill!(dSbdx,0.0)
            calc_dSbosedx!(dSbdx,holstein)

            # dS/dx(t+Δt) ==> Q⋅dS/dx(t+Δt)
            fourier_accelerate!(QdSbdx,fa,dSbdx,-1.0,use_mass=true)

            # v(t+Δt) = v(t+Δt/2) - Δt/2⋅Q⋅dSb/dx(t+Δt)
            @. v = v - Δt′/2*QdSbdx
        end

        # update exp{-Δτ⋅V[x]}
        construct_expnΔτV!(holstein)

        # calculate dSf/dx(t+Δt) value
        iter_t = calc_dSfdx!(hmc, holstein, preconditioner)
        iters += iter_t
        println("Iters = ",iter_t)

        # dSf/dx(t+Δt) ==> Q⋅dSf/dx(t+Δt)
        fourier_accelerate!(QdSfdx,fa,dSfdx,-1.0,use_mass=true)

        # v(t+Δt) = v(t) - Δt/2⋅Q⋅dSf/dx(t)
        @. v = v - Δt/2*QdSfdx

        # calculate energy
        H₁, S, K = calc_H(hmc, holstein, fa)
        println("S = ",S)

        # update change in energy
        ΔH̃ += H₁-H₀

        # apply BDP Thermostat
        bdp_thermostat!(v,hmc.u,K,hmc.τ,hmc.Δt)
    end

    # calculate probability of acceptance
    P = min(1.0, exp(-ΔH̃))

    # get the number of iterations
    iters = cld(iters,Nt)

    # Metropolis-Hasting Accept/Reject Step
    if rand() < P # if accepted

        println("Accepted")
        return true, T1(iters)

    else # if rejected

        # reset to original phonon field
        copyto!(x,x0)

        # reset to reflected original velocities.
        # note: this does not do anything unless doing partial momentum refreshes.
        @. v = -v0

        # update exp{-Δτ⋅V[x]}
        construct_expnΔτV!(holstein)

        println("Rejected")
        return false, T1(iters)
    end
end


"""
The kinetics energy in our dyanics is given by `K=v⋅Q⁻¹⋅v/2`, so therefore it is possible to
refresh the velocity `v` according `v=√(Q)⋅R` where `R` is a vector of normal random numbers
and `Q` is the fourier acceleration matrix. More specifically, this function supports partial
momentum refreshes of the form `v = α⋅v + √(1-α²)⋅v′` where `v′=√(Q)⋅R`⋅
"""
function refresh_v!(hmc::HybridMonteCarlo{T},fa::FourierAccelerator{T}) where {T<:AbstractFloat}

    R       = hmc.R
    sqrtQR  = hmc.R
    v       = hmc.v
    α       = hmc.α

    randn!(R)
    fourier_accelerate!(sqrtQR,fa,R,-0.5,use_mass=true)
    @. v = α*v + sqrt(1.0-α^2)*sqrtQR

    return nothing
end


"""
Refresh `ϕ` according to the relationship `ϕ ~ Mᵀ⋅R` where `R` is a vector of normal random numbers.
"""
function refresh_ϕ!(hmc::HybridMonteCarlo{T1},holstein::HolsteinModel{T1,T2},fa::FourierAccelerator{T1}) where {T1<:AbstractFloat,T2<:Number}

    R     = hmc.R

    ϕ₊    = hmc.ϕ₊
    M⁻ᵀϕ₊ = hmc.M⁻ᵀϕ₊
    O⁻¹ϕ₊ = hmc.O⁻¹ϕ₊
    ϕ₋    = hmc.ϕ₋
    M⁻ᵀϕ₋ = hmc.M⁻ᵀϕ₋
    O⁻¹ϕ₋ = hmc.O⁻¹ϕ₋

    M⁻ᵀϕ₊′ = hmc.M⁻ᵀϕ₊′
    O⁻¹ϕ₊′ = hmc.O⁻¹ϕ₊′
    M⁻ᵀϕ₋′ = hmc.M⁻ᵀϕ₋′
    O⁻¹ϕ₋′ = hmc.O⁻¹ϕ₋′

    # REFRESH ϕ₊

    # ϕ₊ = Mᵀ⋅R₊
    randn!(R)
    mulMᵀ!(ϕ₊,holstein,R)

    # intially M⁻ᵀ⋅ϕ₊ = R₊
    copyto!(M⁻ᵀϕ₊,R)
    copyto!(M⁻ᵀϕ₊′,R)

    fill!(O⁻¹ϕ₊,0.0)
    fill!(O⁻¹ϕ₊′,0.0)

    # REFRESH ϕ₋

    # ϕ₋ = Mᵀ⋅R₋
    randn!(R)
    mulMᵀ!(ϕ₋,holstein,R)

    # intially M⁻ᵀ⋅ϕ₋ = R₋
    copyto!(M⁻ᵀϕ₋,R)
    copyto!(M⁻ᵀϕ₋′,R)

    fill!(O⁻¹ϕ₋,0.0)
    fill!(O⁻¹ϕ₋′,0.0)

    return nothing
end


"""
Calculate the total energy `H = K + S`.
"""
function calc_H(hmc::HybridMonteCarlo{T1}, holstein::HolsteinModel{T1,T2}, fa::FourierAccelerator{T1}) where {T1<:AbstractFloat,T2<:Number}
    
    S = calc_S(hmc, holstein)
    K = calc_K(hmc,fa)
    H = S + K

    return H, S, K
end


"""
Calculate the kintetic energy `K = v⋅Q⁻¹⋅v/2` where `Q` is the acceleration matrix.
"""
function calc_K(hmc::HybridMonteCarlo{T}, fa::FourierAccelerator{T})::T where {T<:AbstractFloat}

    v    = hmc.v
    Q⁻¹v = hmc.u
    fourier_accelerate!(Q⁻¹v,fa,v,1.0,use_mass=true)
    K = dot(v,Q⁻¹v)/2

    return K
end

"""
Calcualte the action S = Sb + ϕ₊ᵀ⋅O⁻¹⋅ϕ₊/2 + ϕ₋ᵀ⋅O⁻¹⋅ϕ/2
"""
function calc_S(hmc::HybridMonteCarlo{T1}, holstein::HolsteinModel{T1,T2})::T1 where {T1<:AbstractFloat,T2<:Number}
    
    # S = ϕ₊ᵀ⋅O⁻¹⋅ϕ₊/2 + ϕ₋ᵀ⋅O⁻¹⋅ϕ₋/2
    S = calc_Sf(hmc)

    # S = Sb + ϕ₊ᵀ⋅O⁻¹⋅ϕ₊/2 + ϕ₋ᵀ⋅O⁻¹⋅ϕ₋/2
    S += calc_Sbose(holstein)

    return S
end

"""
Calculate the derivative of the action dS/dx = dSb/dx - ϕ₊ᵀ⋅O⁻ᵀ⋅[Mᵀ⋅dM/dx]⋅O⁻¹⋅ϕ₊ - ϕ₋ᵀ⋅O⁻ᵀ⋅[Mᵀ⋅dM/dx]⋅O⁻¹⋅ϕ₋
"""
function calc_dSdx!(hmc::HybridMonteCarlo{T1}, holstein::HolsteinModel{T1,T2}, preconditioner=I)::Int where {T1<:AbstractFloat,T2<:Number}
    
    dSdx = hmc.dSdx
    
    # dS/dx = -ϕ₊ᵀ⋅O⁻ᵀ⋅[dMᵀ/dx⋅M]⋅O⁻¹⋅ϕ₊ - ϕ₋ᵀ⋅O⁻ᵀ⋅[dMᵀ/dx⋅M]⋅O⁻¹⋅ϕ₋
    iters = calc_dSfdx!(hmc,holstein,preconditioner)

    # dS/dx = dSb/dx - ϕ₊ᵀ⋅O⁻ᵀ⋅[dMᵀ/dx⋅M]⋅O⁻¹⋅ϕ₊ - ϕ₋ᵀ⋅O⁻ᵀ⋅[dMᵀ/dx⋅M]⋅O⁻¹⋅ϕ₋
    calc_dSbosedx!(dSdx, holstein)

    return iters
end


"""
Calculate the fermionic action S = ϕ₊ᵀ⋅O⁻¹⋅ϕ₊/2 + ϕ₋ᵀ⋅O⁻¹⋅ϕ/2 = ϕ₊ᵀ⋅[Mᵀ⋅M]⁻¹⋅ϕ₊/2 + ϕ₋ᵀ⋅[Mᵀ⋅M]⁻¹⋅ϕ/2
"""
function calc_Sf(hmc::HybridMonteCarlo{T})::T where {T<:AbstractFloat}

    ϕ₊    = hmc.ϕ₊
    O⁻¹ϕ₊ = hmc.O⁻¹ϕ₊
    ϕ₋    = hmc.ϕ₋
    O⁻¹ϕ₋ = hmc.O⁻¹ϕ₋

    Sf    = dot(ϕ₊,O⁻¹ϕ₊)/2
    Sf   += dot(ϕ₋,O⁻¹ϕ₋)/2

    return Sf
end

"""
Calculate the derivative of the fermionic action `dSf/dx = -ϕ₊ᵀ⋅O⁻ᵀ⋅[Mᵀ⋅dM/dx]⋅O⁻¹⋅ϕ₊ + ₋ϕᵀ⋅O⁻ᵀ⋅[Mᵀ⋅dM/dx]⋅O⁻¹⋅ϕ₋` where `O=MᵀM`.
More specicially each partial derivative `∂S/∂xᵢ(τ)` will be stored to the corresponding element in the array dSdx.
"""
function calc_dSfdx!(hmc::HybridMonteCarlo{T1}, holstein::HolsteinModel{T1,T2}, preconditioner=I)::Int where {T1<:AbstractFloat,T2<:Number}
    
    dSdx     = hmc.dSdx
    dSdx′    = hmc.R
    dMdxO⁻¹ϕ = hmc.R
    MO⁻¹ϕ    = hmc.u
    
    O⁻¹ϕ₊ = hmc.O⁻¹ϕ₊
    O⁻¹ϕ₋ = hmc.O⁻¹ϕ₋

    # calculate O⁻¹⋅ϕ₊ and O⁻¹⋅ϕ₋
    iters = calc_O⁻¹ϕ!(hmc,holstein,preconditioner)

    # calculate dM/dx⋅O⁻¹⋅ϕ₊
    muldMdx!(dMdxO⁻¹ϕ,holstein,O⁻¹ϕ₊)

    # calculate M⋅O⁻¹⋅ϕ₊
    mulM!(MO⁻¹ϕ,holstein,O⁻¹ϕ₊)

    # calculate -ϕ₊ᵀ⋅O⁻ᵀ⋅[Mᵀ⋅dM/dx]⋅O⁻¹⋅ϕ₊ = -[M⋅O⁻¹⋅ϕ₊]ᵀ⋅[dM/dx⋅O⁻¹⋅ϕ₊]
    @. dSdx = - MO⁻¹ϕ * dMdxO⁻¹ϕ

    # calculate dM/dx⋅O⁻¹⋅ϕ₋
    muldMdx!(dMdxO⁻¹ϕ,holstein,O⁻¹ϕ₋)

    # calculate M⋅O⁻¹⋅ϕ₋
    mulM!(MO⁻¹ϕ,holstein,O⁻¹ϕ₋)

    # calculate -ϕ₋ᵀ⋅O⁻ᵀ⋅[Mᵀ⋅dM/dx]⋅O⁻¹⋅ϕ₋ = -[M⋅O⁻¹⋅ϕ₋]ᵀ⋅[dM/dx⋅O⁻¹⋅ϕ₋]
    @. dSdx′ = dSdx - MO⁻¹ϕ * dMdxO⁻¹ϕ

    # In the section of code below there is a subtle detail that is addressed that results from defining
    # our M matrix with the -B[τ] matrices on the upper diagonal instead of the lower diagonal.
    # After doing matrix-vector multiplies by all the  dM/dx's, the expectation value for the partial
    # derivatives corresponding to the τ time slice lives in the array indices corresponding to τ-1.
    # Therefore, the values need to be shifted one time slice forward. This is done by first calculating
    # and storing the ∂Sf/∂xᵢ(τ) partial derivative values in the vector dSdx′, and then copying a properly
    # shifted version into the final vector dSdx.
    @inbounds @fastmath for site in 1:holstein.nsites
        for τ in 1:holstein.Lτ
            τp1           = τ%holstein.Lτ+1
            idx_τ         = get_index(τ,   site, holstein.Lτ)
            idx_τp1       = get_index(τp1, site, holstein.Lτ)
            dSdx[idx_τp1] = dSdx′[idx_τ]
        end
    end

    return iters
end

"""
Solve `O⋅x=ϕ₊ ==> x=O⁻¹⋅ϕ₊` and `O⋅x=ϕ₋ ==> x=O⁻¹⋅ϕ₋` where `O = Mᵀ⋅M`.
"""
function calc_O⁻¹ϕ!(hmc::HybridMonteCarlo{T1}, holstein::HolsteinModel{T1,T2}, preconditioner=I)::Int where {T1<:AbstractFloat,T2<:Number}

    ϕ₊     = hmc.ϕ₊
    M⁻ᵀϕ₊  = hmc.M⁻ᵀϕ₊
    O⁻¹ϕ₊  = hmc.O⁻¹ϕ₊
    ϕ₋     = hmc.ϕ₋
    M⁻ᵀϕ₋  = hmc.M⁻ᵀϕ₋
    O⁻¹ϕ₋  = hmc.O⁻¹ϕ₋

    M⁻ᵀϕ₊′ = hmc.M⁻ᵀϕ₊′
    O⁻¹ϕ₊′ = hmc.O⁻¹ϕ₊′
    M⁻ᵀϕ₋′ = hmc.M⁻ᵀϕ₋′
    O⁻¹ϕ₋′ = hmc.O⁻¹ϕ₋′
    
    u      = hmc.u

    #######################
    ## CALCULATE  O⁻¹⋅ϕ₊ ##
    #######################

    # count iterative solver iterations
    iters = 0

    if !holstein.mul_by_M # if using Conjugate Gradient

        if hmc.construct_guess
            # construct initial guess for solution to linear system
            copyto!(u,O⁻¹ϕ₊)
            @. O⁻¹ϕ₊ = 2*O⁻¹ϕ₊ - O⁻¹ϕ₊′
            copyto!(O⁻¹ϕ₊′,u)
        else
            fill!(O⁻¹ϕ₊,0.0)
        end

        # solve linear system
        iters += ldiv!(O⁻¹ϕ₊,holstein,ϕ₊,preconditioner)

    else # if using GMRES

        # setup the precontioer
        KPMPreconditioners.setup!(preconditioner)

        if hmc.construct_guess
            # construct initial guess for solution to linear system
            copyto!(u,M⁻ᵀϕ₊)
            @. M⁻ᵀϕ₊ = 2*M⁻ᵀϕ₊ - M⁻ᵀϕ₊′
            copyto!(M⁻ᵀϕ₊′,u)
        else
            fill!(M⁻ᵀϕ₊,0.0)
        end

        # solve linear system
        holstein.transposed=true
        iters += ldiv!(M⁻ᵀϕ₊,holstein,ϕ₊,preconditioner)

        if hmc.construct_guess
            # construct initial guess for solution to linear system
            copyto!(u,O⁻¹ϕ₊)
            @. O⁻¹ϕ₊ = 2*O⁻¹ϕ₊ - O⁻¹ϕ₊′
            copyto!(O⁻¹ϕ₊′,u)
        else
            fill!(O⁻¹ϕ₊,0.0)
        end

        # solve linear system
        holstein.transposed=false
        iters += ldiv!(O⁻¹ϕ₊,holstein,M⁻ᵀϕ₊,preconditioner)
    end

    #######################
    ## CALCULATE  O⁻¹⋅ϕ₋ ##
    #######################

    if !holstein.mul_by_M

        if hmc.construct_guess
            # construct initial guess for solution to linear system
            copyto!(u,O⁻¹ϕ₋)
            @. O⁻¹ϕ₋ = 2*O⁻¹ϕ₋ - O⁻¹ϕ₋′
            copyto!(O⁻¹ϕ₋′,u)
        else
            fill!(O⁻¹ϕ₋,0.0)
        end

        # solve linear system
        iters += ldiv!(O⁻¹ϕ₋,holstein,ϕ₋,preconditioner)
        
    else
        
        if hmc.construct_guess
            # construct initial guess for solution to linear system
            copyto!(u,M⁻ᵀϕ₋)
            @. M⁻ᵀϕ₋ = 2*M⁻ᵀϕ₋ - M⁻ᵀϕ₋′
            copyto!(M⁻ᵀϕ₋′,u)
        else
            fill!(M⁻ᵀϕ₋,0.0)
        end

        # solve linear system
        holstein.transposed=true
        iters += ldiv!(M⁻ᵀϕ₋,holstein,ϕ₋,preconditioner)

        if hmc.construct_guess
            # construct initial guess for solution to linear system
            copyto!(u,O⁻¹ϕ₋)
            @. O⁻¹ϕ₋ = 2*O⁻¹ϕ₋ - O⁻¹ϕ₋′
            copyto!(O⁻¹ϕ₋′,u)
        else
            fill!(O⁻¹ϕ₋,0.0)
        end

        # solve linear system
        holstein.transposed=false
        iters += ldiv!(O⁻¹ϕ₋,holstein,M⁻ᵀϕ₋,preconditioner)
    end

    # accounting for the fact that there is a spin up and spin down
    # linear system that needs to be solved
    iters = cld(iters,2)

    return iters
end

"""
Apply the BDP thermostat as defined in equation A7 of the appendix of the paper
"Canonical sampling through velocity rescaling"
"""
function bdp_thermostat!(v::AbstractVector{T},R::AbstractVector{T},K::T,τ::T,Δt::T) where {T<:AbstractFloat}

    if isfinite(τ)
        randn!(R)
        R² = norm(R)^2
        R₁ = R[1]
        N  = length(v)
        K̄  = N/2
        c  = exp(-Δt/τ)
        α² = c + K̄/(N*K)*(1-c)*R² + 2*sqrt(K̄/(N*K)*c*(1-c))*R₁
        @. v = sqrt(α²) * v
    end

    return nothing
end

end