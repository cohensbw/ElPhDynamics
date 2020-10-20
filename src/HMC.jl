module HMC

using Random
using UnsafeArrays
using LinearAlgebra
using Printf
using Parameters
using Printf
using Statistics
using Logging

using ..Utilities: get_index
using ..Models: AbstractModel, HolsteinModel, SSHModel, update_model!, mulM!, muldMdx!, mulMᵀ!
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
    Dimension of M matrix.
    """
    Ndim::Int

    """
    Store initial phonon fields.
    """
    x0::Vector{T}

    """
    Time between refreshing the momentum p and auxialiary fields ϕ.
    """
    tr::T

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
    Temporary storage vector of length Ndim.
    """
    u::Vector{T}

    """
    Temporary storage vector of length Ndof.
    """
    y::Vector{T}
    
    ######################
    ## Status Variables ##
    ######################

    """
    Whether to log HMC data to file.
    """
    log::Bool

    """
    Whether to log HMC data at every timestep.
    """
    verbose::Bool

    """
    log filename
    """
    logfile::String

    """
    How many HMC updates have been performed.
    """
    updates::Int

    """
    Timestep of current HMC update.
    """
    t::Int

    """
    Whether most recent HMC update was accepted or rejected.
    """
    accepted::Bool

    """
    Total energy.
    """
    H::T

    """
    Action.
    """
    S::T

    """
    Kintetic energy
    """
    K::T

    """
    Iteration Count
    """
    iters::Int

    function HybridMonteCarlo(model::AbstractModel,Δt::T,tr::T,α::T,Nb::Int,construct_guess::Bool;
                              log::Bool=false, verbose::Bool=false, logfile::String="") where {T<:AbstractFloat}

        # partial momentum refresh parameter
        @assert 0.0 <= α < 1.0

        Ndof   = model.Ndof
        Ndim   = model.Ndim

        x0     = zeros(T,Ndof)
        dSdx   = zeros(T,Ndof)
        v      = randn(T,Ndof)
        v0     = zeros(T,Ndof)

        R      = zeros(T,Ndim)
        ϕ₊     = zeros(T,Ndim)
        M⁻ᵀϕ₊  = zeros(T,Ndim)
        O⁻¹ϕ₊  = zeros(T,Ndim)
        ϕ₋     = zeros(T,Ndim)
        M⁻ᵀϕ₋  = zeros(T,Ndim)
        O⁻¹ϕ₋  = zeros(T,Ndim)
        M⁻ᵀϕ₊′ = zeros(T,Ndim)
        O⁻¹ϕ₊′ = zeros(T,Ndim)
        M⁻ᵀϕ₋′ = zeros(T,Ndim)
        O⁻¹ϕ₋′ = zeros(T,Ndim)

        u      = zeros(T,Ndim)
        y      = zeros(T,Ndof)

        # the action
        H = 0.0::T

        # number of timesteps
        Nt = round(Int,tr/Δt)

        # size of smaller timestep for Sb
        Δt′ = Δt/Nb

        # checking conditions on parameters
        @assert 0.0 <= α < 1.0

        updates  = 0
        t        = 0
        accepted = false
        H        = 0.0
        S        = 0.0
        K        = 0.0
        iters    = 0

        if log
            open(logfile,"w") do fout
                write(fout,"updates accepted timestep tot_energy action kin_energy iters\n")
            end
        end

        return new{T}(Ndof, Ndim, x0, tr, Δt, Nt, Δt′, Nb, α, dSdx, v, v0, R, ϕ₊, ϕ₋, M⁻ᵀϕ₊, M⁻ᵀϕ₊′, M⁻ᵀϕ₋, M⁻ᵀϕ₋′, O⁻¹ϕ₊, O⁻¹ϕ₊′, O⁻¹ϕ₋, O⁻¹ϕ₋′, construct_guess, u, y,
                     log, verbose, logfile, updates, t, accepted, H, S, K, iters)
    end

    function HybridMonteCarlo(hmc::HybridMonteCarlo{T},Δt::T,tr::T,α::T,Nb::Int,construct_guess::Bool;
                              log::Bool=false, verbose::Bool=false, logfile::String="") where {T<:AbstractFloat}

        @unpack Ndof, Ndim, x0, H, dSdx, v, v0, R, ϕ₊, ϕ₋, M⁻ᵀϕ₊, M⁻ᵀϕ₊′, M⁻ᵀϕ₋, M⁻ᵀϕ₋′, O⁻¹ϕ₊, O⁻¹ϕ₊′, O⁻¹ϕ₋, O⁻¹ϕ₋′, u, y = hmc
        Nt  = round(Int,tr/Δt)
        Δt′ = Δt/Nb

        updates  = 0
        t        = 0
        accepted = false
        H        = 0.0
        S        = 0.0
        K        = 0.0
        iters    = 0

        if log
            open(logfile,"w") do fout
                write(fout,"updates accepted timestep tot_energy action kin_energy iters\n")
            end
        end
    
        return new{T}(Ndof, Ndim, x0, tr, Δt, Nt, Δt′, Nb, α, dSdx, v, v0, R, ϕ₊, ϕ₋, M⁻ᵀϕ₊, M⁻ᵀϕ₊′, M⁻ᵀϕ₋, M⁻ᵀϕ₋′, O⁻¹ϕ₊, O⁻¹ϕ₊′, O⁻¹ϕ₋, O⁻¹ϕ₋′, construct_guess, u, y,
                      log, verbose, logfile, updates, t, accepted, H, S, K, iters)
    end
end


"""
Write status of HMC to log file.
"""
function update_log(hmc::HybridMonteCarlo{T}) where {T<:AbstractFloat}

    @unpack updates, t, accepted, H, S, K, iters = hmc

    if t==-1
        # outcome of HMC update accept/reject decision
        outcome = Int(accepted)
    else
        # trajectory ongoing, accept/reject decision comes after HMC trajectory
        outcome = -1
    end

    open(hmc.logfile,"a") do fout
        write(fout, @sprintf("%d %d %d %.3f %.3f %.3f %d\n", updates, outcome, t, H, S, K, iters))
    end
    return nothing
end


"""
Do a Hybrid Monte Carlo update to the phonon fields.
"""
function update!(model::AbstractModel{T1,T2}, hmc::HybridMonteCarlo{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Tuple{Bool,T1}  where {T1,T2}

    # only do an update if the model has a non-zero number of degrees of freedom.
    if hmc.Ndof > 0

        # increment HMC update counter
        hmc.updates += 1

        # set HMC timestep to zero
        hmc.t = 0

        if hmc.Nb==1
            accepted, iters = standard_update!(model,hmc,fa,preconditioner)
        else
            accepted, iters = multitimestep_update!(model,hmc,fa,preconditioner)
        end

        # write output of logfile
        if hmc.log
            hmc.t = -1
            update_log(hmc)
        end

        return accepted, iters
    else
        return true, 0
    end
end

"""
Standard HMC update.
"""
function standard_update!(model::AbstractModel{T1,T2}, hmc::HybridMonteCarlo{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Tuple{Bool,T1}  where {T1,T2}

    x     = model.x
    x0    = hmc.x0
    v0    = hmc.v0
    dSdx  = hmc.dSdx
    QdSdx = hmc.dSdx
    v     = hmc.v
    Nt    = hmc.Nt
    Δt    = hmc.Δt

    # update exp{-Δτ⋅V[x]}
    update_model!(model)

    # refresh the velocity v
    refresh_v!(hmc,fa)

    # refresh ϕ
    refresh_ϕ!(hmc,model)

    # calculate O⁻¹⋅ϕ₊ and O⁻¹⋅ϕ₋
    iters = calc_O⁻¹ϕ!(hmc,model,preconditioner,2.0)

    # calculate initial energy
    H₀, S, K = calc_H(hmc, model, fa)

    # calculate the initial dS/dx value
    iter_t = calc_dSdx!(hmc, model, preconditioner)
    iters  = iter_t

    # dS/dx(t+Δt) ==> Q⋅dS/dx(t+Δt)
    fourier_accelerate!(QdSdx,fa,dSdx,-1.0,use_mass=true)

    # record intial state
    copyto!(x0,x)
    copyto!(v0,v)

    # log HMC state
    if hmc.log & hmc.verbose
        update_log(hmc)
    end

    # iterate over time steps, doing leapfrog updates to the phonon fields
    for hmc.t in 1:Nt

        # v(t+Δt/2) = v(t) - Δt/2⋅Q⋅dS/dx(t)
        @. v = v - Δt/2*QdSdx

        # x(t+Δt) = x(t) + Δt⋅v(t+Δt/2)
        @. x = x + Δt*v

        # update exp{-Δτ⋅V[x]}
        update_model!(model)

        # calculate O⁻¹⋅ϕ₊ and O⁻¹⋅ϕ₋
        iters += calc_O⁻¹ϕ!(hmc,model,preconditioner,1.0)

        # calculate dS/dx(t+Δt) value
        calc_dSdx!(hmc, model, preconditioner)

        # dS/dx(t+Δt) ==> Q⋅dS/dx(t+Δt)
        fourier_accelerate!(QdSdx,fa,dSdx,-1.0,use_mass=true)

        # v(t+Δt) = v(t+Δt/2) - Δt/2⋅Q⋅dS/dx(t+Δt)
        @. v = v - Δt/2*QdSdx

        # log HMC state
        if hmc.log & hmc.verbose
            update_log(hmc)
        end
    end

    # calculate O⁻¹⋅ϕ₊ and O⁻¹⋅ϕ₋
    iters += calc_O⁻¹ϕ!(hmc,model,preconditioner,2.0)

    # calculate final energy
    H₁, S, K = calc_H(hmc, model, fa)

    # calculate change in energy
    ΔH = H₁ - H₀

    # calculate probability of acceptance
    P = min(1.0, exp(-ΔH))

    # Metropolis-Hasting Accept/Reject Step
    if rand() < P # if accepted

        hmc.accepted = true
        return hmc.accepted, T1(cld(iters,Nt+2))

    else # if rejected

        # reset to original phonon field
        copyto!(x,x0)

        # reset to reflected original velocities.
        # note: this does not do anything unless doing partial momentum refreshes.
        @. v = -v0

        # update exp{-Δτ⋅V[x]}
        update_model!(model)

        hmc.accepted = false
        return hmc.accepted, T1(cld(iters,Nt+2))
    end
end


"""
Multi-timestepping HMC update.
"""
function multitimestep_update!(model::AbstractModel{T1,T2}, hmc::HybridMonteCarlo{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Tuple{Bool,T1}  where {T1,T2}

    x      = model.x
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
    update_model!(model)

    # refresh the velocity v
    refresh_v!(hmc,fa)

    # refresh ϕ
    refresh_ϕ!(hmc,model)

    # calculate O⁻¹⋅ϕ₊ and O⁻¹⋅ϕ₋
    iters = calc_O⁻¹ϕ!(hmc,model,preconditioner,2.0)

    # calculate energy
    H₀, S, K = calc_H(hmc, model, fa)

    # calculate the initial dSf/dx value
    calc_dSfdx!(hmc, model, preconditioner)

    # dSf/dx(t+Δt) ==> Q⋅dSf/dx(t+Δt)
    fourier_accelerate!(QdSfdx,fa,dSfdx,-1.0,use_mass=true)

    # record intial phonon configuration
    copyto!(x0,x)
    copyto!(v0,v)

    # log HMC state
    if hmc.log & hmc.verbose
        update_log(hmc)
    end

    # iterate over timesteps
    for hmc.t in 1:Nt

        # v(t+Δt/2) = v(t) - Δt/2⋅Q⋅dSf/dx(t)
        @. v = v - Δt/2*QdSfdx

        # calculate the initial dSb/dx value
        fill!(dSbdx,0.0)
        calc_dSbosedx!(dSbdx,model)

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
            calc_dSbosedx!(dSbdx,model)

            # dS/dx(t+Δt) ==> Q⋅dS/dx(t+Δt)
            fourier_accelerate!(QdSbdx,fa,dSbdx,-1.0,use_mass=true)

            # v(t+Δt) = v(t+Δt/2) - Δt/2⋅Q⋅dSb/dx(t+Δt)
            @. v = v - Δt′/2*QdSbdx
        end

        # update exp{-Δτ⋅V[x]}
        update_model!(model)

        # calculate O⁻¹⋅ϕ₊ and O⁻¹⋅ϕ₋
        iters += calc_O⁻¹ϕ!(hmc,model,preconditioner,1.0)

        # calculate dSf/dx(t+Δt) value
        calc_dSfdx!(hmc, model, preconditioner)

        # dSf/dx(t+Δt) ==> Q⋅dSf/dx(t+Δt)
        fourier_accelerate!(QdSfdx,fa,dSfdx,-1.0,use_mass=true)

        # v(t+Δt) = v(t) - Δt/2⋅Q⋅dSf/dx(t)
        @. v = v - Δt/2*QdSfdx

        # log HMC state
        if hmc.log & hmc.verbose
            update_log(hmc)
        end
    end

    # calculate O⁻¹⋅ϕ₊ and O⁻¹⋅ϕ₋
    iters += calc_O⁻¹ϕ!(hmc,model,preconditioner,2.0)

    # calculate final energy
    H₁, S, K = calc_H(hmc, model, fa)

    # calculate change in energy
    ΔH = H₁ - H₀

    # calculate probability of acceptance
    P = min(1.0, exp(-ΔH))

    # Metropolis-Hasting Accept/Reject Step
    if rand() < P # if accepted

        hmc.accepted = true
        return hmc.accepted, T1(cld(iters,Nt+2))

    else # if rejected

        # reset to original phonon field
        copyto!(x,x0)

        # reset to reflected original velocities.
        # note: this does not do anything unless doing partial momentum refreshes.
        @. v = -v0

        # update exp{-Δτ⋅V[x]}
        update_model!(model)

        hmc.accepted = false
        return hmc.accepted, T1(cld(iters,Nt+2))
    end
end


"""
The kinetics energy in our dyanics is given by `K=v⋅Q⁻¹⋅v/2`, so therefore it is possible to
refresh the velocity `v` according `v=√(Q)⋅R` where `R` is a vector of normal random numbers
and `Q` is the fourier acceleration matrix. More specifically, this function supports partial
momentum refreshes of the form `v = α⋅v + √(1-α²)⋅v′` where `v′=√(Q)⋅R`⋅
"""
function refresh_v!(hmc::HybridMonteCarlo{T},fa::FourierAccelerator{T}) where {T<:AbstractFloat}

    R       = hmc.y
    sqrtQR  = hmc.y
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
function refresh_ϕ!(hmc::HybridMonteCarlo{T1},model::AbstractModel{T1,T2}) where {T1,T2}

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
    mulMᵀ!(ϕ₊,model,R)

    # intially M⁻ᵀ⋅ϕ₊ = R₊
    copyto!(M⁻ᵀϕ₊ ,R)
    copyto!(M⁻ᵀϕ₊′,R)

    fill!(O⁻¹ϕ₊ ,0.0)
    fill!(O⁻¹ϕ₊′,0.0)

    # REFRESH ϕ₋

    # ϕ₋ = Mᵀ⋅R₋
    randn!(R)
    mulMᵀ!(ϕ₋,model,R)

    # intially M⁻ᵀ⋅ϕ₋ = R₋
    copyto!(M⁻ᵀϕ₋ ,R)
    copyto!(M⁻ᵀϕ₋′,R)

    fill!(O⁻¹ϕ₋ ,0.0)
    fill!(O⁻¹ϕ₋′,0.0)

    return nothing
end


"""
Calculate the total energy `H = K + S`.
"""
function calc_H(hmc::HybridMonteCarlo{T1}, model::AbstractModel{T1,T2}, fa::FourierAccelerator{T1}) where {T1,T2}
    
    S = calc_S(hmc, model)
    K = calc_K(hmc,fa)
    hmc.H = S + K

    return hmc.H, S, K
end


"""
Calculate the kintetic energy `K = v⋅Q⁻¹⋅v/2` where `Q` is the acceleration matrix.
"""
function calc_K(hmc::HybridMonteCarlo{T}, fa::FourierAccelerator{T})::T where {T<:AbstractFloat}

    v    = hmc.v
    Q⁻¹v = hmc.y
    fourier_accelerate!(Q⁻¹v,fa,v,1.0,use_mass=true)
    hmc.K = dot(v,Q⁻¹v)/2

    return hmc.K
end

"""
Calcualte the action S = Sb + ϕ₊ᵀ⋅O⁻¹⋅ϕ₊/2 + ϕ₋ᵀ⋅O⁻¹⋅ϕ/2
"""
function calc_S(hmc::HybridMonteCarlo{T1}, model::AbstractModel{T1,T2})::T1 where {T1,T2}
    
    # S = ϕ₊ᵀ⋅O⁻¹⋅ϕ₊/2 + ϕ₋ᵀ⋅O⁻¹⋅ϕ₋/2
    hmc.S = calc_Sf(hmc)

    # S = Sb + ϕ₊ᵀ⋅O⁻¹⋅ϕ₊/2 + ϕ₋ᵀ⋅O⁻¹⋅ϕ₋/2
    hmc.S += calc_Sbose(model)

    return hmc.S
end

"""
Calculate the derivative of the action dS/dx = dSb/dx - ϕ₊ᵀ⋅O⁻ᵀ⋅[Mᵀ⋅dM/dx]⋅O⁻¹⋅ϕ₊ - ϕ₋ᵀ⋅O⁻ᵀ⋅[Mᵀ⋅dM/dx]⋅O⁻¹⋅ϕ₋
"""
function calc_dSdx!(hmc::HybridMonteCarlo{T1}, model::AbstractModel{T1,T2}, preconditioner=I) where {T1,T2}
    
    dSdx = hmc.dSdx
    
    # dS/dx = -ϕ₊ᵀ⋅O⁻ᵀ⋅[dMᵀ/dx⋅M]⋅O⁻¹⋅ϕ₊ - ϕ₋ᵀ⋅O⁻ᵀ⋅[dMᵀ/dx⋅M]⋅O⁻¹⋅ϕ₋
    calc_dSfdx!(hmc,model,preconditioner)

    # dS/dx = dSb/dx - ϕ₊ᵀ⋅O⁻ᵀ⋅[dMᵀ/dx⋅M]⋅O⁻¹⋅ϕ₊ - ϕ₋ᵀ⋅O⁻ᵀ⋅[dMᵀ/dx⋅M]⋅O⁻¹⋅ϕ₋
    calc_dSbosedx!(dSdx, model)

    return nothing
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
function calc_dSfdx!(hmc::HybridMonteCarlo{T1}, model::AbstractModel{T1,T2}, preconditioner=I) where {T1,T2}
    
    dSdx  = hmc.dSdx
    dMdx  = hmc.y
    MO⁻¹ϕ = hmc.u
    
    O⁻¹ϕ₊ = hmc.O⁻¹ϕ₊
    O⁻¹ϕ₋ = hmc.O⁻¹ϕ₋

    # calculate M⋅O⁻¹⋅ϕ₊
    mulM!(MO⁻¹ϕ,model,O⁻¹ϕ₊)

    # calculate -ϕ₊ᵀ⋅O⁻ᵀ⋅[Mᵀ⋅dM/dx]⋅O⁻¹⋅ϕ₊ = -[M⋅O⁻¹⋅ϕ₊]ᵀ⋅dM/dx⋅[O⁻¹⋅ϕ₊]
    muldMdx!(dMdx,MO⁻¹ϕ,model,O⁻¹ϕ₊)
    @. dSdx = -dMdx

    # calculate M⋅O⁻¹⋅ϕ₋
    mulM!(MO⁻¹ϕ,model,O⁻¹ϕ₋)

    # calculate -ϕ₋ᵀ⋅O⁻ᵀ⋅[Mᵀ⋅dM/dx]⋅O⁻¹⋅ϕ₋ = -[M⋅O⁻¹⋅ϕ₋]ᵀ⋅dM/dx⋅[O⁻¹⋅ϕ₋]
    muldMdx!(dMdx,MO⁻¹ϕ,model,O⁻¹ϕ₋)
    @. dSdx = dSdx - dMdx

    return nothing
end

"""
Solve `O⋅x=ϕ₊ ==> x=O⁻¹⋅ϕ₊` and `O⋅x=ϕ₋ ==> x=O⁻¹⋅ϕ₋` where `O = Mᵀ⋅M`.
"""
function calc_O⁻¹ϕ!(hmc::HybridMonteCarlo{T1}, model::AbstractModel{T1,T2}, preconditioner=I, power::T1=1.0)::Int where {T1,T2}

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

    # udpate tolerance used by solver
    tol = model.solver.tol::T1
    model.solver.tol = tol^power

    # count iterative solver iterations
    hmc.iters = 0

    # setup the precontioer
    KPMPreconditioners.setup!(preconditioner)

    if !model.mul_by_M # if using Conjugate Gradient

        if hmc.construct_guess
            # construct initial guess for solution to linear system
            copyto!(u,O⁻¹ϕ₊)
            @. O⁻¹ϕ₊ = 2*O⁻¹ϕ₊ - O⁻¹ϕ₊′
            copyto!(O⁻¹ϕ₊′,u)
        else
            fill!(O⁻¹ϕ₊,0.0)
        end

        # solve linear system
        model.transposed=false
        iters, err = ldiv!(O⁻¹ϕ₊,model,ϕ₊,preconditioner)
        hmc.iters += iters

    else # if using GMRES

        if hmc.construct_guess
            # construct initial guess for solution to linear system
            copyto!(u,M⁻ᵀϕ₊)
            @. M⁻ᵀϕ₊ = 2*M⁻ᵀϕ₊ - M⁻ᵀϕ₊′
            copyto!(M⁻ᵀϕ₊′,u)
        else
            fill!(M⁻ᵀϕ₊,0.0)
        end

        # solve linear system
        model.transposed=true
        iters, err = ldiv!(M⁻ᵀϕ₊,model,ϕ₊,preconditioner)
        hmc.iters += iters

        if hmc.construct_guess
            # construct initial guess for solution to linear system
            copyto!(u,O⁻¹ϕ₊)
            @. O⁻¹ϕ₊ = 2*O⁻¹ϕ₊ - O⁻¹ϕ₊′
            copyto!(O⁻¹ϕ₊′,u)
        else
            fill!(O⁻¹ϕ₊,0.0)
        end

        # solve linear system
        model.transposed=false
        iters, err = ldiv!(O⁻¹ϕ₊,model,M⁻ᵀϕ₊,preconditioner)
        hmc.iters += iters
    end

    #######################
    ## CALCULATE  O⁻¹⋅ϕ₋ ##
    #######################

    if !model.mul_by_M

        if hmc.construct_guess
            # construct initial guess for solution to linear system
            copyto!(u,O⁻¹ϕ₋)
            @. O⁻¹ϕ₋ = 2*O⁻¹ϕ₋ - O⁻¹ϕ₋′
            copyto!(O⁻¹ϕ₋′,u)
        else
            fill!(O⁻¹ϕ₋,0.0)
        end

        # solve linear system
        model.transposed=false
        iters, err = ldiv!(O⁻¹ϕ₋,model,ϕ₋,preconditioner)
        hmc.iters += iters
        
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
        model.transposed = true
        iters, err       = ldiv!(M⁻ᵀϕ₋,model,ϕ₋,preconditioner)
        hmc.iters        += iters

        if hmc.construct_guess
            # construct initial guess for solution to linear system
            copyto!(u,O⁻¹ϕ₋)
            @. O⁻¹ϕ₋ = 2*O⁻¹ϕ₋ - O⁻¹ϕ₋′
            copyto!(O⁻¹ϕ₋′,u)
        else
            fill!(O⁻¹ϕ₋,0.0)
        end

        # solve linear system
        model.transposed = false
        iters, err       = ldiv!(O⁻¹ϕ₋,model,M⁻ᵀϕ₋,preconditioner)
        hmc.iters       += iters
    end

    # accounting for the fact that there is a spin up and spin down
    # linear system that needs to be solved
    hmc.iters = cld(hmc.iters,2)

    # revert solvers tolerance to original value
    model.solver.tol = tol

    return hmc.iters
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