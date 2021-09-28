module HMC

using LinearAlgebra: fill!
using Random
using LinearAlgebra
using Printf
using Parameters
using Printf
using Statistics
using Logging

using ..Utilities: get_index, reshaped, δ
using ..Models: AbstractModel, HolsteinModel, SSHModel, update_model!, mulM!, muldMdx!, mulMᵀ!
using ..PhononAction: calc_dSbdx!, calc_Sb
using ..FourierAcceleration: FourierAccelerator, fourier_accelerate!
import ..KPMPreconditioners

export HybridMonteCarlo, update!, refresh_ϕ!, calc_S

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
    R₊::Vector{T}

    """
    Random gaussian vector.
    """
    R₋::Vector{T}

    """
    Auxiliary fields for spin up electrons.
    """
    ϕ₊::Vector{T}

    """
    Auxiliary fields for spin down electrons.
    """
    ϕ₋::Vector{T}

    """
    The vector Λ⋅ϕ₊
    """
    Λϕ₊::Vector{T}

    """
    The vector Λ⋅ϕ₋
    """
    Λϕ₋::Vector{T}

    """
    O⁻¹⋅ϕ₊ where O=MᵀM
    """
    O⁻¹Λϕ₊::Vector{T}

    """
    O⁻¹⋅ϕ₋ where O=MᵀM
    """
    O⁻¹Λϕ₋::Vector{T}

    """
    Diagonal Λ matrix. Definition varies for SSH and Holstein models.
    """
    Λ::Vector{T}

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
    logfile::IOStream

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

    function HybridMonteCarlo(model::AbstractModel,Δt::T,tr::T,α::T,Nb::Int;
                              log::Bool=false, verbose::Bool=false, logfilename::String="",updates::Int=1) where {T<:AbstractFloat}

        # partial momentum refresh parameter
        @assert 0.0 <= α < 1.0

        Ndof   = model.Ndof
        Ndim   = model.Ndim

        x0     = zeros(T,Ndof)
        dSdx   = zeros(T,Ndof)
        v      = zeros(T,Ndof)
        v0     = zeros(T,Ndof)
        Λ      = ones(T,Ndim)
        R₊     = zeros(T,Ndim)
        R₋     = zeros(T,Ndim)
        ϕ₊     = zeros(T,Ndim)
        Λϕ₊    = zeros(T,Ndim)
        O⁻¹Λϕ₊ = zeros(T,Ndim)
        ϕ₋     = zeros(T,Ndim)
        Λϕ₋    = zeros(T,Ndim)
        O⁻¹Λϕ₋ = zeros(T,Ndim)

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

        t        = 0
        accepted = false
        H        = 0.0
        S        = 0.0
        K        = 0.0
        iters    = 0

        if isfile(logfilename)
            logfile = open(logfilename,"a")
        else
            logfile = open(logfilename,"w")
            write(logfile,"updates accepted timestep tot_energy action kin_energy iters\n")
        end
        if !log
            close(logfile)
        end

        return new{T}(Ndof, Ndim, x0, tr, Δt, Nt, Δt′, Nb, α, dSdx, v, v0, R₊, R₋,
                      ϕ₊, ϕ₋, Λϕ₊, Λϕ₋, O⁻¹Λϕ₊, O⁻¹Λϕ₋, Λ,
                      u, y, log, verbose, logfile, updates, t, accepted, H, S, K, iters)
    end

    function HybridMonteCarlo(hmc::HybridMonteCarlo{T},Δt::T,tr::T,α::T,Nb::Int;
                              log::Bool=false, verbose::Bool=false, logfilename::String="",updates::Int=1) where {T<:AbstractFloat}

        @unpack Ndof, Ndim, x0, H, dSdx, v, v0, R₊, R₋, Λ, ϕ₊, ϕ₋, Λϕ₊, Λϕ₋, O⁻¹Λϕ₊, O⁻¹Λϕ₋, u, y = hmc
        Nt  = round(Int,tr/Δt)
        Δt′ = Δt/Nb

        t        = 0
        accepted = false
        H        = 0.0
        S        = 0.0
        K        = 0.0
        iters    = 0

        if isfile(logfilename)
            logfile = open(logfilename,"a")
        else
            logfile = open(logfilename,"w")
            write(logfile,"updates accepted timestep tot_energy action kin_energy iters\n")
        end
        if !log
            close(logfile)
        end
    
        return new{T}(Ndof, Ndim, x0, tr, Δt, Nt, Δt′, Nb, α, dSdx, v, v0, R₊, R₋,
                      ϕ₊, ϕ₋, Λϕ₊, Λϕ₋, O⁻¹Λϕ₊, O⁻¹Λϕ₋, Λ,
                      u, y, log, verbose, logfile, updates, t, accepted, H, S, K, iters)
    end
end


"""
Write status of HMC to log file.
"""
function update_log(hmc::HybridMonteCarlo{T},model::AbstractModel{T},fa::FourierAccelerator{T}) where {T<:AbstractFloat}

    @unpack updates, t, accepted, iters, logfile = hmc

    # get energies
    H, S, K = calc_H(hmc, model, fa)

    if t==-1
        # outcome of HMC update accept/reject decision
        outcome = Int(accepted)
    else
        # trajectory ongoing, accept/reject decision comes after HMC trajectory
        outcome = -1
    end

    write(logfile, @sprintf("%d %d %d %.8f %.8f %.8f %d\n", updates, outcome, t, H, S, K, iters))
    flush(logfile)

    return nothing
end


"""
Do a Hybrid Monte Carlo update to the phonon fields.
"""
function update!(model::AbstractModel{T1,T2}, hmc::HybridMonteCarlo{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Tuple{Bool,T1}  where {T1,T2}

    # only do an update if the model has a non-zero number of degrees of freedom.
    if hmc.Ndof > 0

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
            update_log(hmc,model,fa)
        end

        # increment HMC update counter
        hmc.updates += 1

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

    # keep track for linear solve flag
    flag = 0

    # keep track of iterations
    iters = 0

    # update exp{-Δτ⋅V[x]}
    update_model!(model)

    # refresh the velocity v
    refresh_v!(hmc,model,fa)

    # record intial state
    copyto!(x0,x)
    copyto!(v0,v)

    # refresh ϕ
    refresh_ϕ!(hmc,model)

    # calculate O⁻¹⋅ϕ₊ and O⁻¹⋅ϕ₋
    iters, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)

    if iszero(flag)

        # calculate initial energy
        H₀, S, K = calc_H(hmc, model, fa)

        # calculate the initial dS/dx value
        fill!(dSdx,0.0)
        calc_dSdx!(hmc, model)

        # dS/dx(t+Δt) ==> Q⋅dS/dx(t+Δt)
        fourier_accelerate!(QdSdx,fa,dSdx,-1.0,use_mass=true)

        # log HMC state
        if hmc.log && hmc.verbose
            update_log(hmc,model,fa)
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
            itrs, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,1.0)
            iters += itrs

            # kill trajectory if error flag
            if flag > 0
                break
            end

            # calculate dS/dx(t+Δt) value
            fill!(dSdx,0.0)
            calc_dSdx!(hmc, model)

            # dS/dx(t+Δt) ==> Q⋅dS/dx(t+Δt)
            fourier_accelerate!(QdSdx,fa,dSdx,-1.0,use_mass=true)

            # v(t+Δt) = v(t+Δt/2) - Δt/2⋅Q⋅dS/dx(t+Δt)
            @. v = v - Δt/2*QdSdx

            # log HMC state
            if hmc.log && hmc.verbose
                update_log(hmc,model,fa)
            end
        end
    end

    # calcualte acceptance probability
    P = 0.0
    if iszero(flag)
        
        # calculate O⁻¹⋅ϕ₊ and O⁻¹⋅ϕ₋
        itrs, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)
        iters += itrs

        if iszero(flag)

            # calculate final energy
            H₁, S, K = calc_H(hmc, model, fa)

            # calculate change in energy
            ΔH = H₁ - H₀

            # calculate probability of acceptance
            P = min(1.0, exp(-ΔH))
        end
    end

    # Metropolis-Hasting Accept/Reject Step
    if rand(model.rng) < P && iszero(flag) # if accepted

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

    # keep track for linear solve flag
    flag = 0

    # keep track of iterations
    iters = 0

    # update exp{-Δτ⋅V[x]}
    update_model!(model)

    # refresh the velocity v
    refresh_v!(hmc,model,fa)

    # record intial phonon configuration
    copyto!(x0,x)
    copyto!(v0,v)

    # refresh ϕ
    refresh_ϕ!(hmc,model)

    # calculate O⁻¹⋅ϕ₊ and O⁻¹⋅ϕ₋
    itrs, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)
    iters     += iters

    if iszero(flag)

        # calculate energy
        H₀, S, K = calc_H(hmc, model, fa)

        # calculate the initial dSf/dx value
        fill!(dSfdx,0.0)
        calc_dSfdx!(hmc, model)

        # dSf/dx(t+Δt) ==> Q⋅dSf/dx(t+Δt)
        fourier_accelerate!(QdSfdx,fa,dSfdx,-1.0,use_mass=true)

        # log HMC state
        if hmc.log && hmc.verbose
            update_log(hmc,model,fa)
        end

        # iterate over timesteps
        for hmc.t in 1:Nt

            # v(t+Δt/2) = v(t) - Δt/2⋅Q⋅dSf/dx(t)
            @. v = v - Δt/2*QdSfdx

            # calculate the initial dSb/dx value
            fill!(dSbdx,0.0)
            calc_dSbdx!(dSbdx,model)

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
                calc_dSbdx!(dSbdx,model)

                # dS/dx(t+Δt) ==> Q⋅dS/dx(t+Δt)
                fourier_accelerate!(QdSbdx,fa,dSbdx,-1.0,use_mass=true)

                # v(t+Δt) = v(t+Δt/2) - Δt/2⋅Q⋅dSb/dx(t+Δt)
                @. v = v - Δt′/2*QdSbdx
            end

            # update exp{-Δτ⋅V[x]}
            update_model!(model)

            # calculate O⁻¹⋅ϕ₊ and O⁻¹⋅ϕ₋
            itrs, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,1.0)
            iters += itrs

            # kill trajectory if error occurred
            if flag > 0
                break
            end

            # calculate dSf/dx(t+Δt) value
            fill!(dSfdx,0.0)
            calc_dSfdx!(hmc, model)

            # dSf/dx(t+Δt) ==> Q⋅dSf/dx(t+Δt)
            fourier_accelerate!(QdSfdx,fa,dSfdx,-1.0,use_mass=true)

            # v(t+Δt) = v(t) - Δt/2⋅Q⋅dSf/dx(t)
            @. v = v - Δt/2*QdSfdx

            # log HMC state
            if hmc.log && hmc.verbose
                update_log(hmc,model,fa)
            end
        end
    end

    # calcualte acceptance probability
    P = 0.0
    if iszero(flag)

        # calculate O⁻¹⋅ϕ₊ and O⁻¹⋅ϕ₋
        itrs, flag = calc_O⁻¹Λϕ!(hmc,model,preconditioner,2.0)
        iters += itrs

        if iszero(flag)

            # calculate final energy
            H₁, S, K = calc_H(hmc,model,fa)

            # calculate change in energy
            ΔH = H₁ - H₀

            # calculate probability of acceptance
            P = min(1.0, exp(-ΔH))
        end
    end

    # Metropolis-Hasting Accept/Reject Step
    if rand(model.rng) < P && iszero(flag) # if accepted

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
The kinetics energy in our dyanics is given by `K=v⋅Q⁻¹⋅v/2=v⋅M⋅v/2`, so therefore it is possible to
refresh the velocity `v` according `v=√(Q)⋅R` where `R` is a vector of normal random numbers
and `Q` is the fourier acceleration matrix and `M` is the dynamical mass matrix.
More specifically, this function supports partial momentum refreshes of the
form `v = α⋅v + √(1-α²)⋅v′` where `v′=√(Q)⋅R=√(M⁻¹)⋅R`⋅
"""
function refresh_v!(hmc::HybridMonteCarlo{T},model::AbstractModel{T},fa::FourierAccelerator{T}) where {T<:AbstractFloat}

    R       = hmc.y
    sqrtQR  = hmc.y
    v       = hmc.v
    α       = hmc.α

    randn!(R,model)
    fourier_accelerate!(sqrtQR,fa,R,-0.5,use_mass=true)
    @. v = α*v + sqrt(1.0-α^2)*sqrtQR

    return nothing
end


"""
Refresh `ϕ` according to the relationship `ϕ ~ Λ⁻¹⋅Mᵀ⋅R` where `R` is a vector of normal random numbers.
"""
function refresh_ϕ!(hmc::HybridMonteCarlo{T1},model::AbstractModel{T1,T2};sample_R::Bool=true) where {T1,T2}

    @unpack R₊, R₋, ϕ₊, ϕ₋, Λϕ₊, Λϕ₋, Λ = hmc

    # update Λ
    update_Λ!(hmc,model)

    # sample new random vectors
    if sample_R
        randn!(model.rng,R₊)
        randn!(model.rng,R₋)
    end

    # ϕ₊ = Λ⁻¹⋅Mᵀ⋅R₊
    mulMᵀ!(Λϕ₊,model,R₊)
    mulΛ⁻¹!(ϕ₊,Λϕ₊,hmc,model)

    # ϕ₋ = Λ⁻¹⋅Mᵀ⋅R₋
    mulMᵀ!(Λϕ₋,model,R₋)
    mulΛ⁻¹!(ϕ₋,Λϕ₋,hmc,model)

    return nothing
end


"""
Calculate the total energy `H = K + S`.
"""
function calc_H(hmc::HybridMonteCarlo{T1}, model::AbstractModel{T1,T2}, fa::FourierAccelerator{T1}) where {T1,T2}
    
    S = calc_S(hmc, model)
    K = calc_K(hmc, model, fa)
    hmc.H = S + K

    return hmc.H, S, K
end


"""
Calculate the kintetic energy `K = v⋅Q⁻¹⋅v/2` where `Q` is the acceleration matrix.
"""
function calc_K(hmc::HybridMonteCarlo{T1}, model::HolsteinModel{T1,T2,T3}, fa::FourierAccelerator{T1})::T1 where {T1,T2,T3}

    v    = hmc.v
    Q⁻¹v = hmc.y
    fourier_accelerate!(Q⁻¹v,fa,v,1.0,use_mass=true)
    hmc.K = dot(v,Q⁻¹v)/2

    return hmc.K
end

function calc_K(hmc::HybridMonteCarlo{T1}, model::SSHModel{T1,T2,T3}, fa::FourierAccelerator{T1})::T1 where {T1,T2,T3}

    v  = hmc.v
    mv = hmc.y
    fourier_accelerate!(mv,fa,v,1.0,use_mass=true)
    
    hmc.K = 0.0

    # iterate over fields
    for field in 1:hmc.Ndof
        # if a primary field
        if model.primary_field[field]==field
            # increment total kinetic energy accordingly
            hmc.K += v[field]*mv[field]/2
        end
    end

    return hmc.K
end


"""
Calcualte the action S = Sb + ϕ₊ᵀ⋅O⁻¹⋅ϕ₊/2 + ϕ₋ᵀ⋅O⁻¹⋅ϕ/2
"""
function calc_S(hmc::HybridMonteCarlo{T1}, model::AbstractModel{T1,T2})::T1 where {T1,T2}
    
    # S = ϕ₊ᵀ⋅O⁻¹⋅ϕ₊/2 + ϕ₋ᵀ⋅O⁻¹⋅ϕ₋/2
    hmc.S  = calc_Sf(hmc,model)

    # S = Sb + ϕ₊ᵀ⋅O⁻¹⋅ϕ₊/2 + ϕ₋ᵀ⋅O⁻¹⋅ϕ₋/2
    hmc.S += calc_Sb(model)

    return hmc.S
end


"""
Calculate the derivative of the action dS/dx = dSb/dx - ϕ₊ᵀ⋅O⁻ᵀ⋅[Mᵀ⋅dM/dx]⋅O⁻¹⋅ϕ₊ - ϕ₋ᵀ⋅O⁻ᵀ⋅[Mᵀ⋅dM/dx]⋅O⁻¹⋅ϕ₋
"""
function calc_dSdx!(hmc::HybridMonteCarlo{T1}, model::AbstractModel{T1,T2}) where {T1,T2}
    
    # dS/dx = -ϕ₊ᵀ⋅O⁻ᵀ⋅[dMᵀ/dx⋅M]⋅O⁻¹⋅ϕ₊ - ϕ₋ᵀ⋅O⁻ᵀ⋅[dMᵀ/dx⋅M]⋅O⁻¹⋅ϕ₋
    calc_dSfdx!(hmc, model)

    # dS/dx = dSb/dx - ϕ₊ᵀ⋅O⁻ᵀ⋅[dMᵀ/dx⋅M]⋅O⁻¹⋅ϕ₊ - ϕ₋ᵀ⋅O⁻ᵀ⋅[dMᵀ/dx⋅M]⋅O⁻¹⋅ϕ₋
    calc_dSbdx!(hmc.dSdx, model)

    return nothing
end


"""
Calculate the fermionic action S = [ϕ₊ᵀ⋅Λ⋅O⁻¹⋅Λ⋅ϕ₊ + ϕ₋ᵀ⋅Λ⋅O⁻¹⋅Λ⋅ϕ₋]/2 = [ϕ₊ᵀ⋅Λ⋅[Mᵀ⋅M]⁻¹⋅Λ⋅ϕ₊ + ϕ₋ᵀ⋅Λ⋅[Mᵀ⋅M]⁻¹⋅Λ⋅ϕ₋]/2
"""
function calc_Sf(hmc::HybridMonteCarlo{T},model::AbstractModel{T})::T where {T<:AbstractFloat}

    @unpack Λϕ₊, Λϕ₋, O⁻¹Λϕ₊, O⁻¹Λϕ₋ = hmc

    Sf  = dot(Λϕ₊,O⁻¹Λϕ₊)/2
    Sf += dot(Λϕ₋,O⁻¹Λϕ₋)/2

    return Sf
end


"""
Calculate the derivative of the fermionic action.
Each partial derivative `∂S/∂xᵢ(τ)` will be stored to the corresponding element in the array dSdx.
"""
function calc_dSfdx!(hmc::HybridMonteCarlo{T1}, model::AbstractModel{T1,T2}) where {T1,T2}
    
    @unpack O⁻¹Λϕ₊, O⁻¹Λϕ₋, Λϕ₊, Λϕ₋, ϕ₊, ϕ₋ = hmc
    dSfdx   = hmc.dSdx
    dMdx    = hmc.y
    MO⁻¹Λϕ  = hmc.u

    # dSf/dx += -[M⋅O⁻¹⋅Λ⋅ϕ₊]ᵀ⋅dM/dx⋅[O⁻¹⋅Λ⋅ϕ₊]
    mulM!(MO⁻¹Λϕ,model,O⁻¹Λϕ₊)
    muldMdx!(dMdx,MO⁻¹Λϕ,model,O⁻¹Λϕ₊)
    @. dSfdx += -dMdx

    # dSf/dx += -[M⋅O⁻¹⋅Λ⋅ϕ₋]ᵀ⋅dM/dx⋅[O⁻¹⋅Λ⋅ϕ₋]
    mulM!(MO⁻¹Λϕ,model,O⁻¹Λϕ₋)
    muldMdx!(dMdx,MO⁻¹Λϕ,model,O⁻¹Λϕ₋)
    @. dSfdx += -dMdx

    # dSf/dx += [ϕ₊]ᵀ⋅dΛ/dx⋅[O⁻¹⋅Λ⋅ϕ₊]
    muldΛdx!(dSfdx,ϕ₊,O⁻¹Λϕ₊,hmc,model)

    # dSf/dx += [ϕ₋]ᵀ⋅dΛ/dx⋅[O⁻¹⋅Λ⋅ϕ₋]
    muldΛdx!(dSfdx,ϕ₋,O⁻¹Λϕ₋,hmc,model)

    return nothing
end


"""
Solve `O⋅x=Λ⋅ϕ₊ ==> x=O⁻¹⋅Λ⋅ϕ₊` and `O⋅x=Λ⋅ϕ₋ ==> x=O⁻¹⋅Λ⋅ϕ₋` where `O = Mᵀ⋅M`.
"""
function calc_O⁻¹Λϕ!(hmc::HybridMonteCarlo{T1}, model::AbstractModel{T1,T2}, preconditioner=I, power::T1=1.0)::Tuple{Int,Int} where {T1,T2}

    @unpack Λ, ϕ₊, ϕ₋, Λϕ₊, Λϕ₋, O⁻¹Λϕ₊, O⁻¹Λϕ₋ = hmc
    M⁻ᵀΛϕ₊ = hmc.u
    M⁻ᵀΛϕ₋ = hmc.u

    # udpate tolerance used by solver
    tol = model.solver.tol::T1
    model.solver.tol = tol^power

    # count iterative solver iterations
    hmc.iters = 0

    # setup the precontioer
    KPMPreconditioners.setup!(preconditioner)

    #############################
    ## CALCULATE Λ⋅ϕ₊ AND Λ⋅ϕ₋ ##
    #############################

    update_Λ!(hmc,model)
    mulΛ!(Λϕ₊,ϕ₊,hmc,model)
    mulΛ!(Λϕ₋,ϕ₋,hmc,model)

    #########################
    ## CALCULATE  O⁻¹⋅Λ⋅ϕ₊ ##
    #########################

    # linear solve status flag
    flag = 0

    if !model.mul_by_M # if using Conjugate Gradient

        # solve linear system
        fill!(O⁻¹Λϕ₊,0.0)
        model.transposed = false
        iters, err, flag = ldiv!(O⁻¹Λϕ₊,model,Λϕ₊,preconditioner)
        hmc.iters += iters

    else # if using GMRES

        # solve linear system
        fill!(M⁻ᵀΛϕ₊,0.0)
        model.transposed  = true
        iters, err, flag1 = ldiv!(M⁻ᵀΛϕ₊,model,Λϕ₊,preconditioner)
        hmc.iters        += iters
        flag              = max(flag,flag1)

        # solve linear system
        fill!(O⁻¹Λϕ₊,0.0)
        model.transposed  = false
        iters, err, flag2 = ldiv!(O⁻¹Λϕ₊,model,M⁻ᵀΛϕ₊,preconditioner)
        hmc.iters        += iters
        flag              = max(flag,flag2)
    end

    #########################
    ## CALCULATE  O⁻¹⋅Λ⋅ϕ₋ ##
    #########################

    if !model.mul_by_M && iszero(flag)

        # solve linear system
        fill!(O⁻¹Λϕ₋,0.0)
        model.transposed =false
        iters, err, flag = ldiv!(O⁻¹Λϕ₋,model,Λϕ₋,preconditioner)
        hmc.iters += iters
        
    elseif iszero(flag)

        # solve linear system
        fill!(M⁻ᵀΛϕ₋,0.0)
        model.transposed  = true
        iters, err, flag1 = ldiv!(M⁻ᵀΛϕ₋,model,Λϕ₋,preconditioner)
        hmc.iters        += iters
        flag              = max(flag,flag1)

        # solve linear system
        fill!(O⁻¹Λϕ₋,0.0)
        model.transposed  = false
        iters, err, flag2 = ldiv!(O⁻¹Λϕ₋,model,M⁻ᵀΛϕ₋,preconditioner)
        hmc.iters        += iters
        flag              = max(flag,flag2)
    end

    # accounting for the fact that there is a spin up and spin down
    # linear system that needs to be solved
    if iszero(flag)
        hmc.iters = cld(hmc.iters,2)
    end

    # revert solvers tolerance to original value
    model.solver.tol = tol

    return hmc.iters, flag
end


"""
Construct the Λ matrix.
"""
function update_Λ!(hmc::HybridMonteCarlo{T1}, model::HolsteinModel{T1,T2}) where {T1,T2}

    @unpack λ, Δτ, Lτ, Nph = model
    x = reshaped(model.x, Lτ, Nph)
    Λ = reshaped(hmc.Λ,   Lτ, Nph)

    @fastmath @inbounds for i in 1:Nph

        xᵢ = @view x[:,i]
        Λᵢ = @view Λ[:,i]
        λᵢ = λ[i]

        for τ in 1:Lτ
            Λᵢ[τ] = exp(-Δτ*λᵢ*xᵢ[τ]/2)
        end
    end

    return nothing
end

function update_Λ!(hmc::HybridMonteCarlo{T1}, model::SSHModel{T1,T2}) where {T1,T2}

    return nothing
end

"""
Calculate v′=Λ⋅v.
"""
function mulΛ!(v′::AbstractVector{T}, v::AbstractVector{T}, hmc::HybridMonteCarlo{T}, model::HolsteinModel{T}) where {T}

    @unpack x, λ, Δτ, Lτ, Nph, Nsites = model

    u  = reshaped(v ,Lτ,Nsites)
    u′ = reshaped(v′,Lτ,Nsites)
    Λ  = reshaped(hmc.Λ,Lτ,Nsites)

    @fastmath @inbounds for i in 1:Nsites
        u1i = u[1,i]
        for τ in 1:Lτ-1
            u′[τ,i] = -Λ[τ+1,i] * u[τ+1,i]
        end
        u′[Lτ,i] = Λ[1,i] * u1i
    end

    return nothing
end

function mulΛ!(v′::AbstractVector{T}, v::AbstractVector{T}, hmc::HybridMonteCarlo{T}, model::AbstractModel{T}) where {T}

    return nothing
end

"""
Calculate v′=Λ⋅v.
"""
function mulΛ⁻¹!(v′::AbstractVector{T}, v::AbstractVector{T}, hmc::HybridMonteCarlo{T}, model::HolsteinModel{T}) where {T}

    @unpack x, λ, Δτ, Lτ, Nph, Nsites = model

    u  = reshaped(v ,Lτ,Nsites)
    u′ = reshaped(v′,Lτ,Nsites)
    Λ  = reshaped(hmc.Λ,Lτ,Nsites)

    @fastmath @inbounds for i in 1:Nsites
        uLi = u[Lτ,i]
        for τ in 2:Lτ
            u′[τ,i] = -inv(Λ[τ,i]) * u[τ-1,i]
        end
        u′[1,i] = inv(Λ[1,i]) * uLi
    end

    return nothing
end

function mulΛ⁻¹!(v′::AbstractVector{T}, v::AbstractVector{T}, hmc::HybridMonteCarlo{T}, model::AbstractModel{T}) where {T}

    return nothing
end

"""
Calculate ⟨vₗ|∂Λ/∂x(τ)|vᵣ⟩ for all τ, adding each result to the corresponding element in the vector dΛdx.
"""
function muldΛdx!(dΛdx::Vector{T1},vₗ::Vector{T1},vᵣ::Vector{T1},hmc::HybridMonteCarlo{T1},model::HolsteinModel{T1,T2}) where {T1,T2}

    @unpack x, λ, Δτ, Lτ, Nph = model
    @unpack Λ = hmc

    @fastmath @inbounds for i in 1:Nph
        λᵢ       = λ[i]
        n′       = get_index(Lτ,i,Lτ)
        n        = get_index(1,i,Lτ)
        dΛdx[n] += vₗ[n] * (-Δτ*λᵢ/2)*Λ[n] * vᵣ[n′]
        for τ in 2:Lτ
            τm1      = mod1(τ-1,Lτ)
            n′       = get_index(τm1,i,Lτ)
            n        = get_index(τ,i,Lτ)
            dΛdx[n] += vₗ[n] * (Δτ*λᵢ/2)*Λ[n] * vᵣ[n′]
        end
    end

    return nothing
end

function muldΛdx!(dΛdx::Vector{T1},vₗ::Vector{T1},vᵣ::Vector{T1},hmc::HybridMonteCarlo{T1}, model::AbstractModel{T1,T2}) where {T1,T2}

    return nothing
end

end