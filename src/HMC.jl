module HMC

using Random
using UnsafeArrays
using LinearAlgebra
using Printf

using ..Utilities: get_index
using ..HolsteinModels: HolsteinModel, construct_expnΔτV!, mulM!, muldMdx!, mulMᵀ!
using ..PhononAction: calc_dSbosedx!, calc_Sbose
using ..FourierAcceleration: FourierAccelerator, forward_fft!, inverse_fft!, accelerate!
import ..KPMPreconditioners

export HybridMonteCarlo, update!

mutable struct HybridMonteCarlo{T<:AbstractFloat}

    """
    Number of degrees of freedom to update.
    """
    Ndof::Int

    """
    The phonon fields.
    """
    x::Vector{T}

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
    Temporary storage vector.
    """
    u::Vector{T}

    """
    Temporary storage vector.
    """
    z::Vector{Complex{T}}

    function HybridMonteCarlo(Ndof::Int,Δt::T,tr::T=1.0) where {T<:AbstractFloat}

        @assert 0.0<tr<=1.0

        x      = zeros(T,Ndof)
        dSdx   = zeros(T,Ndof)
        v      = randn(T,Ndof)
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
        z      = zeros(Complex{T},Ndof)

        # partial momentum refresh parameter
        α = 1.0 - tr

        # the action
        H = 0.0::T

        # number of timesteps
        Nt = round(Int,tr/Δt)

        return new{T}(Ndof, x, tr, Δt, Nt, α, H, dSdx, v, R, ϕ₊, ϕ₋, M⁻ᵀϕ₊, M⁻ᵀϕ₊′, M⁻ᵀϕ₋, M⁻ᵀϕ₋′, O⁻¹ϕ₊, O⁻¹ϕ₊′, O⁻¹ϕ₋, O⁻¹ϕ₋′, u, z)
    end
end


"""
Do a Hybrid Monte Carlo update to the phonon fields.
"""
function update!(holstein::HolsteinModel{T1,T2}, hmc::HybridMonteCarlo{T1}, fa::FourierAccelerator{T1}, preconditioner=I)::Tuple{Bool,T1}  where {T1<:AbstractFloat,T2<:Number}

    x     = holstein.x
    x0    = hmc.x
    dSdx  = hmc.dSdx
    QdSdx = hmc.dSdx
    dSdx′ = hmc.z
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
    calc_dSdx!(hmc, holstein, preconditioner)

    # fourier accelerate dS/dx ==> Q⋅dS/dx
    forward_fft!(dSdx′,dSdx,fa)
    accelerate!(dSdx′,fa,1.0)
    inverse_fft!(QdSdx,dSdx′,fa)

    # calculate the total energy H
    H0 = calc_H(hmc, holstein, fa)

    # record intial phonon configuration
    copyto!(x0,x)

    # iterate over time steps, doing leapfrog updates to the phonon fields
    iters = 0
    for t in 1:Nt

        # v(t+Δt/2) = v(t) - Δt/2⋅Q⋅dS/dx(t)
        @. v = v - Δt/2*QdSdx

        # x(t+Δt) = x(t) + Δt⋅v(t+Δt/2)
        @. x = x + Δt*v

        # update exp{-Δτ⋅V[x]}
        construct_expnΔτV!(holstein)

        # calculate dS/dx(t+Δt) value
        iter_t = calc_dSdx!(hmc, holstein, preconditioner)
        iters += iter_t
        # println(iter_t)

        # fourier accelerate dS/dx(t+Δt)
        forward_fft!(dSdx′,dSdx,fa)
        accelerate!(dSdx′,fa,1.0)
        inverse_fft!(QdSdx,dSdx′,fa)

        # v(t+Δt) = v(t+⋅Δt/2) - Δt/2⋅Q⋅dS/dx(t+Δt)
        @. v = v - Δt/2*QdSdx
    end

    # calculate the final energy
    H = calc_H(hmc, holstein, fa)

    # calculate probability of acceptance
    P = min(1.0, exp(H0-H))

    # get the number of iterations
    iters = cld(iters,Nt)

    # raising warning if dynamics has stalled.
    if iters==hmc.Ndof || iters==2*hmc.Ndof
        @warn @sprintf("HMC Update Failed to Converge Dynamics; Iterations Per Solve = %d",iters)
    end

    # Metropolis-Hasting Accept/Reject Step
    if rand() < P # if accepted

        # println("ACCEPTED")
        return true, T1(iters)

    else # if rejected

        # reset to original phonon field
        copyto!(x,x0)

        # update exp{-Δτ⋅V[x]}
        construct_expnΔτV!(holstein)

        # reflect velocity
        @. v = -v

        # println("REJECTED")
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

    sqrtQR  = hmc.R
    v  = hmc.v
    v′ = hmc.z
    α  = hmc.α

    randn!(sqrtQR)
    forward_fft!(v′,sqrtQR,fa)
    accelerate!(v′,fa,0.5)
    inverse_fft!(sqrtQR,v′,fa)

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
function calc_H(hmc::HybridMonteCarlo{T1}, holstein::HolsteinModel{T1,T2}, fa::FourierAccelerator{T1})::T1 where {T1<:AbstractFloat,T2<:Number}
    
    H  = calc_S(hmc, holstein)
    H += calc_K(hmc,fa)

    return H
end


"""
Calculate the kintetic energy `K = v⋅Q⁻¹⋅v/2` where `Q` is the acceleration matrix.
"""
function calc_K(hmc::HybridMonteCarlo{T}, fa::FourierAccelerator{T})::T where {T<:AbstractFloat}

    v    = hmc.v
    v′   = hmc.z
    Q⁻¹v = hmc.u

    forward_fft!(v′,v,fa)
    accelerate!(v′,fa,-1.0)
    inverse_fft!(Q⁻¹v,v′,fa)
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

    # In the lines of code below there is a subtle detail that is addressed that is a result of defining
    # our M matrix with the -B[τ] matrices on the upper diagonal instead of the lower diagonal.
    # After doing matrix-vector multiplies by all the  dM/dx's, the expectation value for the partial
    # derivatives corresponding to the τ time slice lives in the array indices corresponding to τ-1.
    # Therefore, the values need to be shifted one time slice forward. This is done by first calculating
    # and storing the ∂Sf/∂xᵢ(τ) partial derivative values in the vector dSdx′, and then copying a properly
    # shifted version into the vector dSdx.

    # iterate over sites
    @inbounds @fastmath for site in 1:holstein.nsites
        # iterate over time slices
        for τ in 1:holstein.Lτ
            idx_τ   = get_index(τ,               site, holstein.Lτ)
            idx_τp1 = get_index(τ%holstein.Lτ+1, site, holstein.Lτ)
            # shifting values one time slice forward
            dSdx[idx_τp1] = real(dSdx′[idx_τ])
        end
    end

    return iters
end

"""
Solve `O⋅x=ϕ₊ ==> x=O⁻¹⋅ϕ₊` and `O⋅x=ϕ₋ ==> x=O⁻¹⋅ϕ₋` where `O = Mᵀ⋅M`.
"""
function calc_O⁻¹ϕ!(hmc::HybridMonteCarlo{T1}, holstein::HolsteinModel{T1,T2}, preconditioner=I)::Int where {T1<:AbstractFloat,T2<:Number}

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

    u      = hmc.u

    #######################
    ## CALCULATE  O⁻¹⋅ϕ₊ ##
    #######################

    # count iterative solver iterations
    iters = 0

    if !holstein.mul_by_M # if using Conjugate Gradient

        # construct initial guess for solution to linear system
        copyto!(u,O⁻¹ϕ₊)
        @. O⁻¹ϕ₊ = 2*O⁻¹ϕ₊ - O⁻¹ϕ₊′
        copyto!(O⁻¹ϕ₊′,u)

        # fill!(O⁻¹ϕ₊,0.0)

        # solve linear system
        iters += ldiv!(O⁻¹ϕ₊,holstein,ϕ₊,preconditioner)

    else # if using GMRES

        # setup the precontioer
        KPMPreconditioners.setup!(preconditioner)

        # construct initial guess for solution to linear system
        copyto!(u,M⁻ᵀϕ₊)
        @. M⁻ᵀϕ₊ = 2*M⁻ᵀϕ₊ - M⁻ᵀϕ₊′
        copyto!(M⁻ᵀϕ₊′,u)

        # fill!(M⁻ᵀϕ₊,0.0)

        # solve linear system
        holstein.transposed=true
        iters += ldiv!(M⁻ᵀϕ₊,holstein,ϕ₊,preconditioner)

        # construct initial guess for solution to linear system
        copyto!(u,O⁻¹ϕ₊)
        @. O⁻¹ϕ₊ = 2*O⁻¹ϕ₊ - O⁻¹ϕ₊′
        copyto!(O⁻¹ϕ₊′,u)

        # fill!(O⁻¹ϕ₊,0.0)

        # solve linear system
        holstein.transposed=false
        iters += ldiv!(O⁻¹ϕ₊,holstein,M⁻ᵀϕ₊,preconditioner)
    end

    #######################
    ## CALCULATE  O⁻¹⋅ϕ₋ ##
    #######################

    if !holstein.mul_by_M # if using Conjugate Gradient

        # construct initial guess for solution to linear system
        copyto!(u,O⁻¹ϕ₋)
        @. O⁻¹ϕ₋ = 2*O⁻¹ϕ₋ - O⁻¹ϕ₋′
        copyto!(O⁻¹ϕ₋′,u)

        # fill!(O⁻¹ϕ₋,0.0)

        # solve linear system
        iters += ldiv!(O⁻¹ϕ₋,holstein,ϕ₋,preconditioner)
        
    else # if using GMRES

        # construct initial guess for solution to linear system
        copyto!(u,M⁻ᵀϕ₋)
        @. M⁻ᵀϕ₋ = 2*M⁻ᵀϕ₋ - M⁻ᵀϕ₋′
        copyto!(M⁻ᵀϕ₋′,u)

        # fill!(M⁻ᵀϕ₋,0.0)

        # solve linear system
        holstein.transposed=true
        iters += ldiv!(M⁻ᵀϕ₋,holstein,ϕ₋,preconditioner)

        # construct initial guess for solution to linear system
        copyto!(u,O⁻¹ϕ₋)
        @. O⁻¹ϕ₋ = 2*O⁻¹ϕ₋ - O⁻¹ϕ₋′
        copyto!(O⁻¹ϕ₋′,u)

        # fill!(O⁻¹ϕ₋,0.0)

        # solve linear system
        holstein.transposed=false
        iters += ldiv!(O⁻¹ϕ₋,holstein,M⁻ᵀϕ₋,preconditioner)
    end

    # accounting for the fact that there is a spin up and spin down
    # linear system that needs to be solved
    iters = cld(iters,2)

    return iters
end

end