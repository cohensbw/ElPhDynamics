module IterativeSolvers

using LinearAlgebra
using UnsafeArrays
using Parameters

import LinearAlgebra: ldiv!

export Continuous, IterativeSolver, ConjugateGradient, BiCGStab, GMRES, solve!

"""
Allows the identity operator `I` to be used as a preconditioner.
"""
function ldiv!(vout::AbstractVector,I::UniformScaling,vin::AbstractVector)
    copyto!(vout,vin)
    return nothing
end

"""
Abstract type to represent continuous real or complex numbers.
"""
Continuous = Union{AbstractFloat,Complex{<:AbstractFloat}}

"""
Abstract type to represent Iterative Solver methods.
"""
abstract type IterativeSolver{Ttol<:AbstractFloat,Tdata<:Continuous} end

#######################################
## Conjugate Gradient Implementation ##
#######################################

"""
To represent Conjugate Gradient algorithm.
"""
mutable struct ConjugateGradient{Ttol,Tdata} <: IterativeSolver{Ttol,Tdata}
    
    tol::Ttol
    maxiter::Int
    N::Int
    r::Vector{Tdata}
    p::Vector{Tdata}
    z::Vector{Tdata}
    
    function ConjugateGradient(z::AbstractVector{Tdata};tol::Ttol=1e-4,maxiter::Int=0) where {Ttol<:AbstractFloat,Tdata<:Continuous}
        
        N = length(z)
        if maxiter<1
            maxiter = N
        end
        r = zeros(Tdata,N)
        p = zeros(Tdata,N)
        z = zeros(Tdata,N)
        return new{Ttol,Tdata}(tol,maxiter,N,r,p,z)
    end
end


"""
Solve `A⋅x=b` using the Conjugate Gradient method with a split preconditioner
such that `[L⁻¹⋅A⋅L⁻ᵀ]⋅u=L⁻¹⋅b` where `u=L⁻ᵀ⋅x`.
"""
function solve!(x::AbstractVector{Tdata},A,b::AbstractVector{Tdata},cg::ConjugateGradient{Ttol,Tdata},L,Lt)::Int where {Ttol,Tdata}
    
    r = cg.r
    p = cg.p
    z = cg.z
    
    # r₀ = b - A⋅x₀
    mul!(r,A,x)
    axpby!(1.0,b,-1.0,r)
    
    # r₀ = L⁻¹⋅r₀ = L⁻¹⋅(b - A⋅x₀)
    ldiv!(L,r)
    
    # p₀ = L⁻ᵀ⋅r₀ = L⁻ᵀ⋅L⁻¹⋅(b - A⋅x₀)
    ldiv!(p,Lt,r)
    
    # |L⁻ᵀ⋅L⁻¹⋅b|
    ldiv!(z,L,b)
    ldiv!(Lt,z)
    normL⁻ᵀL⁻¹b = norm(z)

    # r₀⋅r₀
    rdotr = dot(r,r)
    
    @fastmath @inbounds for j in 1:cg.maxiter
        
        # αⱼ = (rⱼ⋅rⱼ)/(pⱼ⋅A⋅pⱼ)
        mul!(z,A,p)
        α = rdotr/dot(p,z)
        
        # xⱼ₊₁ = xⱼ + αⱼ⋅pⱼ
        axpy!(α,p,x)
        
        # rⱼ₊₁ = rⱼ - αⱼ⋅L⁻¹⋅A⋅pⱼ
        ldiv!(L,z)
        axpy!(-α,z,r)
        
        # βⱼ = (rⱼ₊₁⋅rⱼ₊₁)/(rⱼ⋅rⱼ)
        new_rdotr = dot(r,r)
        β     = new_rdotr/rdotr
        rdotr = new_rdotr
        
        # pⱼ₊₁ = L⁻ᵀ⋅rⱼ₊₁ + βⱼ⋅pⱼ
        ldiv!(z,Lt,r)
        axpby!(1.0,z,β,p)
        
        # δ = |pⱼ₊₁|/|L⁻ᵀ⋅L⋅b| = |L⁻ᵀ⋅L⋅(A⋅xⱼ₊₁-b)|/|L⁻ᵀ⋅L⋅b| 
        δ = norm(p)/normL⁻ᵀL⁻¹b
        
        # check stop criteria
        if δ<cg.tol
            return j
        end
    end
    
    return cg.maxiter
    
end

"""
Solve `A⋅x=b` using the Conjugate Gradient method with a left preconditioner
such that `P⁻¹⋅A⋅x=P⁻¹⋅b`.
"""
function solve!(x::AbstractVector{Tdata},A,b::AbstractVector{Tdata},cg::ConjugateGradient{Ttol,Tdata},P)::Int where {Ttol,Tdata}
    
    r = cg.r
    p = cg.p
    z = cg.z
    
    # |P⁻¹⋅b|
    ldiv!(z,P,b)
    normP⁻¹b = norm(z)
    
    # r₀ = b - A⋅x₀
    mul!(r,A,x)
    axpby!(1.0,b,-1.0,r)
    
    # z₀ = P⁻¹⋅r₀ = P⁻¹⋅(b - A⋅x₀)
    ldiv!(z,P,r)
    
    # p₀ = z₀
    copyto!(p,z)

    # r₀⋅z₀
    rdotz = dot(r,z)
    
    @fastmath @inbounds for j in 1:cg.maxiter
        
        # αⱼ = (rⱼ⋅zⱼ)/(pⱼ⋅A⋅pⱼ)
        mul!(z,A,p)
        α = rdotz/dot(p,z)
        
        # xⱼ₊₁ = xⱼ + αⱼ⋅pⱼ
        axpy!(α,p,x)
        
        # rⱼ₊₁ = rⱼ - αⱼ⋅A⋅pⱼ
        axpy!(-α,z,r)
        
        # zⱼ₊₁ = P⁻¹⋅rⱼ₊₁
        ldiv!(z,P,r)

        # δ = |zⱼ₊₁|/|P⁻¹⋅b| = |P⁻¹(b-A⋅xⱼ₊₁)|/|P⁻¹⋅b|
        δ = norm(z)/normP⁻¹b
        
        # check stop criteria
        if δ<cg.tol
            return j
        end
        
        # βⱼ = (rⱼ₊₁⋅zⱼ₊₁)/(rⱼ⋅zⱼ)
        new_rdotz = dot(r,z)
        β     = new_rdotz/rdotz
        rdotz = new_rdotz
        
        # pⱼ₊₁ = zⱼ₊₁ + βⱼ⋅pⱼ
        axpby!(1.0,z,β,p)
    end
    
    return cg.maxiter
end

"""
Solve `A⋅x=b` using the Conjugate Gradient method with no preconditioning.
"""
function solve!(x::AbstractVector{Tdata},A,b::AbstractVector{Tdata},cg::ConjugateGradient{Ttol,Tdata})::Int where {Ttol,Tdata}
    
    r = cg.r
    p = cg.p
    z = cg.z
    
    # |b|
    normb = norm(b)
    
    # r₀ = b - A⋅x₀
    mul!(r,A,x)
    axpby!(1.0,b,-1.0,r)
    
    # p₀ = r₀
    copyto!(p,r)

    # r₀⋅r₀
    rdotr = dot(r,r)
    
    @fastmath @inbounds for j in 1:cg.maxiter
        
        # αⱼ = (rⱼ⋅rⱼ)/(pⱼ⋅A⋅pⱼ)
        mul!(z,A,p)
        α = rdotr/dot(p,z)
        
        # xⱼ₊₁ = xⱼ + αⱼ⋅pⱼ
        axpy!(α,p,x)
        
        # rⱼ₊₁ = rⱼ - αⱼ⋅A⋅pⱼ
        axpy!(-α,z,r)

        # δ = |rⱼ₊₁|/|b| = |b-A⋅xⱼ₊₁|/|b|
        δ = norm(r)/normb
        
        # check stop criteria
        if δ<cg.tol
            return j
        end
        
        # βⱼ = (rⱼ₊₁⋅rⱼ₊₁)/(rⱼ⋅rⱼ)
        new_rdotr = dot(r,r)
        β     = new_rdotr/rdotr
        rdotr = new_rdotr
        
        # pⱼ₊₁ = rⱼ₊₁ + βⱼ⋅pⱼ
        axpby!(1.0,r,β,p)
    end
    
    return cg.maxiter
end

#############################
## BiCGStab Implementation ##
#############################

"""
To represent BiCGStab algorithm.
"""
mutable struct BiCGStab{Ttol,Tdata} <: IterativeSolver{Ttol,Tdata}

    tol::Ttol
    maxiter::Int
    r::Vector{Tdata}
    r̃::Vector{Tdata}
    p::Vector{Tdata}
    p̂::Vector{Tdata}
    s::Vector{Tdata}
    ŝ::Vector{Tdata}
    v::Vector{Tdata}
    t::Vector{Tdata}

    function BiCGStab(x::AbstractVector{Tdata};tol::Ttol=1e-4,maxiter::Int=0) where {Ttol<:AbstractFloat,Tdata<:Continuous}

        N = length(x)
        r = zeros(Tdata,N)
        r̃ = zeros(Tdata,N)
        p = zeros(Tdata,N)
        p̂ = zeros(Tdata,N)
        s = zeros(Tdata,N)
        ŝ = zeros(Tdata,N)
        v = zeros(Tdata,N)
        t = zeros(Tdata,N)
        return new{Ttol,Tdata}(tol,maxiter,r,r̃,p,p̂,s,ŝ,v,t)
    end
end

"""
Solve linear system using preconditioned BiCGStab.
"""
function solve!(x::AbstractVector{Tdata},A,b::AbstractVector{Tdata},bicgstab::BiCGStab{Ttol,Tdata},P=I)::Int where {Ttol,Tdata}

    @unpack tol, maxiter, r, r̃, p, p̂, s, ŝ, v, t = bicgstab

    # r = b - A⋅x
    mul!(r,A,x)
    @. r = b - r

    # r̃ = r
    copyto!(r̃,r)

    # |b|
    b̄ = norm(b)

    # intializing values
    β    = 0.0
    α    = 0.0
    ρᵢ₋₁ = 1.0
    ρᵢ₋₂ = 1.0
    ω    = 1.0

    # iterate to convergence
    @fastmath @inbounds for i in 1:maxiter
        ρᵢ₋₂ = ρᵢ₋₁
        ρᵢ₋₁ = dot(r̃,r)
        if ρᵢ₋₁==0.0
            break
        end
        β    = (ρᵢ₋₁/ρᵢ₋₂)*(α/ω)
        @. p = r + β*(p-ω*v)
        ldiv!(p̂,P,p)
        mul!(v,A,p̂)
        α    = ρᵢ₋₁/dot(r̃,v)
        @. s = r - α*v
        δ    = norm(s)/b̄
        if δ<tol
            @. x += α*p̂
            return i
        end
        ldiv!(ŝ,P,s)
        mul!(t,A,ŝ)
        ω     = dot(t,s)/dot(t,t)
        @. x += α*p̂ + ω*ŝ
        @. r  = s - ω*t
        δ     = norm(r)/b̄
        if δ<tol
            return i
        end
        if ω==0.0
            break
        end
    end

    return maxiter
end

##########################
## GMRES Implementation ##
##########################

"""
To represent GMRES algorithm.
"""

mutable struct GMRES{Ttol,Tdata} <: IterativeSolver{Ttol,Tdata}
    
    ndim::Int
    maxiter::Int
    restart::Int
    tol::Ttol
    H::Matrix{Tdata}
    V::Matrix{Tdata}
    r::Vector{Tdata}
    w::Vector{Tdata}
    s::Vector{Tdata}
    y::Vector{Tdata}
    sn::Vector{Tdata}
    cs::Vector{Tdata}
    
    function GMRES(x::AbstractVector{Tdata}; maxiter::Int=-1,restart::Int=-1,tol::Ttol=1e-4) where {Ttol<:AbstractFloat,Tdata<:Continuous}
        
        ndim = length(x)
        if maxiter<0
            maxiter = ndim
        end
        if restart<0
            restart = min(20,ndim)
        end
        H  = zeros(Tdata,restart+1,restart)
        V  = zeros(Tdata,ndim,restart+1)
        r  = zeros(Tdata,ndim)
        w  = zeros(Tdata,ndim)
        s  = zeros(Tdata,restart+1)
        y  = zeros(Tdata,restart+1)
        sn = zeros(Tdata,restart+1)
        cs = zeros(Tdata,restart+1)
        return new{Ttol,Tdata}(ndim,maxiter,restart,tol,H,V,r,w,s,y,sn,cs)
    end
end


function solve!(x::AbstractVector{Tdata},A,b::AbstractVector{Tdata},gmres::GMRES{Ttol,Tdata},M=I)::Int where {Ttol,Tdata}
    
    H  = gmres.H
    V  = gmres.V
    r  = gmres.r
    w  = gmres.w
    s  = gmres.s
    y  = gmres.y
    sn = gmres.sn
    cs = gmres.cs
    
    # calculating norm of b
    copyto!(r,b)
    ldiv!(M,r)
    normb = norm(r)
    if normb==0.0
        normb = 1.0
    end
    
    # calculating residual vector r and getting its norm
    mul!(r,A,x)  # r = A⋅x
    @. r = b - r # r = b - A⋅x
    ldiv!(M,r)   # r = M \ (b - A⋅x)
    β = norm(r)
    
    # iteration counter
    iter = 0
    
    # initialize error
    Δ = β/normb
    if Δ<gmres.tol
        return iter
    end
    
    @uviews V begin
        @fastmath @inbounds while iter < gmres.maxiter        
            @. V[:,1] = r/β
            fill!(s,0.0)
            s[1]  = β
            for i in 1:gmres.restart
                iter += 1
                vi    = @view V[:,i]
                mul!(w,A,vi)
                ldiv!(M,w)
                for k in 1:i
                    vk     = @view V[:,k]
                    H[k,i] = dot(vk,w)
                    @. w  -= H[k,i] * vk
                end
                H[i+1,i]    = norm(w)
                @. V[:,i+1] = w / H[i+1,i]
                for k in 1:i-1
                    H[k,i], H[k+1,i] = apply_plane_rotation(H[k,i], H[k+1,i], cs[k], sn[k])
                end
                cs[i], sn[i]     = generate_plane_rotation(H[i,i], H[i+1,i])
                H[i,i], H[i+1,i] = apply_plane_rotation(H[i,i], H[i+1,i], cs[i], sn[i])
                s[i], s[i+1]     = apply_plane_rotation(s[i], s[i+1], cs[i], sn[i])
                Δ                = abs(s[i+1])/normb
                if Δ < gmres.tol
                    update!(x,i,H,s,y,V)
                    return iter
                end
                if iter==gmres.maxiter
                    break
                end
            end
            update!(x,gmres.restart,H,s,y,V)
            mul!(r,A,x)  # r = A⋅x
            @. r = b - r # r = b - A⋅x
            ldiv!(M,r)   # r = M \ (b - A⋅x) = M⁻¹⋅(b - A⋅x)
            β    = norm(r)
            Δ    = β/normb
            if Δ<gmres.tol
                return iter
            end
        end
    end

    return iter
end

function update!(x::AbstractVector,k::Int,H::Matrix,s::AbstractVector,y::AbstractVector,V::AbstractMatrix)
    
    copyto!(y,s)
    
    @fastmath @inbounds for i in k:-1:1
        y[i] /= H[i,i]
        for j in i-1:-1:1
            y[j] -= H[j,i] * y[i]
        end
    end

    @fastmath @inbounds for j in 1:k
        for i in 1:length(x)
            x[i] += y[j] * V[i,j]
        end
    end

    return nothing
end

function apply_plane_rotation(dx::T,dy::T,cs::T,sn::T)::Tuple{T,T} where {T<:Continuous}
    
    dxp =  cs * dx + conj(sn) * dy
    dyp = -sn * dx +      cs  * dy
    return dxp, dyp
end

function generate_plane_rotation(dx::T,dy::T)::Tuple{T,T} where {T<:Real}
    
    dz = dx + im*dy
    θ  = angle(dz)
    return cos(θ), sin(θ)
end

function generate_plane_rotation(dx::T,dy::T)::Tuple{T,T} where {T<:Complex}
    
    c = 0.0
    s = 1.0
    if dx != 0.0
        c = abs(dx)/sqrt(abs2(dx)+abs2(dy))
        s = (dy/dx)*c
    end
    return c, s
end

end