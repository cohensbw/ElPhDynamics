module ConjugateGradients

using LinearAlgebra
import LinearAlgebra: ldiv!

export ConjugateGradient, solve!

mutable struct ConjugateGradient{T1<:Number,T2<:AbstractFloat}
    
    tol::T2
    maxiter::Int
    N::Int
    r::Vector{T1}
    p::Vector{T1}
    z::Vector{T1}
    
    function ConjugateGradient(z::AbstractVector{T1};tol::T2=1e-4,maxiter::Int=0) where {T1<:Number,T2<:AbstractFloat}
        
        N = length(z)
        if maxiter<1
            maxiter = N
        end
        r = zeros(T1,N)
        p = zeros(T1,N)
        z = zeros(T1,N)
        return new{T1,T2}(tol,maxiter,N,r,p,z)
    end
end


"""
Solve `A⋅x=b` using the Conjugate Gradient method with a split preconditioner
such that `[L⁻¹⋅A⋅L⁻ᵀ]⋅u=L⁻¹⋅b` where `u=L⁻ᵀ⋅x`.
"""
function solve!(x::AbstractVector{T1},A,b::AbstractVector{T1},cg::ConjugateGradient{T1,T2},L,Lt)::Int where {T1<:Number,T2<:AbstractFloat}
    
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
function solve!(x::AbstractVector{T1},A,b::AbstractVector{T1},cg::ConjugateGradient{T1,T2},P)::Int where {T1<:Number,T2<:AbstractFloat}
    
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
function solve!(x::AbstractVector{T1},A,b::AbstractVector{T1},cg::ConjugateGradient{T1,T2})::Int where {T1<:Number,T2<:AbstractFloat}
    
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


# This function allows `I` to be used as a preconditioner.
function ldiv!(vout::AbstractVector,I::UniformScaling,vin::AbstractVector)
    copyto!(vout,vin)
    return nothing
end

end