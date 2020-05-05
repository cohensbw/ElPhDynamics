module RestartedGMRES

using LinearAlgebra
using UnsafeArrays

export GMRES, solve!

mutable struct GMRES{T1<:AbstractFloat,T2<:Number}
    
    ndim::Int
    maxiter::Int
    restart::Int
    tol::T1
    H::Matrix{T2}
    V::Matrix{T2}
    r::Vector{T2}
    w::Vector{T2}
    s::Vector{T2}
    y::Vector{T2}
    sn::Vector{T2}
    cs::Vector{T2}
    
    function GMRES(x::AbstractVector{T2}; maxiter::Int=-1,restart::Int=-1,tol::T1=1e-4) where {T1<:AbstractFloat,T2<:Number}
        
        ndim = length(x)
        if maxiter<0
            maxiter = ndim
        end
        if restart<0
            restart = min(20,ndim)
        end
        H  = zeros(T2,restart+1,restart)
        V  = zeros(T2,ndim,restart+1)
        r  = zeros(T2,ndim)
        w  = zeros(T2,ndim)
        s  = zeros(T2,restart+1)
        y  = zeros(T2,restart+1)
        sn = zeros(T2,restart+1)
        cs = zeros(T2,restart+1)
        return new{T1,T2}(ndim,maxiter,restart,tol,H,V,r,w,s,y,sn,cs)
    end
end


function solve!(x::AbstractVector{T2},A,b::AbstractVector{T2},gmres::GMRES{T1,T2},M=I)::Tuple{Int,Int,T1} where {T1<:AbstractFloat,T2<:Number}
    
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
        return 3, iter, Δ
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
                    return 2, iter, Δ
                end
                if iter==gmres.maxiter
                    break
                end
            end
            update!(x,gmres.restart,H,s,y,V)
            mul!(r,A,x)  # r = A⋅x
            @. r = b - r # r = b - A⋅x
            ldiv!(M,r)   # r = M \ (b - A⋅x)
            β    = norm(r)
            Δ    = β/normb
            if Δ<gmres.tol
                return 1, iter, Δ
            end
        end
    end
    return 0, iter, Δ
end


#######################
## PRIVATE FUNCTIONS ##
#######################

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

function apply_plane_rotation(dx::T,dy::T,cs::T,sn::T)::Tuple{T,T} where {T<:Number}
    
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