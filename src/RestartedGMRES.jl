module RestartedGMRES

using LinearAlgebra
using UnsafeArrays
using SparseArrays

struct GMRES{T1<:Number,T2<:AbstractFloat}
    
    ndim::Int
    maxiter::Int
    restart::Int
    tol::T2
    
    H::Matrix{T1}
    V::Matrix{T1}
    
    r::Vector{T1}
    w::Vector{T1}
    
    s::Vector{T1}
    y::Vector{T1}
    sn::Vector{T1}
    cs::Vector{T1}
    
    function GMRES(x::AbstractVector{T1}; maxiter::Int=-1,restart::Int=-1,tol::T2=1e-4) where {T1<:Number,T2<:AbstractFloat}
        
        ndim = length(x)
        if maxiter<0
            maxiter = 2*ndim
        end
        if restart<0
            restart = min(20,2*ndim)
        end
        H = zeros(T1,restart+1,restart)
        V = zeros(T1,ndim,restart+1)
        r = zeros(T1,ndim)
        w = zeros(T1,ndim)
        s = zeros(T1,restart+1)
        y = zeros(T1,restart+1)
        sn = zeros(T1,restart+1)
        cs = zeros(T1,restart+1)
        return new{T1,T2}(ndim,maxiter,restart,tol,H,V,r,w,s,y,sn,cs)
    end
end


function solve!(x::AbstractVector{T1},A,b::AbstractVector{T1},gmres::GMRES{T1,T2},M=I) where {T1<:Number,T2<:AbstractFloat}
    
    ndim    = gmres.ndim::Int
    maxiter = gmres.maxiter::Int
    restart = gmres.restart::Int
    tol     = gmres.tol::T2
    H       = gmres.H::Matrix{T1}
    V       = gmres.V::Matrix{T1}
    r       = gmres.r::Vector{T1}
    w       = gmres.w::Vector{T1}
    s       = gmres.s::Vector{T1}
    y       = gmres.y::Vector{T1}
    sn      = gmres.sn::Vector{T1}
    cs      = gmres.cs::Vector{T1}
    
    # calculating norm of b
    @. r = b
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
    if Δ<tol
        return 3, iter, Δ
    end
    
    @uviews V begin
        @fastmath @inbounds while iter < maxiter        
            @. V[:,1] = r/β
            @. s = 0.0
            s[1] = β
            for i in 1:restart
                iter += 1
                vi = @view V[:,i]
                mul!(w,A,vi)
                ldiv!(M,w)
                for k in 1:i
                    vk = @view V[:,k]
                    H[k,i] = dot(w,vk)
                    @. w = w - H[k,i] * vk
                end
                H[i+1,i] = norm(w)
                @. V[:,i+1] = w / H[i+1,i]
                for k in 1:i-1
                    H[k,i], H[k+1,i] = apply_plane_rotation(H[k,i], H[k+1,i], cs[k], sn[k])
                end
                cs[i], sn[i] = generate_plane_rotation(H[i,i], H[i+1,i])
                H[i,i], H[i+1,i] = apply_plane_rotation(H[i,i], H[i+1,i], cs[i], sn[i])
                s[i], s[i+1] = apply_plane_rotation(s[i], s[i+1], cs[i], sn[i])
                Δ = abs(s[i+1])/normb
                if Δ < tol
                    update!(x,i,H,s,y,V)
                    return 2, iter, Δ
                end
                if iter==maxiter
                    break
                end
            end
            update!(x,restart,H,s,y,V)
            mul!(r,A,x)  # r = A⋅x
            @. r = b - r # r = b - A⋅x
            ldiv!(M,r)   # r = M \ (b - A⋅x)
            β = norm(r)
            Δ = β/normb
            if Δ<tol
                return 1, iter, Δ
            end
        end
    end
    return 0, iter, Δ
end


######################
## PRIVATE FUNCTION ##
######################

function update!(x::AbstractVector,k::Int,H::Matrix,s::AbstractVector,y::AbstractVector,V::AbstractMatrix)
    
    @. y = s
    @fastmath @inbounds for i in k:-1:1
        y[i] /= H[i,i]
        for j in i-1:-1:1
            y[j] = y[j] - H[j,i] * y[i]
        end
    end
    @fastmath @inbounds for j in 1:k
        @views @. x = x + V[:,j] * y[j]
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