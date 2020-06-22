module Utilities

##############################
## INDEX MAPPTING FUNCTIONS ##
##############################

"""
maps (τ,site) ==> index in vector
"""
@inline function get_index(τ::Int, site::Int, Lτ::Int)::Int

    return (site-1)*Lτ + τ
end

"""
maps index in vector ==> site in lattice
"""
@inline function get_site(index::Int, Lτ::Int)::Int

    return div(index-1,Lτ) + 1
end

"""
maps index in vector ==> τ imaginary time slice
"""
@inline function get_τ(index::Int, Lτ::Int)::Int

    return (index-1)%Lτ + 1
end

####################
## MATH FUNCTIONS ##
####################

"""
Delta function.
"""
@inline function δ(i::T,j::T)::T where {T<:Number}

    return i==j
end

"""
Heaviside step function.
"""
@inline function θ(i::T)::T where {T<:Number}

    return i>0
end

#########################
## INTEGRATION METHODS ##
#########################

"""
Simposon integration rule on periodic vector.
"""
function simpson_periodic(f::AbstractVector{T1}, dx::T2)::T1  where {T1<:Number,T2<:AbstractFloat}

    # number of intervals
    N = length(f)

    # integrated value
    F = T1(0.0)

    @inbounds @fastmath for i in 2:2:N
        F += f[i-1]         * 1.0/3.0 * dx
        F += f[i]           * 4.0/3.0 * dx
        F += f[mod1(i+1,N)] * 1.0/3.0 * dx
    end

    if isodd(N)
        F -= f[N-1] * 1.0/12.0 * dx
        F += f[N]   * 2.0/3.0  * dx
        F += f[1]   * 5.0/12.0 * dx
    end

    return F
end

"""
Trapezoid Integration.
"""
function trapezoid(f::AbstractVector{T1}, dx::T2; extrapolate::Bool=false)::T1  where {T1<:Number,T2<:AbstractFloat}

    N = length(f)
    F = T1(0.0)
    @inbounds @fastmath for i in 2:N
        F += (f[i-1]+f[i])/2 * dx
    end
    if extrapolate
        df = f[N] - f[N-1]
        F += (2*f[N]+df)/2 * dx
    end
    return F
end

end