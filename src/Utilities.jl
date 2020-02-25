module Utilities

using UnsafeArrays

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
Simposon integration rule.
"""
function simpson(f::AbstractVector{T1}, dx::T2, pbc::Bool)::T1  where {T1<:Number,T2<:AbstractFloat}

    # length of array
    n = length(f)

    # number of intervals
    N = n-1

    # if periodic boundary conditions
    if pbc
        N += 1
    end

    # integrated value
    F = T1(0.0)

    @inbounds @fastmath for i in 2:2:N
        F += f[i-1]   * 1.0/3.0 * dx
        F += f[i]     * 4.0/3.0 * dx
        F += f[i%n+1] * 1.0/3.0 * dx
    end

    if (N+1)%2==0
        F -= f[N-1]   * 1.0/12.0 * dx
        F += f[N]     * 2.0/3.0  * dx
        F += f[N%n+1] * 5.0/12.0 * dx
    end

    return F
end

end