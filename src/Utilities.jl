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

"""
Reshapes with zero allocations, returns an instance of Base.ReshapedArray.
Discussion found at: https://github.com/JuliaLang/julia/issues/24237
"""
function reshaped(a::Array{T,M}, dims::NTuple{N,Int}) where {T,N,M}
    return reshape(view(a, :), dims)
end

function reshaped(a::AbstractArray{T,M}, dims::NTuple{N,Int}) where {T,N,M}
    return reshape(a, dims)
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

@inline function δ(i::T)::T where {T<:Number}

    return i==T(0)
end

"""
Heaviside step function.
"""
@inline function θ(i::T)::T where {T<:Number}

    return i>0
end

end