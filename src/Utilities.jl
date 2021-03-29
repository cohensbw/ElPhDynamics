module Utilities

using FFTW

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
    return reshaped(view(a, :), dims)
end

function reshaped(a::AbstractArray{T,M}, dims::NTuple{N,Int}) where {T,N,M}
    return reshaped(a, dims)
end

function reshaped(a::AbstractArray{T,M}, dims...) where {T,M}
    return reshaped(a, dims)
end

"""
Averages over translational symmetry between arrays `f` and `g` by doing an FFT accelerated convolution.
The result is output into the array `fg`. The arrays `f` and `g` are left modified by this function.
"""
function translational_average!(fg::AbstractArray{T},f::AbstractArray{T},g::AbstractArray{T}) where {T<:Complex}
    
    fft!(f)
    fft!(g)
    N  = length(f)
    f′ = fg
    circshift!(f′, f, size(fg,d)-1 for d in 1:ndims(fg))
    reverse!(f′)
    @. fg = f′ * g / N
    ifft!(fg)
    return nothing
end

"""
Simpson integration over vector.
"""
function simpson(f::AbstractVector{T1},dx::T2)::T1 where {T1<:Number,T2<:Number}

    L = length(f)
    F = T1(0.0)
    for i in 2:2:L-1
        F += dx * ( 1/3*f[i-1] + 4/3*f[i] + 1/3*f[i+1] )
    end
    if iseven(L)
        F += dx * ( 5/12*f[L] + 2/3*f[L-1] - 1/12*f[L-2] )
    end
    return F
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