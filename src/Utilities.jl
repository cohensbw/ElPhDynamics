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
    return reshape(view(a, :), dims)
end

function reshaped(a::AbstractArray{T,M}, dims::NTuple{N,Int}) where {T,N,M}
    return reshape(a, dims)
end

function reshaped(a::AbstractArray{T,M}, dims...) where {T,M}
    return reshaped(a, dims)
end

"""
Translationally shift elements in vector corresponding to a displacement vector [l1,l2,l3] given in terms
of unit cells.
"""
function translational_shift!(vout::AbstractArray,vin::AbstractArray,L1::Int,L2::Int,L3::Int,l1::Int,l2::Int,l3::Int)

    @assert length(vout)==length(vin)
    NL = length(vout)
    N  = L1*L2*L3
    L  = div(NL,N)
    @assert mod(NL,L)==0
    uin  = reshaped(vin,L,L1,L2,L3)
    uout = reshaped(vout,L,L1,L2,L3)
    circshift!(uout,uin,(0,l1,l2,l3))
    return nothing
end

"""
Averages over translational symmetry between arrays `f` and `g` by doing an FFT accelerated convolution.
The result is output into the array `fg`. The arrays `f` and `g` are left modified by this function.
"""
function translational_average!(fg::AbstractArray{T},f::AbstractArray{T},g::AbstractArray{T}) where {T<:Complex}
    
    fft!(f)
    fft!(g)
    N = length(f)
    copyto!(fg,f)
    circshift!(f, fg, size(fg,d)-1 for d in 1:ndims(fg) )
    for (i,f_reverse) in enumerate(Iterators.reverse(f))
        fg[i] = f_reverse * g[i] / N
    end
    ifft!(fg)
    return nothing
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