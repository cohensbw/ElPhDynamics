import Base: eltype, size, length, *
import LinearAlgebra: mul!

using LinearAlgebra
using SparseArrays
using UnsafeArrays
using Langevin.Checkerboard: checkerboard_mul!, checkerboard_transpose_mul!

export mulM!, mulMᵀ!, mulMᵀM!, muldMdϕ!, construct_M
export view_by_site!, view_by_τ!


# overload `eltype` from Base
function eltype(holstein::HolsteinModel{T1,T2})::DataType where {T1<:AbstractFloat,T2<:Number}

    return T2
end

# overloading `size` from Base
function length(holstein::HolsteinModel{T1,T2})::Int where {T1<:AbstractFloat,T2<:Number}

    return holstein.nindices
end


# overloading `size` from Base
function size(holstein::HolsteinModel{T1,T2})::Typle{Int,Int} where {T1<:AbstractFloat,T2<:Number}

    return (holstein.nindices, holstein.nindices)
end

function size(holstein::HolsteinModel{T1,T2},dim::Int)::Int where {T1<:AbstractFloat,T2<:Number}

    return holstein.nindices
end


# overloading `*` operator from Base
function *(holstein::HolsteinModel{T1,T2},v::AbstractVector{T2})::Vector{T2} where {T1<:AbstractFloat,T2<:Number}

    y = Vector{T2}(undef,holstein.nindices)
    mul!(y,holstein,v)
    return y
end


"""
Perform specified multiplicaiton by M matrix.
"""
function mul!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T2},v::AbstractVector{T2})  where {T1<:AbstractFloat,T2<:Number}

    mulMᵀM!(y,holstein,v)
    # mulM!(y,holstein,v)
end


"""
Perform the multiplication y = MᵀM⋅v
"""
function mulMᵀM!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T2},v::AbstractVector{T2})  where {T1<:AbstractFloat,T2<:Number}

    # y' = M⋅v
    mulM!(holstein.y′, holstein, v)

    # y = Mᵀ⋅y' = MᵀM⋅v
    mulMᵀ!(y, holstein, holstein.y′)
end


"""
Perform the multiplication y = M⋅v
"""
function mulM!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T2},v::AbstractVector{T2})  where {T1<:AbstractFloat,T2<:Number}

    ####################################
    ## PERFORM MULTIPLICATION y = M⋅v ##
    ####################################

    # Notes:
    # • y(τ) = [M⋅v](τ) = v(τ) - B(τ+1)⋅v(τ+1) for τ < Lτ
    # • y(τ) = [M⋅v](τ) = v(τ) + B(τ+1)⋅v(τ+1) for τ = Lτ
    # • B(τ) = exp{-Δτ⋅V[ϕ(τ)]} exp{-Δτ⋅K}
    # • exp{-Δτ⋅V[ϕ(τ)]} is the exponentiated interaction matrix and is diagonal,
    #   and as such is stored as a vector
    # • exp{-Δτ⋅K} is given by the checkerboard approximation matrix.

    neighbor_table_tij = holstein.neighbor_table_tij::Matrix{Int}
    coshtij    = holstein.coshtij::Vector{T2}
    sinhtij    = holstein.sinhtij::Vector{T2}
    expnΔτV    = holstein.expnΔτV::Vector{T2}
    yτ′        = holstein.yτ′::Vector{T2}
    Lτ         = holstein.Lτ::Int
    nsites     = holstein.lattice.nsites::Int
    τp1        = 1
    offset_τ   = 1
    offset_τp1 = 1

    # iterate over imaginary time axis
    for τ in 1:Lτ

        # get the τ+1 time slice account for periodic boundary conditions
        τp1 = τ%Lτ+1

        # indexing offset into vectors associated with τ time slice
        offset_τ = (τ-1)*nsites

        # indexing offset into vectors associated with τ+1 time slice
        offset_τp1 = (τp1-1)*nsites

        # y(τ) = v(τ+1)
        for i in 1:nsites
            yτ′[i] = v[i+offset_τp1]
        end

        # y(τ) = exp{-Δτ⋅K}⋅v(τ+1)
        checkerboard_mul!(yτ′,neighbor_table_tij,coshtij,sinhtij)

        if τ<Lτ
            # y(τ) = v(τ) - B(τ+1)⋅v(τ+1)
            for i in 1:nsites
                y[i+offset_τ] = v[i+offset_τ] - expnΔτV[i+offset_τp1] * yτ′[i]
            end
        else
            # y(τ) = v(τ) + B(τ+1)⋅v(τ+1)
            for i in 1:nsites
                y[i+offset_τ] = v[i+offset_τ] + expnΔτV[i+offset_τp1] * yτ′[i]
            end
        end
    end

    return nothing
end


"""
Perform the multiplication y = Mᵀ⋅v
"""
function mulMᵀ!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T2},v::AbstractVector{T2})  where {T1<:AbstractFloat,T2<:Number}

    #####################################
    ## PERFORM MULTIPLICATION y = Mᵀ⋅v ##
    #####################################

    # Notes:
    # • y(τ) = [Mᵀ⋅v](τ) = v(τ) - Bᵀ(τ)⋅v(τ-1)  for τ > 1
    # • y(τ) = [Mᵀ⋅v](τ) = v(τ) + Bᵀ(τ)⋅v(τ-1)  for τ = 1
    # • Bᵀ(τ) = exp{-Δτ⋅K}ᵀ exp{-Δτ⋅V[ϕ(τ)]}ᵀ 
    # • exp{-Δτ⋅V[ϕ(τ)]} is the exponentiated interaction matrix and is diagonal,
    #   and as such is stored as a vector
    # • [exp{-Δτ⋅K}]ᵀ is given by adjoint of the checkerboard approximation matrix.

    neighbor_table_tij = holstein.neighbor_table_tij::Matrix{Int}
    coshtij    = holstein.coshtij::Vector{T2}
    sinhtij    = holstein.sinhtij::Vector{T2}
    expnΔτV    = holstein.expnΔτV::Vector{T2}
    yτ′        = holstein.yτ′::Vector{T2}
    Lτ         = holstein.Lτ::Int
    nsites     = holstein.lattice.nsites::Int
    τm1        = 1
    offset_τ   = 1
    offset_τm1 = 1

    # iterate over imaginary time axis
    for τ in 1:Lτ

        # get the τ-1 time slice account for periodic boundary conditions
        τm1 = (τ+Lτ-2)%Lτ+1

        # indexing offset into vectors associated with τ time slice
        offset_τ = (τ-1)*nsites

        # indexing offset into vectors associated with τ+1 time slice
        offset_τm1 = (τm1-1)*nsites

        # y(τ) = exp{-Δτ⋅V[ϕ(τ)]}ᵀ⋅v(τ-1)
        for i in 1:nsites
            yτ′[i] = conj(expnΔτV[i+offset_τ]) * v[i+offset_τm1]
        end

        # y(τ) = Bᵀ(τ)⋅v(τ-1) = exp{-Δτ⋅K}ᵀ⋅exp{-Δτ⋅V[ϕ(τ)]}ᵀ⋅v(τ-1)
        checkerboard_transpose_mul!(yτ′,neighbor_table_tij,coshtij,sinhtij)

        # finish up the multiplication to get final y(τ) vector
        if τ>1
            # y(τ) = v(τ) - Bᵀ(τ)⋅v(τ-1)
            for i in 1:nsites
                y[i+offset_τ] = v[i+offset_τ] - yτ′[i]
            end
        else
            # y(τ) = v(τ) + Bᵀ(τ)⋅v(τ-1)
            for i in 1:nsites
                y[i+offset_τ] = v[i+offset_τ] + yτ′[i]
            end
        end
    end

    return nothing
end


"""
Performs the multiplication y = (dM/dϕ)⋅v
""" 
function muldMdϕ!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T2},v::AbstractVector{T2})  where {T1<:AbstractFloat,T2<:Number}

    ########################################
    ## PERFORM MULTIPLICATION y = ∂M/∂ϕ⋅v ##
    ########################################

    # Notes:
    # • Consider y = ∂M/∂ϕᵢ(τ)⋅v ==>
    #
    # • yᵢ(τ-1) = -∂B/∂ϕᵢ(τ)⋅vᵢ(τ) for τ < Lτ
    # • yᵢ(τ-1) = +∂B/∂ϕᵢ(τ)⋅vᵢ(τ) for τ = Lτ
    #
    # • B(τ) = exp{-Δτ⋅V[ϕ(τ)]} exp{-Δτ⋅K}
    # • ∂B/∂ϕᵢ(τ) = -Δτ ⋅ dV/dϕᵢ(τ) ⋅ exp{-Δτ⋅V[ϕ(τ)]} ⋅ exp{-Δτ⋅K}
    # • ∂B/∂ϕᵢ(τ) = -Δτ ⋅    λᵢ     ⋅ exp{-Δτ⋅V[ϕ(τ)]} ⋅ exp{-Δτ⋅K}
    #
    # • Therefore the final expression is:
    # • yᵢ(τ-1) =  Δτ⋅λᵢ⋅exp{-Δτ⋅V[ϕ(τ)]}⋅exp{-Δτ⋅K}⋅vᵢ(τ) for τ < Lτ
    # • yᵢ(τ-1) = -Δτ⋅λᵢ⋅exp{-Δτ⋅V[ϕ(τ)]}⋅exp{-Δτ⋅K}⋅vᵢ(τ) for τ = Lτ

    neighbor_table_tij = holstein.neighbor_table_tij::Matrix{Int}
    coshtij    = holstein.coshtij::Vector{T2}
    sinhtij    = holstein.sinhtij::Vector{T2}
    expnΔτV    = holstein.expnΔτV::Vector{T2}
    yτ′        = holstein.yτ′::Vector{T2}
    λ          = holstein.λ::Vector{T1}
    Δτ         = holstein.Δτ::T1
    Lτ         = holstein.Lτ::Int
    nsites     = holstein.lattice.nsites::Int
    τm1        = 0
    offset_τ   = 1
    offset_τm1 = 1

    # iterate over imaginary time slice
    for τ in 1:Lτ

        # get the τ-1 time slice account for periodic boundary conditions for τ < Lτ
        τm1 = (τ+Lτ-2)%Lτ+1

        # indexing offset into vectors associated with τ time slice
        offset_τ = (τ-1)*nsites

        # indexing offset into vectors associated with τ+1 time slice
        offset_τm1 = (τm1-1)*nsites

        # start by setting y(τ) = v(τ)
        for i in 1:nsites
            yτ′[i] = v[i+offset_τ]
        end

        # multiply by the checkerboard matrix exp{-Δτ⋅K}
        checkerboard_mul!(yτ′,neighbor_table_tij,coshtij,sinhtij)

        # finish up by multiplying by Δτ⋅λ⋅exp{-Δτ⋅V[ϕ(τ+1)]}
        if τ<Lτ
            for i in 1:nsites
                y[i+offset_τm1] =  Δτ * λ[i] * expnΔτV[i+offset_τ] * yτ′[i]
            end
        else
            for i in 1:nsites
                y[i+offset_τm1] = -Δτ * λ[i] * expnΔτV[i+offset_τ] * yτ′[i]
            end
        end
    end

    return nothing
end


"""
Returns the M matrix at a sparse matrix.
"""
function construct_M(holstein::HolsteinModel{T1,T2},threshold::T1=0.0) where {T1<:AbstractFloat,T2<:Number}

    # to contain M[row,col]=val info for constructing sparse matrix
    rows = Int[]
    cols = Int[]
    vals = T2[]

    # size of holstein model
    L = length(holstein)

    # represents unit vector
    unitvector = zeros(T2,L)

    # stores columns vector
    colvector = zeros(T2,L)

    # iterating over rows
    for col in 1:L

        # constructing unitvector
        unitvector[(col+L-2)%L+1] = 0.0
        unitvector[col]           = 1.0

        # multiply unit vector by M matrix
        mulM!(colvector,holstein,unitvector)

        # iterate of column vecto
        for row in 1:L

            # if nonzero
            if abs(colvector[row])>threshold

                # save matrix element
                append!(rows,row)
                append!(cols,col)
                append!(vals,colvector[row])
            end
        end
    end

    return sparse(rows,cols,vals)
end


"""
Give a vector of length `nindices=nsites⋅Lτ`, this function returns a view into
that vector for all time slices `τ` associated with a given `site` in the lattice.
"""
function view_by_site(v::AbstractVector,site::Int,nsites::Int)

    return @view v[site:nsites:end]
end


"""
Given a vector of length `nindices=nsites⋅Lτ`, this function returns a view into
that vector of all `sites` in the lattice for fixed `τ`.
"""
function view_by_τ(v::AbstractVector,τ::Int,nsites::Int)

    return @view v[(τ-1)*nsites+1:τ*nsites]
end