import Base: eltype, size, length, *
import LinearAlgebra: mul!

using SparseArrays
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

    y = Vector{T2}(undef,length(holstein))
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
    mulM!(holstein.temporary_vector, holstein, v)

    # y = Mᵀ⋅y' = MᵀM⋅v
    mulMᵀ!(y, holstein, holstein.temporary_vector)
end


"""
Perform the multiplication y = M⋅v
"""
function mulM!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T2},v::AbstractVector{T2})  where {T1<:AbstractFloat,T2<:Number}

    neighbor_table_tij = holstein.neighbor_table_tij::Matrix{Int}
    coshtij = holstein.coshtij::Vector{T2}
    sinhtij = holstein.sinhtij::Vector{T2}
    expnΔτV = holstein.expnΔτV::Vector{T2}
    Lτ      = holstein.Lτ::Int
    nsites  = holstein.lattice.nsites::Int
    τp1     = 1

    ####################################
    ## PERFORM MULTIPLICATION y = M⋅v ##
    ####################################

    # iterate over imaginary time axis
    for τ in 1:Lτ

        # Notes:
        # • y(τ) = [M⋅v](τ) = v(τ) - B(τ+1)⋅v(τ+1) for τ < Lτ
        # • y(τ) = [M⋅v](τ) = v(τ) + B(τ+1)⋅v(τ+1) for τ = Lτ
        # • B(τ) = exp{-Δτ⋅V[ϕ(τ)]} exp{-Δτ⋅K}
        # • exp{-Δτ⋅V[ϕ(τ)]} is the exponentiated interaction matrix and is diagonal,
        #   and as such is stored as a vector
        # • exp{-Δτ⋅K} is given by the checkerboard approximation matrix.

        # get the τ+1 time slice account for periodic boundary conditions
        τp1 = τ%Lτ+1

        # get a view into y for current time slice τ
        yτ = view_by_τ(y,τ,nsites)
        # get a view into v for current time slice τ
        vτ = view_by_τ(v,τ,nsites)
        # getting view into v for time slice τ+1
        vτp1 = view_by_τ(v,τp1,nsites)
        # geting view into exp{-Δτ⋅V[ϕ(τ+1)]} matrix
        expnΔτV_τp1 = view_by_τ(expnΔτV,τp1,nsites)

        # first we need y(τ) = v(τ+1)
        yτ .= vτp1

        # next we need to multiply by exp{-Δτ⋅K}
        checkerboard_mul!(yτ,neighbor_table_tij,coshtij,sinhtij)

        # now we need to multiply by exp{-Δτ⋅V[ϕ(τ+1)]}
        yτ .*= expnΔτV_τp1
        # at this point y(τ) = B(τ+1)⋅v(τ+1)

        # finish up the multiplication to get final y(τ) vector
        if τ<Lτ
            # y(τ) = v(τ) - B(τ+1)⋅v(τ+1)
            @. yτ = vτ - yτ
        else
            # y(τ) = v(τ) + B(τ+1)⋅v(τ+1)
            yτ .+= vτ
        end

    end

    return nothing
end


"""
Perform the multiplication y = Mᵀ⋅v
"""
function mulMᵀ!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T2},v::AbstractVector{T2})  where {T1<:AbstractFloat,T2<:Number}

    neighbor_table_tij = holstein.neighbor_table_tij::Matrix{Int}
    coshtij = holstein.coshtij::Vector{T2}
    sinhtij = holstein.sinhtij::Vector{T2}
    expnΔτV = holstein.expnΔτV::Vector{T2}
    Lτ      = holstein.Lτ::Int
    nsites  = holstein.lattice.nsites::Int
    τm1     = 1
    sgn     = 1

    #####################################
    ## PERFORM MULTIPLICATION y = Mᵀ⋅v ##
    #####################################

    # iterate over imaginary time axis
    for τ in 1:Lτ

        # Notes:
        # • y(τ) = [Mᵀ⋅v](τ) = v(τ) - Bᵀ(τ)⋅v(τ-1)  for τ > 1
        # • y(τ) = [Mᵀ⋅v](τ) = v(τ) + Bᵀ(τ)⋅v(τ-1)  for τ = 1
        # • Bᵀ(τ) = [exp{-Δτ⋅K}]ᵀ [exp{-Δτ⋅V[ϕ(τ)]}]ᵀ 
        # • exp{-Δτ⋅V[ϕ(τ)]} is the exponentiated interaction matrix and is diagonal,
        #   and as such is stored as a vector
        # • [exp{-Δτ⋅K}]ᵀ is given by adjoint of the checkerboard approximation matrix.

        # get the τ-1 time slice account for periodic boundary conditions
        τm1 = (τ+Lτ-2)%Lτ+1

        # get a view into y for current time slice τ
        yτ = view_by_τ(y,τ,nsites)
        # get a view into v for current time slice τ
        vτ = view_by_τ(v,τ,nsites)
        # getting view into v for time slice τ-1
        vτm1 = view_by_τ(v,τm1,nsites)
        # geting view into exp{-Δτ⋅V[ϕ(τ)]} matrix
        expnΔτV_τ = view_by_τ(expnΔτV,τ,nsites)

        # first set y(τ) = [exp{-Δτ⋅V[ϕ(τ)]}]ᵀ⋅v(τ-1)
        @. yτ = conj(expnΔτV_τ) * vτm1

        # next we need to multiply by [exp{-Δτ⋅K}]ᵀ
        checkerboard_transpose_mul!(yτ,neighbor_table_tij,coshtij,sinhtij)
        # at this point y(τ) = Bᵀ(τ)⋅v(τ-1)

        # finish up the multiplication to get final y(τ) vector
        if τ>1
            @. yτ = vτ - yτ
        else
            yτ .+= vτ
        end

    end

    return nothing
end


"""
Performs the multiplication y = (dM/dϕ)⋅v
""" 
function muldMdϕ!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T2},v::AbstractVector{T2})  where {T1<:AbstractFloat,T2<:Number}

    neighbor_table_tij = holstein.neighbor_table_tij::Matrix{Int}
    coshtij = holstein.coshtij::Vector{T2}
    sinhtij = holstein.sinhtij::Vector{T2}
    expnΔτV = holstein.expnΔτV::Vector{T2}
    λ       = holstein.λ::Vector{T1}
    Δτ      = holstein.Δτ::T1
    Lτ      = holstein.Lτ::Int
    nsites  = holstein.lattice.nsites::Int
    τm1     = 0

    # iterate over imaginary time slice
    for τ in 1:Lτ

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

        # get the τ-1 time slice account for periodic boundary conditions for τ < Lτ
        τm1 = (τ+Lτ-2)%Lτ+1

        # get a view into y for time slice τ-1
        yτm1 = view_by_τ(y,τm1,nsites)
        # get a view into v for current time slice τ
        vτ = view_by_τ(v,τ,nsites)
        # geting exp{-Δτ⋅V[ϕ(τ)]} matrix
        expnΔτV_τ = view_by_τ(expnΔτV,τ,nsites)

        # start by setting y(τ-1) = v(τ)
        yτm1 .= vτ

        # multiply by the checkerboard matrix exp{-Δτ⋅K}
        checkerboard_mul!(yτm1,neighbor_table_tij,coshtij,sinhtij)

        # multiply by the diagonal matrix exp{-Δτ⋅V[ϕ(τ+1)]}
        yτm1 .*= expnΔτV_τ

        # multiply by (Δτ⋅dV/dϕ = Δτ⋅λ) next.
        if τ>1
            @. yτm1 *=  Δτ * λ
        else
            @. yτm1 *= -Δτ * λ
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