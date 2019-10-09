import Base: eltype, size, length, *
import LinearAlgebra: mul!

using LinearAlgebra
using SparseArrays
using ..Checkerboard: checkerboard_mul!, checkerboard_transpose_mul!

export mulM!, mulMᵀ!, mulMᵀM!, muldMdϕ!, construct_M


# overload `eltype` from Base
function eltype(holstein::HolsteinModel{T1,T2})::DataType where {T1<:AbstractFloat,T2<:Number}

    return T2
end

# overloading `size` from Base
function length(holstein::HolsteinModel{T1,T2})::Int where {T1<:AbstractFloat,T2<:Number}

    return holstein.nindices
end


# overloading `size` from Base
function size(holstein::HolsteinModel{T1,T2})::Tuple{Int,Int} where {T1<:AbstractFloat,T2<:Number}

    return holstein.nindices, holstein.nindices
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

    # y  = Mᵀ⋅y' = MᵀM⋅v
    mulMᵀ!(y, holstein, holstein.y′)
end


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

    # y = v
    copyto!(y, v)

    # y(τ) = exp{-Δτ⋅K}⋅v(τ)
    checkerboard_mul!(y, holstein.neighbor_table_tij, holstein.coshtij, holstein.sinhtij, holstein.Lτ)

    # iterating over sites in lattice
    @fastmath @inbounds for i in 1:holstein.nsites

        # y(Lτ) = v(Lτ) + B(1)⋅v(1)
        idx_L = get_index(holstein.Lτ, i,  holstein.Lτ)
        idx_1 = get_index(1,           i,  holstein.Lτ)
        yL_temp = v[idx_L] + holstein.expnΔτV[idx_1] * y[idx_1]

        # iterating over time slices
        for τ in 1:(holstein.Lτ-1)

            # y(τ) = v(τ) - B(τ+1)⋅v(τ+1) for τ<Lτ
            idx_τ  = get_index(τ,   i, holstein.Lτ)
            idx_τp = get_index(τ+1, i, holstein.Lτ)
            y[idx_τ] = v[idx_τ] - holstein.expnΔτV[idx_τp] * y[idx_τp]
        end

        # y(Lτ) = v(Lτ) + B(1)⋅v(1)
        y[idx_L] = yL_temp
    end
end


function mulMᵀ!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T2},v::AbstractVector{T2})  where {T1<:AbstractFloat,T2<:Number}

    #####################################
    ## PERFORM MULTIPLICATION y = Mᵀ⋅v ##
    #####################################

    # Notes:
    # • y(τ) = [Mᵀ⋅v](τ) = v(τ) - Bᵀ(τ)⋅v(τ-1)  for τ > 1
    # • y(τ) = [Mᵀ⋅v](τ) = v(τ) + Bᵀ(τ)⋅v(τ-1)  for τ = 1
    # • Bᵀ(τ) = exp{-Δτ⋅K}ᵀ⋅exp{-Δτ⋅V[ϕ(τ)]}ᵀ 
    # • exp{-Δτ⋅V[ϕ(τ)]} is the exponentiated interaction matrix and is diagonal,
    #   and as such is stored as a vector
    # • [exp{-Δτ⋅K}]ᵀ is given by adjoint of the checkerboard approximation matrix.

    # iterating over sites in lattice
    @fastmath @inbounds for i in 1:holstein.nsites

        # iterating over imaginary time slices
        for τ in 2:holstein.Lτ

            # y(τ) = -exp{-Δτ⋅V[ϕ(τ)]}ᵀ⋅v(τ-1) for τ > 1
            idx_τm = get_index(τ-1, i, holstein.Lτ)
            idx_τ  = get_index(τ,   i, holstein.Lτ)
            y[idx_τ] = -conj(holstein.expnΔτV[idx_τ]) * v[idx_τm]
        end

        # y(1) = +exp{-Δτ⋅V[ϕ(1)]}ᵀ⋅v(Lτ) for τ=1
        idx_L = get_index(holstein.Lτ, i, holstein.Lτ)
        idx_1 = get_index(1,           i, holstein.Lτ)
        y[idx_1] = conj(holstein.expnΔτV[idx_1]) * v[idx_L ]
    end

    # y(τ) = -Bᵀ(τ)⋅v(τ-1) for τ > 1
    # y(τ) = +Bᵀ(τ)⋅v(τ-1) for τ = 1 
    checkerboard_transpose_mul!(y, holstein.neighbor_table_tij, holstein.coshtij, holstein.sinhtij, holstein.Lτ)

    # y(τ) = v(τ) - Bᵀ(τ)⋅v(τ-1) for τ > 1
    # y(τ) = v(τ) + Bᵀ(τ)⋅v(τ-1) for τ = 1
    y .+= v
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
    # • yᵢ(τ-1) = -∂B/∂ϕᵢ(τ)⋅vᵢ(τ) for τ > 1
    # • yᵢ(Lτ)  = +∂B/∂ϕᵢ(1)⋅vᵢ(1) for τ = 1
    #
    # • B(τ) = exp{-Δτ⋅V[ϕ(τ)]} exp{-Δτ⋅K}
    # • ∂B/∂ϕᵢ(τ) = -Δτ⋅dV/dϕᵢ(τ)⋅exp{-Δτ⋅V[ϕ(τ)]}⋅exp{-Δτ⋅K}
    # • ∂B/∂ϕᵢ(τ) = -Δτ⋅   λᵢ    ⋅exp{-Δτ⋅V[ϕ(τ)]}⋅exp{-Δτ⋅K}
    #
    # • Therefore the final expression is:
    # • yᵢ(τ-1) = +Δτ⋅λᵢ⋅exp{-Δτ⋅V[ϕ(τ)]}⋅exp{-Δτ⋅K}⋅vᵢ(τ) for τ > 1
    # • yᵢ(Lτ)  = -Δτ⋅λᵢ⋅exp{-Δτ⋅V[ϕ(1)]}⋅exp{-Δτ⋅K}⋅vᵢ(1) for τ = 1
    #
    # • Simplifying a little bit:
    # • yᵢ(τ-1) = +Δτ⋅λᵢ⋅B(τ)⋅vᵢ(τ) for τ > 1
    # • yᵢ(Lτ)  = -Δτ⋅λᵢ⋅B(1)⋅vᵢ(1) for τ = 1

    # y(τ) = v(τ)
    copyto!(y, v)

    # y(τ) = exp{-Δτ⋅K}⋅v(τ)
    checkerboard_mul!(y, holstein.neighbor_table_tij, holstein.coshtij, holstein.sinhtij, holstein.Lτ)
    
    # iterating over sites in lattice
    @fastmath @inbounds for i in 1:holstein.nsites

        # y(Lτ) = -Δτ⋅λᵢ⋅B(1)⋅v(1) for τ=1
        idx_1 = get_index(1,           i,  holstein.Lτ)
        idx_L = get_index(holstein.Lτ, i,  holstein.Lτ)
        yL_temp = -holstein.Δτ * holstein.λ[i] * holstein.expnΔτV[idx_1] * y[idx_1]

        # iterating over time slices
        for τ in 2:holstein.Lτ

            # y(τ-1) = +Δτ⋅λ⋅B(τ)⋅v(τ) for τ>1
            idx_τm1 = get_index(τ-1, i, holstein.Lτ)
            idx_τ   = get_index(τ  , i, holstein.Lτ)
            y[idx_τm1] = holstein.Δτ * holstein.λ[i] * holstein.expnΔτV[idx_τ] * y[idx_τ]
        end

        # y(Lτ) = -Δτ⋅λ⋅B(1)⋅v(1) for τ=1
        y[idx_L] = yL_temp
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
    NLτ = length(holstein)

    # represents unit vector
    unitvector = zeros(T2,NLτ)

    # stores columns vector
    colvector = zeros(T2,NLτ)

    # iterating over rows
    for col in 1:NLτ

        # constructing unitvector
        unitvector[(col+NLτ-2)%NLτ+1] = 0.0
        unitvector[col] = 1.0

        # multiply unit vector by M matrix
        mulM!(colvector,holstein,unitvector)

        # iterate of column vecto
        for row in 1:NLτ

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