import Base: eltype, size, length, *
import LinearAlgebra: mul!, ldiv!, transpose!

using LinearAlgebra
using SparseArrays
using Printf
using Random

using ..Checkerboard: checkerboard_mul!, checkerboard_transpose_mul!
using ..Utilities: get_index

import ..ConjugateGradients
using  ..ConjugateGradients: ConjugateGradient

import ..RestartedGMRES
using  ..RestartedGMRES: GMRES

export mulM!, mulMᵀ!, mulMᵀM!, muldMdx!, construct_M, write_M_matrix


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


"""
Iteratively solve the linear system M⋅x=b ==> x=M⁻¹⋅b or MᵀM⋅x=b ==> x=[MᵀM]⁻¹⋅b
"""
function ldiv!(x::AbstractVector{T2}, holstein::HolsteinModel{T1,T3}, b::AbstractVector{T2}, P=I)::Int where {T1<:AbstractFloat,T2<:Number,T3<:Number}

    # keeps track of number of iterations for iterative solver to execute.
    iters = 0

    if holstein.mul_by_M
        # Solve M⋅x=b  ==> x=M⁻¹⋅b using GMRES + Preconditioning (transposed=false) OR
        # Solve Mᵀ⋅x=b ==> x=M⁻ᵀ⋅b using GMRES + Preconditioning (transposed=true)
        flag, iters, Δ = RestartedGMRES.solve!(x, holstein, b, holstein.gmres, P)
    else
        # Solve MᵀM⋅x=b ==> x=[MᵀM]⁻¹⋅b using Conjugate Gradient
        iters = ConjugateGradients.solve!(x, holstein, b, holstein.cg)
    end

    return iters
end


"""
Tranpose M ⇆ Mᵀ with regards to application of mul! routine.
"""
function transpose!(holstein::HolsteinModel)

    holstein.transposed = !holstein.transposed
    return nothing
end


"""
Overloading `*` operator from Base
"""
function *(holstein::HolsteinModel{T1,T2},v::AbstractVector{T2})::Vector{T2} where {T1<:AbstractFloat,T2<:Number}

    y = Vector{T2}(undef,holstein.nindices)
    mul!(y,holstein,v)
    return y
end


"""
Default multiplication routine for HolsteinModel type.
"""
function mul!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T3},v::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Number,T3<:Number}

    if !holstein.mul_by_M
        mulMᵀM!(y,holstein,v)
    elseif !holstein.transposed
        mulM!(y,holstein,v)
    else
        mulMᵀ!(y,holstein,v)
    end
    return nothing
end


"""
Perform the multiplication y = MᵀM⋅v
"""
function mulMᵀM!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T3},v::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Number,T3<:Number}

    # y' = M⋅v
    mulM!(holstein.ytemp, holstein, v)

    # y  = Mᵀ⋅y' = MᵀM⋅v
    mulMᵀ!(y, holstein, holstein.ytemp)

    return nothing
end


function mulM!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T3},v::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Number,T3<:Number}

    ####################################
    ## PERFORM MULTIPLICATION y = M⋅v ##
    ####################################

    # Notes:
    # • y(τ) = [M⋅v](τ) = v(τ) - B(τ+1)⋅v(τ+1) for τ < Lτ
    # • y(τ) = [M⋅v](τ) = v(τ) + B(τ+1)⋅v(τ+1) for τ = Lτ
    # • B(τ) = exp{-Δτ⋅V[x(τ)]} exp{-Δτ⋅K}
    # • exp{-Δτ⋅V[x(τ)]} is the exponentiated interaction matrix and is diagonal,
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

    return nothing
end


function mulMᵀ!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T3},v::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Number,T3<:Number}

    #####################################
    ## PERFORM MULTIPLICATION y = Mᵀ⋅v ##
    #####################################

    # Notes:
    # • y(τ) = [Mᵀ⋅v](τ) = v(τ) - Bᵀ(τ)⋅v(τ-1)  for τ > 1
    # • y(τ) = [Mᵀ⋅v](τ) = v(τ) + Bᵀ(τ)⋅v(τ-1)  for τ = 1
    # • Bᵀ(τ) = exp{-Δτ⋅K}ᵀ⋅exp{-Δτ⋅V[x(τ)]}ᵀ 
    # • exp{-Δτ⋅V[x(τ)]} is the exponentiated interaction matrix and is diagonal,
    #   and as such is stored as a vector
    # • [exp{-Δτ⋅K}]ᵀ is given by adjoint of the checkerboard approximation matrix.

    # iterating over sites in lattice
    @fastmath @inbounds for i in 1:holstein.nsites

        # iterating over imaginary time slices
        for τ in 2:holstein.Lτ

            # y(τ) = -exp{-Δτ⋅V[x(τ)]}ᵀ⋅v(τ-1) for τ > 1
            idx_τm = get_index(τ-1, i, holstein.Lτ)
            idx_τ  = get_index(τ,   i, holstein.Lτ)
            y[idx_τ] = -conj(holstein.expnΔτV[idx_τ]) * v[idx_τm]
        end

        # y(1) = +exp{-Δτ⋅V[x(1)]}ᵀ⋅v(Lτ) for τ=1
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

    return nothing
end


"""
Performs the multiplication y = (dM/dx)⋅v
""" 
function muldMdx!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T3},v::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Number,T3<:Number}

    ########################################
    ## PERFORM MULTIPLICATION y = ∂M/∂x⋅v ##
    ########################################

    # Notes:
    # • Consider y = ∂M/∂xᵢ(τ)⋅v ==>
    #
    # • yᵢ(τ-1) = -∂B/∂xᵢ(τ)⋅vᵢ(τ) for τ > 1
    # • yᵢ(Lτ)  = +∂B/∂xᵢ(1)⋅vᵢ(1) for τ = 1
    #
    # • B(τ) = exp{-Δτ⋅V[x(τ)]}⋅exp{-Δτ⋅K}
    # • ∂B/∂xᵢ(τ) = -Δτ⋅dV/dxᵢ(τ)⋅exp{-Δτ⋅V[x(τ)]}⋅exp{-Δτ⋅K}
    # • ∂B/∂xᵢ(τ) = -Δτ⋅   λᵢ    ⋅exp{-Δτ⋅V[x(τ)]}⋅exp{-Δτ⋅K}
    #
    # • Therefore the final expression is:
    # • yᵢ(τ-1) = +Δτ⋅λᵢ⋅exp{-Δτ⋅V[x(τ)]}⋅exp{-Δτ⋅K}⋅vᵢ(τ) for τ > 1
    # • yᵢ(Lτ)  = -Δτ⋅λᵢ⋅exp{-Δτ⋅V[x(1)]}⋅exp{-Δτ⋅K}⋅vᵢ(1) for τ = 1
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
Returns the M matrix as a sparse matrix.
"""
function construct_M(holstein::HolsteinModel{T1,T2}, threshold::T1=1e-10) where {T1<:AbstractFloat,T2<:Number}

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

    return rows,cols,vals
end

"""
Write M matrix to file.
"""
function write_M_matrix(holstein::HolsteinModel{T1,T2}, filename::String, threshold::T1=1e-10) where {T1<:AbstractFloat,T2<:Number}

    # construct M matrix
    rows, cols, vals = construct_M(holstein,threshold)

    # open file
    open(filename,"w") do file

        # write header to file
        write(file,"col row real imag\n")

        # iterate over non-zero matrix elements
        for i in 1:length(rows)

            # write matrix element to file
            write( file , @sprintf( "%d %d %.6f %.6f\n", cols[i], rows[i], real(vals[i]), imag(vals[i]) ) )
        end
    end

    return nothing
end