module Models

using LinearAlgebra

import Base: eltype, size, length, *
import LinearAlgebra: mul!, ldiv!, transpose!

using ..IterativeSolvers: IterativeSolver, GMRES, ConjugateGradient, BiCGStab, solve!
using ..Checkerboard: checkerboard_mul!, checkerboard_transpose_mul!
using ..Utilities: get_index

export mulM!, mulMᵀ!, mulMᵀM!, muldMdx!, construct_M, write_M_matrix

"""
Abstract type to represent continuous real or complex numbers.
"""
Continuous = Union{AbstractFloat,Complex{<:AbstractFloat}}


"""
Abstract type to represent models.
    T1: data type for parameter values that are always real
    T2: data type for matrix elements of M
    T3: data type describing which iterative solver is being used
"""
abstract type AbstractModel{T1<:AbstractFloat,T2<:Continuous,T3<:IterativeSolver} end

# include code for models
include("HolsteinModels.jl")
include("SSHModels.jl")


"""
Iteratively solve the linear system M⋅x=b ==> x=M⁻¹⋅b or MᵀM⋅x=b ==> x=[MᵀM]⁻¹⋅b.
"""
function ldiv!(x::AbstractVector, model::AbstractModel, b::AbstractVector, P)::Int

    iters = solve!(x, model, b, model.solver, P)
    return iters
end

function ldiv!(x::AbstractVector, model::AbstractModel, b::AbstractVector)::Int

    iters = solve!(x, model, b, model.solver)
    return iters
end


"""
Default multiplication routine for AbstractModel type.
"""
function mul!(y::AbstractVector{T2},model::AbstractModel{T1,T2},v::AbstractVector{T2}) where {T1,T2}

    if !model.mul_by_M
        mulMᵀM!(y,model,v)
    elseif !model.transposed
        mulM!(y,model,v)
    else
        mulMᵀ!(y,model,v)
    end
    return nothing
end


"""
Perform the multiplication y = MᵀM⋅v
"""
function mulMᵀM!(y::AbstractVector{T2},model::AbstractModel{T1,T2},v::AbstractVector{T2}) where {T1,T2}

    # y' = M⋅v
    mulM!(model.ytemp, model, v)

    # y  = Mᵀ⋅y' = MᵀM⋅v
    mulMᵀ!(y, model, model.ytemp)

    return nothing
end


"""
Tranpose M ⇆ Mᵀ with regards to application of mul! routine.
"""
function transpose!(model::AbstractModel)

    model.transposed = !model.transposed
    return nothing
end


"""
Data type of matrix elements of M matrix.
"""
function eltype(model::AbstractModel{T1,T2,T3})::DataType where {T1<:AbstractFloat,T2<:Number,T3<:IterativeSolver}

    return T2
end


"""
Dimension of M matrix.
"""
function length(model::AbstractModel)::Int

    return model.Ndims
end


"""
Dimension of M matrix.
"""
function size(model::AbstractModel)::Tuple{Int,Int}

    return model.Ndims, model.Ndim
end


"""
Dimension of M matrix.
"""
function size(model::AbstractModel,dim::Int)::Int

    return model.Ndims
end


"""
Returns the M matrix as a sparse matrix.
"""
function construct_M(model::AbstractModel{T1,T2}, threshold::T1=1e-10) where {T1<:AbstractFloat,T2<:Number}

    # to contain M[row,col]=val info for constructing sparse matrix
    rows = Int[]
    cols = Int[]
    vals = T2[]

    # size of model model
    NLτ = length(model)

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
        mulM!(colvector,model,unitvector)

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
function write_M_matrix(model::AbstractModel{T1,T2}, filename::String, threshold::T1=1e-10) where {T1<:AbstractFloat,T2<:Number}

    # construct M matrix
    rows, cols, vals = construct_M(model,threshold)

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

end