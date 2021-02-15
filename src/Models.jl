module Models

using LinearAlgebra

using Logging
using Printf

import Base: eltype, size, length, *
import LinearAlgebra: mul!, ldiv!, transpose!

using ..IterativeSolvers: IterativeSolver, GMRES, ConjugateGradient, BiCGStab, solve!
using ..Checkerboard: checkerboard_mul!, checkerboard_transpose_mul!
using ..Utilities: get_index

export mulM!, mulMᵀ!, mulMᵀM!, muldMdx!, construct_M, write_M_matrix!

"""
Abstract type to represent continuous real or complex numbers.
"""
Continuous = Union{AbstractFloat,Complex{<:AbstractFloat}}


"""
Abstract type to represent type of bond/hopping in lattice.
"""
abstract type AbstractBond end


"""
Represent a type of bare hopping/bond in lattice.
"""
struct Bond{T<:Continuous} <: AbstractBond

    "Average hopping energy."
    t::T

    "Standard Deviation of hopping energy."
    σt::T

    "Starting site/orbital in unit cell."
    o₁::Int

    "Ending site/orbital in unit cell."
    o₂::Int

    "Displacement in unit cells."
    v::Vector{Int}

    function Bond(t::T,σt::T,o₁::Int,o₂::Int,v::AbstractVector{Int}) where {T<:Continuous}

        @assert length(v)==3
        v′ = zeros(T,3)
        copyto!(v′,v)
        return new{T}(t,σt,o₁,o₂,v′)
    end
end


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
function ldiv!(x::AbstractVector, model::AbstractModel{T1,T2,T3}, b::AbstractVector, P)::Tuple{Int,T1} where {T1,T2,T3}
    
    iters = solve!(x, model, b, model.solver, P)

    # calculate residual error
    v    = model.v‴
    mul!(v,model,x)
    @. v = v - b
    residual_error = norm(v)/norm(b)

    # if large residual error then attempt solve without preconditioner
    if residual_error > sqrt(model.solver.tol)

        @info("Large Residual Error = $residual_error, Iterations = $iters")
        logger = global_logger()
        flush(logger.stream)
        fill!(x,0)

        return ldiv!(x,model,b)
    else # if small residual error return result

        return (iters,residual_error)
    end
end

function ldiv!(x::AbstractVector, model::AbstractModel{T1,T2,T3}, b::AbstractVector)::Tuple{Int,T1} where {T1,T2,T3}

    iters = solve!(x, model, b, model.solver)

    # calculate residual error
    v    = model.v‴
    mul!(v,model,x)
    @. v = v - b
    residual_error  = norm(v)/norm(b)
    
    if residual_error > sqrt(model.solver.tol)

        write_phonons!(model, joinpath(model.datafolder, "failing_phonons.out"))
        write_M_matrix!(model, joinpath(model.datafolder, "failing_matrix.out"))
        error("Large Residual Error = $residual_error, Iterations = $iters")
    end

    return (iters,residual_error)
end


"""
Default multiplication routine for AbstractModel type.
"""
function mul!(y::AbstractVector{T2},model::AbstractModel{T1,T2},v::AbstractVector{T2}) where {T1,T2}

    if model.mul_by_M
        if model.transposed
            mulMᵀ!(y,model,v)
        else
            mulM!(y,model,v)
        end
    else
        if model.transposed
            mulMMᵀ!(y,model,v)
        else
            mulMᵀM!(y,model,v)
        end
    end

    return nothing
end


"""
Perform the multiplication y = MᵀM⋅v
"""
function mulMᵀM!(y::AbstractVector{T2},model::AbstractModel{T1,T2},v::AbstractVector{T2}) where {T1,T2}

    # y' = M⋅v
    mulM!(model.v′, model, v)

    # y  = Mᵀ⋅y' = MᵀM⋅v
    mulMᵀ!(y, model, model.v′)

    return nothing
end

"""
Perform the multiplication y = MMᵀ⋅v
"""
function mulMMᵀ!(y::AbstractVector{T2},model::AbstractModel{T1,T2},v::AbstractVector{T2}) where {T1,T2}

    # y' = Mᵀ⋅v
    mulMᵀ!(model.v′, model, v)

    # y  = M⋅y' = MMᵀ⋅v
    mulM!(y, model, model.v′)

    return nothing
end


"""
Tranpose M ⇆ Mᵀ (MᵀM ⇆ MMᵀ) with regards to application of mul! routine.
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

    return model.Ndim
end


"""
Dimension of M matrix.
"""
function size(model::AbstractModel)::Tuple{Int,Int}

    return model.Ndim, model.Ndim
end


"""
Dimension of M matrix.
"""
function size(model::AbstractModel,dim::Int)::Int

    return model.Ndim
end


"""
Assign data folder.
"""
function assign_datafolder!(model::AbstractModel,datafolder::String)

    model.datafolder = datafolder
    return nothing
end


"""
Returns the M matrix as a sparse matrix.
"""
function construct_M(model::AbstractModel{T1,T2}, threshold::T1=1e-10) where {T1,T2}

    # to contain M[row,col]=val info for constructing sparse matrix
    rows = Int[]
    cols = Int[]
    vals = T2[]

    # size of model model
    NLτ = size(model,1)

    # represents unit vector
    unitvector = zeros(T2,NLτ)

    # stores columns vector
    colvector = zeros(T2,NLτ)

    # iterating over rows
    for col in 1:NLτ

        # constructing unitvector
        unitvector[mod1(col-1,NLτ)] = 0.0
        unitvector[col]             = 1.0

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
function write_M_matrix!(model::AbstractModel{T1,T2}, filename::String, threshold::T1=1e-10) where {T1,T2}

    # construct M matrix
    rows, cols, vals = construct_M(model,threshold)

    # open file
    open(filename,"w") do file

        # write header to file
        write(file,"col row real imag\n")

        # iterate over non-zero matrix elements
        for i in 1:length(rows)

            # write matrix element to file
            write( file , @sprintf( "%d %d %.10f %.10f\n", cols[i], rows[i], real(vals[i]), imag(vals[i]) ) )
        end
    end

    return nothing
end

end