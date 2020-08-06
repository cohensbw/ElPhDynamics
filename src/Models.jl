module Models

using ..IterativeSolvers: IterativeSolver

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

# Defining HolsteinModel type
include("HolsteinModels.jl")
include("HolsteinModelMatrix.jl")

end