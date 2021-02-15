module Checkerboard

using LinearAlgebra
using SparseArrays

export checkerboard_groups!, checkerboard_groups, checkerboard_order!, checkerboard_order
export checkerboard_mul!, checkerboard_transpose_mul!, checkerboard_inverse_mul!, checkerboard_inverse_transpose_mul!
export checkerboard_matrix

"""
Construct full checkerboard matrix. For code 
testing rather than use in the final Langevin simulation.
"""
function checkerboard_matrix(neighbor_table::Matrix{Int},vals::AbstractVector{T2},Δτ::T1;transposed::Bool=false,threshold::T1=0.0) where {T1<:AbstractFloat,T2<:Number}
    
    # to contain M[row,col]=val info for constructing sparse matrix
    rows = Int[]
    cols = Int[]
    elements = T2[]
    # size of matrix
    N = maximum(neighbor_table)
    # stores columns vector
    colvector = zeros(T2,N)
    # iterating over rows
    for col in 1:N
        # initialize column vector as unit vector
        colvector      .= 0.0
        colvector[col]  = 1.0
        # doing matrix vector multiply
        if transposed==true
            # multiply unit vector by Mᵀ matrix
            checkerboard_transpose_mul!(colvector,neighbor_table,vals,Δτ,1)
        else
            # multiply unit vector by M matrix
            checkerboard_mul!(colvector,neighbor_table,vals,Δτ,1)
        end
        # iterate of column vecto
        for row in 1:N
            # if nonzero
            if abs(colvector[row])>threshold
                # save matrix element
                append!(rows,row)
                append!(cols,col)
                append!(elements,colvector[row])
            end
        end
    end
    return sparse(rows, cols, elements, N, N)
end


"""
In-place multiplication of vector with checkerboard matrix.
This method assumes the `neighbor_table` and associated `coshs` and `sinhs` are already ordered correctly
for the checkerboard decomposition.
"""
function checkerboard_mul!(y::AbstractVector{T1}, neighbor_table_tij::Matrix{Int}, coshtij::Vector{T2}, sinhtij::Vector{T2}, Lτ::Int) where {T1<:Number,T2<:Number}

    # iterating over pairs of neighboring sites
    @fastmath @inbounds for n in 1:length(coshtij)

        c = coshtij[n]
        s = sinhtij[n]
        i = neighbor_table_tij[1, n]
        j = neighbor_table_tij[2, n]

        # iterating over time slices
        @simd for τ in 1:Lτ

            # mapping (τ,site) ==> index
            idx_i = (i-1)*Lτ + τ
            idx_j = (j-1)*Lτ + τ

            t1 = y[idx_i]
            t2 = y[idx_j]

            y[idx_i] = c*t1 + s*t2
            y[idx_j] = c*t2 + conj(s)*t1
        end
    end

    return nothing
end


function checkerboard_mul!(y::AbstractVector{T1}, neighbor_table_tij::Matrix{Int}, coshtij::Matrix{T2}, sinhtij::Matrix{T2}) where {T1<:Number,T2<:Number}

    # number of bonds
    N = size(coshtij,2)

    # length of imaginary time axis
    Lτ = size(coshtij,1)

    # iterating over pairs of neighboring sites
    @fastmath @inbounds for n in 1:N

        i = neighbor_table_tij[1, n]
        j = neighbor_table_tij[2, n]

        # iterating over time slices
        @simd for τ in 1:Lτ

            # get matrix elements
            c = coshtij[τ,n]
            s = sinhtij[τ,n]

            # mapping (τ,site) ==> index
            idx_i = (i-1)*Lτ + τ
            idx_j = (j-1)*Lτ + τ

            t1 = y[idx_i]
            t2 = y[idx_j]

            # perform matrix-vector multiply
            y[idx_i] = c*t1 + s*t2
            y[idx_j] = c*t2 + conj(s)*t1
        end
    end

    return nothing
end

function checkerboard_mul!(y::AbstractVector{T1}, neighbor_table_tij::Matrix{Int}, coshtij::Vector{T2}, sinhtij::Vector{T2}) where {T1<:Number,T2<:Number}

    # iterating over pairs of neighboring sites
    @fastmath @inbounds @simd for n in 1:length(coshtij)

        c = coshtij[n]
        s = sinhtij[n]
        i = neighbor_table_tij[1, n]
        j = neighbor_table_tij[2, n]

        t1 = y[i]
        t2 = y[j]

        y[i] = c*t1 + s*t2
        y[j] = c*t2 + conj(s)*t1
    end

    return nothing
end


"""
In-place multiplication of vector with checkerboard matrix.
This method assumes the `neighbor_table` and associated `coshs` and `sinhs` are already ordered correctly
for the checkerboard decomposition.
"""
function checkerboard_transpose_mul!(y::AbstractVector{T1}, neighbor_table_tij::Matrix{Int}, coshtij::Vector{T2}, sinhtij::Vector{T2}, Lτ::Int)  where {T1<:Number,T2<:Number}

    # iterating over pairs of neighboring sites
    @fastmath @inbounds for n in length(coshtij):-1:1

        c = coshtij[n]
        s = sinhtij[n]
        i = neighbor_table_tij[1, n]
        j = neighbor_table_tij[2, n]

        # iterating over time slices
        @simd for τ in 1:Lτ

            # mapping (τ,site) ==> index
            idx_i = (i-1)*Lτ + τ
            idx_j = (j-1)*Lτ + τ

            t1 = y[idx_i]
            t2 = y[idx_j]

            y[idx_i] = c*t1 + s*t2
            y[idx_j] = c*t2 + conj(s)*t1
        end
    end

    return nothing
end

function checkerboard_transpose_mul!(y::AbstractVector{T1}, neighbor_table_tij::Matrix{Int}, coshtij::Matrix{T2}, sinhtij::Matrix{T2})  where {T1<:Number,T2<:Number}

    # number of bonds
    N = size(coshtij,2)

    # length of imaginary time axis
    Lτ = size(coshtij,1)

    # iterating over pairs of neighboring sites
    @fastmath @inbounds for n in N:-1:1

        i = neighbor_table_tij[1, n]
        j = neighbor_table_tij[2, n]

        # iterating over time slices
        @simd for τ in 1:Lτ

            c = coshtij[τ,n]
            s = sinhtij[τ,n]

            # mapping (τ,site) ==> index
            idx_i = (i-1)*Lτ + τ
            idx_j = (j-1)*Lτ + τ

            t1 = y[idx_i]
            t2 = y[idx_j]

            y[idx_i] = c*t1 + s*t2
            y[idx_j] = c*t2 + conj(s)*t1
        end
    end

    return nothing
end

function checkerboard_transpose_mul!(y::AbstractVector{T1}, neighbor_table_tij::Matrix{Int}, coshtij::Vector{T2}, sinhtij::Vector{T2})  where {T1<:Number,T2<:Number}

    # iterating over pairs of neighboring sites
    @fastmath @inbounds @simd for n in length(coshtij):-1:1

        c = coshtij[n]
        s = sinhtij[n]
        i = neighbor_table_tij[1, n]
        j = neighbor_table_tij[2, n]

        t1 = y[i]
        t2 = y[j]

        y[i] = c*t1 + s*t2
        y[j] = c*t2 + conj(s)*t1
    end

    return nothing
end


"""
In-place multiplication of vector with the inverse of the checkerboard matrix.
This method assumes the `neighbor_table` and associated `coshs` and `sinhs` are already ordered correctly
for the checkerboard decomposition.
"""
function checkerboard_inverse_mul!(y::AbstractVector{T1}, neighbor_table_tij::Matrix{Int}, coshtij::Vector{T2}, sinhtij::Vector{T2}, Lτ::Int)  where {T1<:Number,T2<:Number}

    # iterating over pairs of neighboring sites
    @fastmath @inbounds for n in length(coshtij):-1:1

        c = coshtij[n]
        s = sinhtij[n]
        i = neighbor_table_tij[1, n]
        j = neighbor_table_tij[2, n]

        # iterating over time slices
        @simd for τ in 1:Lτ

            # mapping (τ,site) ==> index
            idx_i = (i-1)*Lτ + τ
            idx_j = (j-1)*Lτ + τ

            t1 = y[idx_i]
            t2 = y[idx_j]

            y[idx_i] = c*t1 - s*t2
            y[idx_j] = c*t2 - conj(s)*t1
        end
    end

    return nothing
end

function checkerboard_inverse_mul!(y::AbstractVector{T1}, neighbor_table_tij::Matrix{Int}, coshtij::Matrix{T2}, sinhtij::Matrix{T2})  where {T1<:Number,T2<:Number}

    N  = size(coshtij,2)
    Lτ = size(coshtij,1)

    # iterating over pairs of neighboring sites
    @fastmath @inbounds for n in N:-1:1

        i = neighbor_table_tij[1, n]
        j = neighbor_table_tij[2, n]

        # iterating over time slices
        @simd for τ in 1:Lτ

            c = coshtij[τ,n]
            s = sinhtij[τ,n]

            # mapping (τ,site) ==> index
            idx_i = (i-1)*Lτ + τ
            idx_j = (j-1)*Lτ + τ

            t1 = y[idx_i]
            t2 = y[idx_j]

            y[idx_i] = c*t1 - s*t2
            y[idx_j] = c*t2 - conj(s)*t1
        end
    end

    return nothing
end

function checkerboard_inverse_mul!(y::AbstractVector{T1}, neighbor_table_tij::Matrix{Int}, coshtij::Vector{T2}, sinhtij::Vector{T2})  where {T1<:Number,T2<:Number}

    # iterating over pairs of neighboring sites
    @fastmath @inbounds for n in length(coshtij):-1:1

        c = coshtij[n]
        s = sinhtij[n]
        i = neighbor_table_tij[1, n]
        j = neighbor_table_tij[2, n]

        t1 = y[i]
        t2 = y[j]

        y[i] = c*t1 - s*t2
        y[j] = c*t2 - conj(s)*t1
    end

    return nothing
end

"""
In-place multiplication of vector with the inverse of the checkerboard matrix.
This method assumes the `neighbor_table` and associated `coshs` and `sinhs` are already ordered correctly
for the checkerboard decomposition.
"""
function checkerboard_inverse_transpose_mul!(y::AbstractVector{T1}, neighbor_table_tij::Matrix{Int}, coshtij::Vector{T2}, sinhtij::Vector{T2}, Lτ::Int)  where {T1<:Number,T2<:Number}

    # iterating over pairs of neighboring sites
    @fastmath @inbounds for n in 1:length(coshtij)

        c = coshtij[n]
        s = sinhtij[n]
        i = neighbor_table_tij[1, n]
        j = neighbor_table_tij[2, n]

        # iterating over time slices
        @simd for τ in 1:Lτ

            # mapping (τ,site) ==> index
            idx_i = (i-1)*Lτ + τ
            idx_j = (j-1)*Lτ + τ

            t1 = y[idx_i]
            t2 = y[idx_j]

            y[idx_i] = c*t1 - s*t2
            y[idx_j] = c*t2 - conj(s)*t1
        end
    end

    return nothing
end

function checkerboard_inverse_transpose_mul!(y::AbstractVector{T1}, neighbor_table_tij::Matrix{Int}, coshtij::Matrix{T2}, sinhtij::Matrix{T2})  where {T1<:Number,T2<:Number}

    N  = size(coshtij,2)
    Lτ = size(coshtij,1)

    # iterating over pairs of neighboring sites
    @fastmath @inbounds for n in 1:N

        i = neighbor_table_tij[1, n]
        j = neighbor_table_tij[2, n]

        # iterating over time slices
        @simd for τ in 1:Lτ

            c = coshtij[τ,n]
            s = sinhtij[τ,n]

            # mapping (τ,site) ==> index
            idx_i = (i-1)*Lτ + τ
            idx_j = (j-1)*Lτ + τ

            t1 = y[idx_i]
            t2 = y[idx_j]

            y[idx_i] = c*t1 - s*t2
            y[idx_j] = c*t2 - conj(s)*t1
        end
    end

    return nothing
end

function checkerboard_inverse_transpose_mul!(y::AbstractVector{T1}, neighbor_table_tij::Matrix{Int}, coshtij::Vector{T2}, sinhtij::Vector{T2})  where {T1<:Number,T2<:Number}

    # iterating over pairs of neighboring sites
    @fastmath @inbounds for n in 1:length(coshtij)

        c = coshtij[n]
        s = sinhtij[n]
        i = neighbor_table_tij[1, n]
        j = neighbor_table_tij[2, n]

        t1 = y[i]
        t2 = y[j]

        y[i] = c*t1 - s*t2
        y[j] = c*t2 - conj(s)*t1
    end

    return nothing
end


"""
    function checkerboard_order(neighbor_table::Matrix{Int})::Vector{Int}

Given a `neighbor_table`, this functions determines the `order` the checkerboard ordering
for the neighboring sites. Assumes the `neighbor_table` has already been sorted.
"""
function checkerboard_order(neighbor_table::Matrix{Int})::Vector{Int}

    nneighbor_table = size(neighbor_table,2)
    groups = zeros(Int,nneighbor_table)
    order = zeros(Int,nneighbor_table)
    checkerboard_groups!(groups,neighbor_table)
    checkerboard_order!(order,groups)
    return order
end


"""
    function checkerboard_order(groups::Vector{Int})::Vector{Int}

Given a checkerboard groupings of neighbor_table `groups`, this functions determines
the `order` the associated neighbor_table should be iterated over for performing Matrix
multiplications.
"""
function checkerboard_order(groups::Vector{Int})::Vector{Int}

    order = zeros(Int,length(groups))
    checkerboard_order!(order,groups)
    return order
end

"""
    function checkerboard_order!(order::Vector{Int},groups::Vector{Int})

Given a checkerboard groupings of neighbor_table `groups`, this functions determines
the `order` the associated neighbor_table should be iterated over for performing Matrix
multiplications.
"""
function checkerboard_order!(order::Vector{Int},groups::Vector{Int})

    sortperm!(order,groups)
    return nothing
end


"""
    checkerboard_groups(neighbor_table::Matrix{Int})::Vector{Int}

Constructs the checkerboard decomposition for a given neighbor table.
Assumes the neighbor table has already been sorted using the sort_neighbor_table!
method from the Lattices module.
Returns the number of groups in the checkerboard decomposition.
"""
function checkerboard_groups(neighbor_table::Matrix{Int})::Vector{Int}

    groups = zeros(Int,size(neighbor_table,2))
    checkerboard_groups!(groups,neighbor_table)
    return groups
end

"""
    checkerboard_groups!(groups::Vector{Int},neighbor_table::Matrix{Int})::Int

Constructs the checkerboard decomposition for a given neighbor table.
Assumes the neighbor table has already been sorted using the sort_neighbor_table! method from the Lattices module.
Returns the number of groups in the checkerboard decomposition.
"""
function checkerboard_groups!(groups::Vector{Int},neighbor_table::Matrix{Int})::Int

    # checking dimensions
    @assert size(neighbor_table,2)==length(groups)
    @assert size(neighbor_table,1)==2
    # getting the total number of neighbor pairs
    nneighbors = size(neighbor_table,2)
    # intially all neighbors are unassigned to a group
    groups .= 0
    # keeps track of which group is being constructed
    group = 0
    # while any bond is not assigned to a group
    while any(i->i==0,groups)
        # increment to next group
        group += 1
        # iterate over neighbor_table in lattice
        for neighbor in 1:nneighbors
            # if neighbor is not assigned to a group
            if groups[neighbor]==0
                # assign it to current group
                groups[neighbor] = group
                # iterate over previous neighbors
                for prev_neighbor in 1:neighbor-1
                    # if previous neighbor is a group member
                    if groups[prev_neighbor]==group
                        # if the previous neighbor overlaps with current neighbor
                        if ( neighbor_table[1,neighbor]==neighbor_table[1,prev_neighbor] ||
                             neighbor_table[2,neighbor]==neighbor_table[2,prev_neighbor] || 
                             neighbor_table[1,neighbor]==neighbor_table[2,prev_neighbor] || 
                             neighbor_table[2,neighbor]==neighbor_table[1,prev_neighbor] )
                            # remove current neighbor from group
                            groups[neighbor] = 0
                            break
                        end
                    end
                end
            end
        end
    end
    return group
end

end