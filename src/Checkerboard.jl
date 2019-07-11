module Checkerboard

using LinearAlgebra, SparseArrays
using Langevin.Lattices: sort_neighbors!

export checkerboard_groups!, checkerboard_matrix

"""
    checkerboard_matrix(neighbors::Matrix{Int},vals::Vector,groups::Vector{Int},Δτ::Float64)::SparseMatrixCSC    

Construct the checkerboard decomposition approximation for the matrix exp(-Δτ⋅K).
"""
function checkerboard_matrix(neighbors::Matrix{Int},vals::Vector,groups::Vector{Int},Δτ::Float64)::SparseMatrixCSC

    return checkerboard_matrix(neighbors,vals,groups,Δτ,maximum(groups),maximum(neighbors))
end

"""
    checkerboard_matrix(neighbors::Matrix{Int},vals::Vector,groups::Vector{Int},Δτ::Float64,ngroups::Int,nsites::Int)::SparseMatrixCSC    

Construct the checkerboard decomposition approximation for the matrix exp(-Δτ⋅K) where K is the electron kinetic energy matrix.
# Arguments
- `neighbors::Matrix{Int}`: 2xN matrix where each column contains a pair of neighboring sites and N is the number of neighbors.
- `vals::Vector`: Contains a value associated with each neighboring pair of sites.
- `groups::Vector{Int}`: Maps each neighboring pair of sites to a given Checkerboard Decomposition group.
- `Δτ::Float64`: Obvious.
- `ngroups::Int`: Number of Checkerboard Decompositions groups the neighbors have been broke into.
- `nsites::Int`: Number of sites in lattice.
"""
function checkerboard_matrix(neighbors::Matrix{Int},vals::Vector,groups::Vector{Int},Δτ::Float64,ngroups::Int,nsites::Int)::SparseMatrixCSC

    # intializing matrix so that it is equal to the checkerboard matrix for the first checkerboard group
    expK = checkerboard_group_matrix(neighbors,vals,groups,Δτ,1,nsites)
    # iterating over remaining checkerboard groups
    for group in 2:ngroups
        # updating matrix by multiplying with checkerboard matrix for current checkerboard gorup
        expK *= checkerboard_group_matrix(neighbors,vals,groups,Δτ,group,nsites)
    end
    return expK
end


"""
Construct a sparse matrix representation of exp(-Δτ⋅Kᵢ) for a single checkerboard group of bonds.
"""
function checkerboard_group_matrix(neighbors::Matrix{Int},vals::Vector,groups::Vector{Int},Δτ::Float64,group::Int,nsites::Int)::SparseMatrixCSC

    # determining data type in val array
    dtype = typeof(vals[1])
    # getting number of neighbors
    nneighbors = length(vals)
    # vectors for constructing sparse matrix
    rows = collect(1:nsites)
    cols = collect(1:nsites)
    elements = ones(dtype,nsites)
    # stores the pair of neighbors sites
    n1 = 0
    n2 = 0
    # stores cosh(-Δτ⋅tᵢⱼ) and cosh(-Δτ⋅tᵢⱼ)
    vcosh = 0.0
    vsinh = 0.0
    # iterating over neighbors
    for i in 1:nneighbors
        # determining if neighbors are the current color
        if groups[i]==group
            # calculating cosh and sinh values
            vcosh = cosh(-Δτ*vals[i])
            vsinh = sinh(-Δτ*vals[i])
            # getting neighboring sites
            n1 = neighbors[1,i]
            n2 = neighbors[2,i]
            # setting diagonal elements
            elements[n1] = vcosh
            elements[n2] = vcosh
            # setting first off diagonal element
            append!(rows,n1)
            append!(cols,n2)
            append!(elements,vsinh)
            # setting second off diagonal element
            append!(rows,n2)
            append!(cols,n1)
            append!(elements,conj(vsinh))
        end
    end
    return sparse(rows,cols,elements,nsites,nsites)
end


"""
    checkerboard_groups!(groups::Vector{Int},neighbors::Matrix{Int})::Int

Constructs the checkerboard decomposition for a given neighbor table.
Assumes the neighbor table has already been sorted using the sort_neighbors!
method from the Lattices module.
Returns the number of groups in the checkerboard decomposition.
"""
function checkerboard_groups!(groups::Vector{Int},neighbors::Matrix{Int})::Int

    # checking dimensions
    @assert size(neighbors,2)==length(groups)
    @assert size(neighbors,1)==2
    # getting the total number of neighbor pairs
    nneighbors = size(neighbors,2)
    # intially not neighbors are assigned to a group
    groups .= 0
    # keeps track of which group is being constructed
    group = 0
    # while any bond is not assigned to a group
    while any(i->i==0,groups)
        # increment to next group
        group += 1
        # iterate over neighbors in lattice
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
                        if ( neighbors[1,neighbor]==neighbors[1,prev_neighbor] ||
                             neighbors[2,neighbor]==neighbors[2,prev_neighbor] || 
                             neighbors[1,neighbor]==neighbors[2,prev_neighbor] || 
                             neighbors[2,neighbor]==neighbors[1,prev_neighbor] )
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