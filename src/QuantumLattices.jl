module QuantumLattices

using Langevin.Geometries: Geometry
using Langevin.Lattices: Lattice, translationally_equivalent_sets

export QuantumLattice

"""
Represents the D+1 dimensional lattice that is generated after applying the Suzuki-Trotter
approximation to a D dimensional quantum problem: the temperature discretized,
turning it into an additional imaginary time τ axis.
This type helps map betwen the indices in the D+1 dimensional lattice and the
sites in the physical lattice and imaginary time slices τ: index ⇋ (site,τ)
"""
struct QuantumLattice{T<:AbstractFloat}

    "inverse temperature"
    β::T

    "imaginary time step"
    Δτ::T

    "length of imaginary time axis"
    Lτ::Int

    "Number of sites in physical D dimensional lattice"
    nsites::Int

    "Number of indices in D+1 dimensional lattice."
    nindices::Int

    "maps index_to_τ[index]=τ"
    index_to_τ::Vector{Int}

    "maps index_to_site[index]=site"
    index_to_site::Vector{Int}

    "maps
     to_index[site,τ]=index"
    to_index::Matrix{Int}

    # this array will be very useful when making measurements that average over translational symmentry.
    # use datatype UInt16 to save memory.
    "stores sets of translationally equivalent pairs of sites in lattice."
    trans_equiv_sets::Array{UInt16,7}

    #######################
    ## INNER CONSTRUCTOR ##
    #######################

    """
    Constructor for QuantumLattice type.
    """
    function QuantumLattice(lattice::Lattice,β::T,Δτ::T) where {T<:AbstractFloat}

        # calculate length of imaginary time axis
        Lτ=Int(β/Δτ)

        # number of sites in physical lattice
        nsites = lattice.nsites

        # number of indices in D+1 dimensional lattice
        nindices = Lτ*nsites

        # constructing arrays for mapping between [site,τ]⇆[index]
        index_to_τ    = zeros(Int,nindices)
        index_to_site = zeros(Int,nindices)
        to_index      = zeros(Int,nsites,Lτ)
        τ = 0
        site = 0
        index = 1
        for τ in 1:Lτ
            for site in 1:nsites
                index_to_τ[index]    = τ
                index_to_site[index] = site
                to_index[site,τ]     = index
                index += 1
            end
        end

        # constructing translationally equivalent sets of sites
        trans_equiv_sets = Array{UInt16,7}(translationally_equivalent_sets(lattice))

        new{T}(β,Δτ,Lτ,nsites,nindices,index_to_τ,index_to_site,to_index,trans_equiv_sets)
    end

end

# adding pretty print functionality
function Base.show(io::IO, qlattice::QuantumLattice)

    printstyled("QuantumLattice{",typeof(qlattice.β),"}\n";bold=true)
    print('\n')
    println("•β = ",qlattice.β)
    println("•Δτ = ",qlattice.Δτ)
    println("•Lτ = ",qlattice.Lτ)
    println("•nsites = ",qlattice.nsites)
    println("•nindices = ",qlattice.nindices)
    print('\n')
    println("•index_to_τ: ", typeof(qlattice.index_to_τ),size(qlattice.index_to_τ))
    println("•index_to_site: ", typeof(qlattice.index_to_site),size(qlattice.index_to_site))
    println("•to_index: ", typeof(qlattice.to_index),size(qlattice.to_index))
    println("•trans_equiv_sets: ", typeof(qlattice.trans_equiv_sets),size(qlattice.trans_equiv_sets))

end

end