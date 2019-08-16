module QuantumLattices

using Langevin.Geometries: Geometry
using Langevin.Lattices: Lattice, translationally_equivalent_sets

export QuantumLattice, view_by_site, view_by_τ

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

    "maps index in vector g of length NLτ to a time slice index_to_τ[index]=τ"
    index_to_τ::Vector{Int}

    "maps index in vector g of length NLτ to a site in the lattice index_to_site[index]=site"
    index_to_site::Vector{Int}

    "maps a site in the lattice and a time slice τ to_index[site,τ]=index"
    to_index::Matrix{Int}

    #######################
    ## INNER CONSTRUCTOR ##
    #######################

    """
    Constructor for QuantumLattice type.
    """
    function QuantumLattice(lattice::Lattice,β::T,Δτ::T) where {T<:AbstractFloat}

        # calculate length of imaginary time axis
        Lτ=round(Int,β/Δτ)

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
        index = 0
        for τ in 1:Lτ
            for site in 1:nsites
                index += 1
                index_to_τ[index]    = τ
                index_to_site[index] = site
                to_index[site,τ]     = index
            end
        end


        new{T}(β,Δτ,Lτ,nsites,nindices,index_to_τ,index_to_site,to_index)
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

end


"""
Give a vector of length `nindices=nsites⋅Lτ``, this function returns a view into
that vector for all time slices `τ` for a given `site` in the lattice.
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

end