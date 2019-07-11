module Geometries

using LinearAlgebra

export Geometry, monkhorst_pack_mesh, calc_cell_pos!, calc_cell_pos, calc_site_pos!, calc_site_pos

"""
Represents a specfied lattice geometry.
"""
struct Geometry

    "ndim: the number of dimensions that the geometry lives in."
    ndim::Int

    "norbits: the number of sites per unit cell."
    norbits::Int

    "lvecs:: (3 x 3) matrix where the columns give the lattice vectors."
    lvecs::Array{Float64,2}

    "rlvecs: (3 x 3) matrix where the columns give the reciprocal lattice vectors."
    rlvecs::Array{Float64,2}

    "bvecs: (3 x norbits) matrix where the columns give the basis vectors."
    bvecs::Array{Float64,2}
end

"""
    Geometry(ndim::Int, norbits::Int, lvecs::Matrix{Float64}, bvecs::Array{Float64})::Geometry

Constructor for Geometry type.
"""
function Geometry(ndim::Int, norbits::Int, lvecs::Matrix{Float64}, bvecs::Array{Float64})::Geometry

    # constructing matrix containing lattice vectors
    nrows = size(lvecs,1)
    ncols = size(lvecs,2)
    Lvecs = Matrix{Float64}(I,3,3) # intialized as identity matrix
    Lvecs[1:nrows,1:ncols] = lvecs

    # calculating reciprocal lattice vectors
    RLvecs = 2*π*inv(Lvecs)

    # constucting matrix to contain basis vectors
    nrows = size(bvecs,1)
    ncols = size(bvecs,2)
    Bvecs = zeros(Float64,3,ncols)
    Bvecs[1:nrows,:] = bvecs

    return Geometry(ndim,norbits,Lvecs,RLvecs,Bvecs)
end

function Geometry(ndim::Int, norbits::Int, lvecs::Matrix{Float64}, bvecs::Array{Array{Float64,1},1})::Geometry
    return Geometry( ndim, norbits, lvecs, hcat(bvecs...))
end

function Geometry(ndim::Int, norbits::Int, lvecs::Array{Array{Float64,1},1}, bvecs::Matrix{Float64})::Geometry
    return Geometry( ndim, norbits, hcat(lvecs...), bvecs)
end

function Geometry(ndim::Int, norbits::Int, lvecs::Array{Array{Float64,1},1}, bvecs::Array{Array{Float64,1},1})::Geometry
    return Geometry( ndim, norbits, hcat(lvecs...), hcat(bvecs...))
end


#############################################
## DEFINING METHODS THAT USE GEOMETRY TYPE ##
#############################################

"""
Calculates the position of a unit cell in a lattice.
"""
function calc_cell_pos!(pos::AbstractVector{Float64},geom::Geometry,l1::Int,l2::Int=0,l3::Int=0)

    @assert length(pos)==3
    lv1 = @view geom.lvecs[:,1] # first lattice vector
    lv2 = @view geom.lvecs[:,2] # second lattice vector
    lv3 = @view geom.lvecs[:,3] # third lattice vector
    @. pos[:] = l1*lv1 + l2*lv2 + l3*lv3 # calculating position
    return nothing
end

function calc_cell_pos(geom::Geometry,l1::Int,l2::Int=0,l3::Int=0)::Vector{Float64}

    pos = zeros(Float64,3)
    calc_cell_pos!(pos,geom,l1,l2,l3)
    return pos
end


"""
Calculates the position of a site in a lattice.
"""
function calc_site_pos!(pos::AbstractVector{Float64},geom::Geometry,orbit::Int,l1::Int,l2::Int=0,l3::Int=0)

    @assert orbit>0
    # calculating position of unit cell that site lives in
    calc_cell_pos!(pos,geom,l1,l2,l3)
    # adding basis vector for given orbital to position
    pos .+= @view geom.bvecs[:,orbit]
    return nothing
end

function calc_site_pos(geom::Geometry,orbit::Int,l1::Int,l2::Int=0,l3::Int=0)::Vector{Float64}

    pos = zeros(Float64,3)
    calc_site_pos!(pos,geom,orbit,l1,l2,l3)
    return pos
end


"""
    monkhorst_pack_mesh(geom::Geometry, L1::Int, L2::Int=1, L3::Int=1)::Matrix{Float64}

Returns a matrix where each column is a k-point vector in a Monkhort-Pack meshgrid over the full Brillouin Zone.
"""
function monkhorst_pack_mesh(geom::Geometry, L1::Int, L2::Int=1, L3::Int=1)::Matrix{Float64}

    kpoints = Array{Float64,2}(undef,3,L1*L2*L3)
    v1 = @view geom.rlvecs[:,1]
    v2 = @view geom.rlvecs[:,2]
    v3 = @view geom.rlvecs[:,3]
    i = 1
    for l3=0:L3-1
        for l2=0:L2-1
            for l1=0:L1-1
                @. kpoints[:,i] = (l1/L1)*v1 + (l2/L2)*v2 + (l3/L3)*v3
                i += 1
            end
        end
    end
    return kpoints
end

end