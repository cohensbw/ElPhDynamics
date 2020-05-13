module TightBindingFFTs

using LinearAlgebra
using FFTW
using UnsafeArrays

using ..Geometries: Geometry
using ..Lattices: Lattice, calc_neighbor_table

export TightBindingFFT, add_bond!, calc_basis!, r_to_k!, k_to_r!

"""
Type for representing bond/hopping rules in a tight-binding model.
"""
struct Bond{T<:Number}
    
    "Bond/hopping energy."
    t::Number
    
    "Orbital type bond starts from."
    orbit1::Int
    
    "Orbital type bond terminates at."
    orbit2::Int
    
    "Displacement in unit cells of bond."
    displacement::Vector{Int}
    
    function Bond(t::T,orbit1::Int,orbit2::Int,displacement::Vector{Int}) where {T<:Number}
        
        @assert length(displacement)==3
        return new{T}(t,orbit1,orbit2,displacement)
    end
end


mutable struct TightBindingFFT{T<:AbstractFloat}
    
    """
    Geometry object to represent unit cell.
    """
    geom::Geometry{T}
    
    """
    Lattice object to represent finite periodic lattice.
    """
    lattice::Lattice{T}
    
    """
    Vector of Bond types to represent hoppings in tight binding model.
    """
    bonds::Vector{Bond{Complex{T}}}
    
    """
    Eigenenergies of tight binding model.
    """
    λk::Array{T,4}
    
    """
    Unitary Bloch transformation.
    """
    Uk::Array{Complex{T},5}
    
    """
    Unitary Inverse Bloch transformation.
    """
    invUk::Array{Complex{T},5}
    
    """
    FFT plan to go from real to momentum space.
    """
    fftplan::FFTW.cFFTWPlan{Complex{T},-1,false,4}

    """
    Inverse FFT plan to go from momentum to real space.
    """
    ifftplan::AbstractFFTs.ScaledPlan{Complex{T},FFTW.cFFTWPlan{Complex{T},1,false,4},T}
    
    """
    Temporary storage vector.
    """
    z1::Vector{Complex{T}}
    
    """
    Temporary storage vector.
    """
    z2::Vector{Complex{T}}
    
    function TightBindingFFT(geom::Geometry{T},lattice::Lattice{T}) where {T<:AbstractFloat}
        
        # declaring empty array to contain bonds
        bonds = Vector{Bond{Complex{T}}}()
        
        # size of lattice
        L1 = lattice.L1
        L2 = lattice.L2
        L3 = lattice.L3
        norbits = lattice.norbits
        nsites  = lattice.nsites
        
        λk    = zeros(T,norbits,L1,L2,L3)
        Uk    = zeros(Complex{T},norbits,norbits,L1,L2,L3)
        invUk = zeros(Complex{T},norbits,norbits,L1,L2,L3)
        
        z1      = zeros(Complex{T},nsites)
        z2      = zeros(Complex{T},nsites)
        ztemp   = zeros(Complex{T},norbits,L1,L2,L3)
        fftplan  = plan_fft(ztemp, (2, 3, 4), flags=FFTW.PATIENT)
        ifftplan = plan_ifft(ztemp,(2, 3, 4), flags=FFTW.PATIENT)
        
        return new{T}(geom,lattice,bonds,λk,Uk,invUk,fftplan,ifftplan,z1,z2)
    end
end

"""
Add new type of hopping to tight binding model.
"""
function add_bond!(tbfft::TightBindingFFT{T},t,o1::Int,o2::Int,r::AbstractVector{Int}) where {T<:AbstractFloat}
    
    @assert length(r)==3
    bond = Bond(complex(t),o1,o2,r)
    push!(tbfft.bonds,bond)
    return nothing
end

"""
Calculate diagonal basis for tight binding model.
"""
function calc_basis!(op::TightBindingFFT{T}) where {T<:AbstractFloat}
    
        L1      = op.lattice.L1::Int
        L2      = op.lattice.L2::Int
        L3      = op.lattice.L3::Int
        ndim    = op.lattice.ndim::Int
        norbits = op.lattice.norbits::Int
        N       = op.lattice.nsites::Int
        λk      = op.λk
        Uk      = op.Uk
        invUk   = op.invUk
        
        # tight-binding matrix for single k-point that will need to be diagonalized
        Mk = zeros(Complex{T},(norbits,norbits))
        
        # iterate over k-points
        for k3 in 0:L3-1
            for k2 in 0:L2-1
                for k1 in 0:L1-1
                    # reset matrix
                    fill!(Mk,0.0)
                    # iterate over bonds
                    for bond in op.bonds
                        # getting bond energy
                        t  = bond.t
                        # getting which to orbits are connected
                        o1 = bond.orbit1
                        o2 = bond.orbit2 
                        # getting displacement vector
                        r1 = bond.displacement[1]
                        r2 = bond.displacement[2]
                        r3 = bond.displacement[3]
                        # if the bond corresponds to an on-site chemical potential,
                        # you need to cut it in half so the correct chemical potential is applied.
                        if o1==o2 && r1==r2==r3==0
                            t /= 2.0
                        end
                        # modify the corresponding matrix element of Mₖ
                        Δ = -t * exp(im*2*π*(k1*r1/L1+k2*r2/L2+k3*r3/L3))
                        Mk[o1,o2] += Δ
                        Mk[o2,o1] += conj(Δ)
                    end
                    # diagonalize the matrix
                    einfo = eigen(Mk)
                    # saving energies and bloch functions associated with each k-point.
                    λk[:,k1+1,k2+1,k3+1]      =   real.(einfo.values)
                    invUk[:,:,k1+1,k2+1,k3+1] =         einfo.vectors
                    Uk[:,:,k1+1,k2+1,k3+1]    = adjoint(einfo.vectors)
                end
            end
        end
    return nothing
end

"""
Transform to real to momentum space.
"""
function r_to_k!(zk::AbstractVector{Complex{T}},op::TightBindingFFT{T},zr::AbstractVector{Complex{T}}) where {T<:AbstractFloat}
    
    # getting info about size of lattice
    L1 = op.lattice.L1
    L2 = op.lattice.L2
    L3 = op.lattice.L3
    norbits = op.lattice.norbits
    nsites  = op.lattice.nsites
    
    Uk = op.Uk
    z1 = op.z1
    @uviews zr zk z1 Uk begin
        
        # reshape vector
        ar = reshape(zr,norbits,L1,L2,L3)
        ak = reshape(zk,norbits,L1,L2,L3)
        a1 = reshape(z1,norbits,L1,L2,L3)
        
        # forward fft from r to k
        mul!(a1,op.fftplan,ar)
        @. a1 /= sqrt(nsites)
        
        if norbits>1
            # iterate over k-points
            @fastmath @inbounds for k3 in 1:L3
                for k2 in 1:L2
                    for k1 in 1:L1
                        # apply block matrix unitary transform
                        v1k = @view a1[:,k1,k2,k3]
                        vkk = @view ak[:,k1,k2,k3]
                        uk  = @view Uk[:,:,k1,k2,k3]
                        mul!(vkk,uk,v1k)
                    end
                end
            end
        else
            copyto!(ak,a1)
        end
    end
    
    return nothing
end

function r_to_k!(zk::AbstractVector{Complex{T}},op::TightBindingFFT{T},zr::AbstractVector{T}) where {T<:AbstractFloat}
    
    copyto!(op.z2,zr)
    r_to_k!(zk,op,op.z2)
    return nothing
end

"""
Transform from momentum to real space.
"""
function k_to_r!(zr::AbstractVector{Complex{T}},op::TightBindingFFT{T},zk::AbstractVector{Complex{T}}) where {T<:AbstractFloat}
    
    # getting info about size of lattice
    L1 = op.lattice.L1
    L2 = op.lattice.L2
    L3 = op.lattice.L3
    norbits = op.lattice.norbits
    nsites  = op.lattice.nsites
    
    invUk = op.invUk
    z1    = op.z1
    @uviews zr zk z1 invUk begin
        
        # reshape vector
        ar = reshape(zr,norbits,L1,L2,L3)
        ak = reshape(zk,norbits,L1,L2,L3)
        a1 = reshape(z1,norbits,L1,L2,L3)
        
        if norbits>1
            # iterate over k-points
            @fastmath @inbounds for k3 in 1:L3
                for k2 in 1:L2
                    for k1 in 1:L1
                        # apply block matrix unitary transform
                        v1k    = @view a1[:,k1,k2,k3]
                        vkk    = @view ak[:,k1,k2,k3]
                        invuk  = @view invUk[:,:,k1,k2,k3]
                        mul!(v1k,invuk,vkk)
                    end
                end
            end
        else
            copyto!(a1,ak)
        end
        
        # inverse fft from k to r
        @. a1 *= sqrt(nsites)
        mul!(ar,op.ifftplan,a1)
    end
    
    return nothing
end

function k_to_r!(zr::AbstractVector{Complex{T}},op::TightBindingFFT{T},zk::AbstractVector{T}) where {T<:AbstractFloat}
    
    copyto!(op.z2,zk)
    k_to_r!(zr,op,op.z2)
    return nothing
end

end