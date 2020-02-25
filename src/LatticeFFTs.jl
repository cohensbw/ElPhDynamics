module LatticeFFTs

using UnsafeArrays
using FFTW
using LinearAlgebra
import LinearAlgebra: mul!

using ..Geometries: Geometry
using ..Lattices:   Lattice

########################
## DEFINING BOND TYPE ##
########################

"""
Type for representing bond/hopping rules in a tight-binding model.
"""
struct Bond{T<:Number}
    
    "Bond/hopping energy."
    t::T
    
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

##############################
## DEINING LATTICE FFT TYPE ##
##############################

"""
Type for representing FFT + (unitary transformation) needed to transform the
diagonal basis of translationally invariant non-interacting tight-binding model
for arbitrary lattice geometries.
"""
struct LatticeFFT{T<:AbstractFloat}
    
    "Number of spatial dimenions."
    ndim::Int
    
    "Number of orbits per unit cell."
    norbits::Int
    
    "Total number of sites in lattice."
    N::Int
    
    "Size of lattice in unit cell in direction of first lattice vector."
    L1::Int
    
    "Size of lattice in unit cell in direction of second lattice vector."
    L2::Int
    
    "Size of lattice in unit cell in direction of third lattice vector."
    L3::Int
    
    "Length of imaginary time axis."
    Lτ::Int
    
    "Imaginary time discretization step."
    Δτ::T
    
    "Eigenvalues/energies associated with each k-point."
    λk::Array{T,4}
    
    "Exponentiated eigenvalues/energies associated with each k-point."
    expnΔτλk::Array{T,4}
    
    "Unitary transformation associated with each k-point."
    Uk::Array{Complex{T},5}
    
    "Inverse unitary transformation associated with each k-point."
    invUk::Array{Complex{T},5}
    
    "Temporary momentum space storage vector."
    vk::Array{Complex{T},5}
    
    "Temporary real space storage vector."
    vr::Array{Complex{T},5}
    
    "Temporary storage vector for dealing with a real-valued (not complex) input vector"
    vin::Array{Complex{T},5}
    
    "Temporary storage vector for dealing with a real-valued (not complex) output vector"
    vout::Array{Complex{T},5}
    
    "FFT plan."
    fftplan::FFTW.cFFTWPlan{Complex{T},-1,false,5}

    "Inverse FFT plan."
    ifftplan::AbstractFFTs.ScaledPlan{Complex{T},FFTW.cFFTWPlan{Complex{T},1,false,5},T}
    
    function LatticeFFT(lattice::Lattice{T}, bonds::Vector{Bond{Tb}}, Lτ::Int, Δτ::T) where {T<:AbstractFloat, Tb<:Number}
        
        L1      = lattice.L1::Int
        L2      = lattice.L2::Int
        L3      = lattice.L3::Int
        ndim    = lattice.ndim::Int
        norbits = lattice.norbits::Int
        N       = lattice.nsites::Int
        
        λk    = zeros(Complex{T},L1,L2,L3,norbits)
        Uk    = zeros(Complex{T},L1,L2,L3,norbits,norbits)
        invUk = zeros(Complex{T},L1,L2,L3,norbits,norbits)
        
        # tight-binding matrix for single k-point that will need to be diagonalized
        Mk = zeros(Complex{T},(norbits,norbits))
        
        # iterate over k-points
        for k3 in 0:L3-1
            for k2 in 0:L2-1
                for k1 in 0:L1-1
                    # reset matrix
                    Mk .= 0.0
                    # iterate over bonds
                    for bond in bonds
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
                        # you need to cut t in half so the correct chemical potential is applied.
                        if orbit1==orbit2 && iszero(displacement)
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
                    λk[k1+1,k2+1,k3+1,:]      =   real.(einfo.values)
                    invUk[k1+1,k2+1,k3+1,:,:] =         einfo.vectors
                    Uk[k1+1,k2+1,k3+1,:,:]    = adjoint(einfo.vectors)
                end
            end
        end
        
        # declaring temporary storage vectors
        vk   = zeros(Complex{T},L1,L2,L3,norbits,Lτ)
        vr   = zeros(Complex{T},L1,L2,L3,norbits,Lτ)
        vin  = zeros(Complex{T},L1,L2,L3,norbits,Lτ)
        vout = zeros(Complex{T},L1,L2,L3,norbits,Lτ)
        
        # plan fft              (L1,L2,L3)
        fftplan  = plan_fft(vk, (1, 2, 3), flags=FFTW.PATIENT)
        ifftplan = plan_ifft(vr,(1, 2, 3), flags=FFTW.PATIENT)
        
        # array to represent diagonal matrix exp(-Δτ⋅λₖ)
        expnΔτλk = exp.(-Δτ*λk)
        
        return new{T}(ndim,norbits,N,L1,L2,L3,Lτ,Δτ,λk,expnΔτλk,Uk,invUk,vk,vr,vin,vout,fftplan,ifftplan)
    end
end

###################################
## DEFINE TRANSFORMATION METHODS ##
###################################

"""
Apply the transformation vₖ=[U⋅F]⋅vᵣ where U is the unitary transformation associated with the
bloch states, and F is the FFT operator.
"""
function r_to_k!(vout::AbstractArray{Complex{T},5}, latticefft::LatticeFFT{T}, vin::AbstractArray{Complex{T},5}) where {T<:AbstractFloat}
    
    norbits = latticefft.norbits::Int
    L1 = latticefft.L1::Int
    L2 = latticefft.L2::Int
    L3 = latticefft.L3::Int
    Lτ = latticefft.Lτ::Int
    
    # calculate standard fourier transform first
    vk = latticefft.vk::Array{Complex{T},5}
    mul!(vk,latticefft.fftplan,vin)
    
    # reset values in output vector
    vout .= 0.0

    if norbits>1
        # apply unitary transformation assoicated with bloch states
        Uk = latticefft.Uk
        @uviews Uk vk vout begin
            @inbounds @fastmath for τ in 1:Lτ
                for o2 in 1:norbits
                    vko2 = @view vk[:,:,:,o2,τ]
                    for o1 in 1:norbits
                        Uko1o2 = @view Uk[:,:,:,o1,o2]
                        vouto1 = @view vout[:,:,:,o1,τ]
                        @. vouto1 += Uko1o2 * vko2
                    end
                end
            end
        end
    else
        copyto!(vout,vk)
    end

    return nothing
end

function r_to_k!(vout::AbstractVector,latticefft::LatticeFFT{T},vin::AbstractVector) where {T<:AbstractFloat}

    uin  = latticefft.vin::Array{Complex{T},5}
    uout = latticefft.vout::Array{Complex{T},5}
    reordered_copy!(uin,vin,latticefft)
    r_to_k!(uout,latticefft,uin)
    reordered_copy!(vout,uout,latticefft)
    return nothing
end

function r_to_k!(v::AbstractVector{Complex{T}},latticefft::LatticeFFT{T}) where {T<:AbstractFloat}

    r_to_k!(v,latticefft,v)
end


"""
Apply the inverse transformation vᵣ=[F⁺⋅U⁺]⋅vₖ where U is the unitary transformation
associated with the bloch states, and F is the FFT operator.
"""
function k_to_r!(vout::AbstractArray{Complex{T},5}, latticefft::LatticeFFT{T}, vin::AbstractArray{Complex{T},5}) where {T<:AbstractFloat}
    
    norbits = latticefft.norbits::Int
    L1 = latticefft.L1::Int
    L2 = latticefft.L2::Int
    L3 = latticefft.L3::Int
    Lτ = latticefft.Lτ::Int
    
    # empty temporary array
    vk = latticefft.vk::Array{Complex{T},5}
    vk .= 0.0

    if norbits>1
        # apply unitary transformation assoicated with bloch states
        invUk = latticefft.invUk
        @uviews invUk vk vin begin
            @inbounds @fastmath for τ in 1:Lτ
                for o2 in 1:norbits
                    vino2 = @view vin[:,:,:,o2,τ]
                    for o1 in 1:norbits
                        invUko1o2 = @view invUk[:,:,:,o1,o2]
                        vko1 = @view vk[:,:,:,o1,τ]
                        @. vko1 += invUko1o2 * vino2
                    end
                end
            end
        end
    else
        copyto!(vk,vin)
    end
    
    # perform the inverse fourier transform
    mul!(vout,latticefft.ifftplan,vk)
    
    return nothing
end

function k_to_r!(vout::AbstractVector,latticefft::LatticeFFT{T},vin::AbstractVector) where {T<:AbstractFloat}

    uin  = latticefft.vin::Array{Complex{T},5}
    uout = latticefft.vout::Array{Complex{T},5}
    reordered_copy!(uin,vin,latticefft)
    k_to_r!(uout,latticefft,uin)
    reordered_copy!(vout,uout,latticefft)
    return nothing
end

function k_to_r!(v::AbstractVector{Complex{T}},latticefft::LatticeFFT{T}) where {T<:AbstractFloat}

    k_to_r!(v,latticefft,v)
end


"""
Performs the full multiplication by the exponentiated Kinetic energy matrix.
Does this via following relationship: vₒᵤₜ = exp(-Δτ⋅K)⋅vᵢₙ = [F⁻¹⋅U⁻¹⋅exp(-Δτ⋅λₖ)⋅U⋅F]⋅vᵢₙ.
In the above expression U is a unitary transformation, F is the FFT operator, and
exp(-Δτ⋅λₖ) is a diagonal matrix.
"""
function mul!(vout::AbstractArray{Complex{T},5},latticefft::LatticeFFT{T},vin::AbstractArray{Complex{T},5}) where {T<:AbstractFloat}
    
    # temporary storage
    vr = latticefft.vr

    # forward transformation: vₖ=U⋅F⋅vᵣ
    r_to_k!(vr,latticefft,vin)
    
    # multiply by diagonal matrix: vₖ=exp(-Δτ⋅λₖ)⋅vₖ
    multiply!(vr,latticefft)
    
    # inverse transformation: vᵣ=F⁻¹⋅U⁻¹⋅vₖ
    k_to_r!(vout,latticefft,vr)
    
    return nothing
end

function mul!(vout::AbstractVector{T1},latticefft::LatticeFFT{T2},vin::AbstractVector{T1}) where {T1<:Number,T2<:AbstractFloat}

    uin  = latticefft.vin::Array{Complex{T2},5}
    uout = latticefft.vout::Array{Complex{T2},5}
    reordered_copy!(uin,vin,latticefft)
    mul!(uout,latticefft,uin)
    reordered_copy!(vout,uout,latticefft)
    return nothing
end

function mul!(v::AbstractVector{Complex{T}},latticefft::LatticeFFT{T}) where {T<:AbstractFloat}

    mul!(v,latticefft,v)
end

#######################
## PRIVATE FUNCTIONS ##
#######################

"""
Calculate the matrix-vector produce vₖ=exp(-Δτ⋅λₖ)⋅vₖ. 
"""
function multiply!(vk::AbstractArray{T1,5}, latticefft::LatticeFFT{T2}) where {T1<:Number,T2<:AbstractFloat}

    Lτ = latticefft.Lτ::Int
    expnΔτλk = latticefft.expnΔτλk::AbstractArray{T2,4}
    @uviews vk begin
        @inbounds for τ in 1:Lτ
            @. @views vk[:,:,:,:,τ] *= expnΔτλk
        end
    end
    return nothing
end


"""
Takes a vector `vin` where the assumed ordering is `[x₁(1),...,x₁(L),...,xₙ(1),...,x₁ₙ(L)]`
and copies it contents into the 5-index array `vout`, where the indices correspond to `[L1,L2,L3,norbits,Lτ]`.
"""
function reordered_copy!(vout::AbstractArray{Complex{T1},5},vin::AbstractVector{T2},latticefft::LatticeFFT{T1}) where {T1<:AbstractFloat,T2<:Number}

    norbits = latticefft.norbits::Int
    L1 = latticefft.L1::Int
    L2 = latticefft.L2::Int
    L3 = latticefft.L3::Int
    Lτ = latticefft.Lτ::Int
    @uviews vout vin begin
        uin = reshape(vin,Lτ,norbits,L1,L2,L3)
        for τ in 1:Lτ
            for o in 1:norbits
                @. @views vout[:,:,:,o,τ] = uin[τ,o,:,:,:]
            end
        end
    end
    return nothing
end

"""
Takes the 5-index array `vin`, where the indices correspond to `[L1,L2,L3,norbits,Lτ]`, and copies
it into an the the vector `vout` where the assumed ordering is `[x₁(1),...,x₁(L),...,xₙ(1),...,x₁ₙ(L)]`.
"""
function reordered_copy!(vout::AbstractVector{T},vin::AbstractArray{Complex{T},5},latticefft::LatticeFFT{T}) where {T<:AbstractFloat}

    norbits = latticefft.norbits::Int
    L1 = latticefft.L1::Int
    L2 = latticefft.L2::Int
    L3 = latticefft.L3::Int
    Lτ = latticefft.Lτ::Int
    @uviews vout vin begin
        uout = reshape(vout,Lτ,norbits,L1,L2,L3)
        @inbounds @fastmath for τ in 1:Lτ
            for o in 1:norbits
                @. @views uout[τ,o,:,:,:] = real(vin[:,:,:,o,τ])
            end
        end
    end
    return nothing
end

function reordered_copy!(vout::AbstractVector{Complex{T}},vin::AbstractArray{Complex{T},5},latticefft::LatticeFFT{T}) where {T<:AbstractFloat}

    norbits = latticefft.norbits::Int
    L1 = latticefft.L1::Int
    L2 = latticefft.L2::Int
    L3 = latticefft.L3::Int
    Lτ = latticefft.Lτ::Int
    @uviews vout vin begin
        uout = reshape(vout,Lτ,norbits,L1,L2,L3)
        @inbounds @fastmath for τ in 1:Lτ
            for o in 1:norbits
                @. @views uout[τ,o,:,:,:] = vin[:,:,:,o,τ]
            end
        end
    end
    return nothing
end

end