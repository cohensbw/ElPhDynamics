module GreensFunctions

using IterativeSolvers
using Statistics
using Random

using Langevin.HolsteinModels: HolsteinModel, mulMᵀ!
using Langevin.HolsteinModels: get_index, get_site, get_τ

export EstimateGreensFunction
export update!, estimate, estimate_time_ordered

###########################################
## CODE FOR CALCULATING GREEN'S FUNCTION ##
###########################################

"""
A type for facilitating the stochastic estimation of the the Green's function Gr = ⟨cᵢ(τ)⋅cᵀⱼ(0)⟩.
"""
struct EstimateGreensFunction{T<:AbstractFloat}

    """
    Total number of indices (sites) in D+1 dimensional lattice.
    """
    NLτ::Int

    """
    Number of sites in lattice.
    """
    N::Int

    """
    Number of unit cells in direction of first lattice vector.
    """
    L1

    """
    Number of unit cells in direction of second lattice vector.
    """
    L2

    """
    Number of unit cells in direction of third lattice vector.
    """
    L3

    """
    Length of imaginary time axis.
    """
    Lτ::Int

    """
    Random vector of length NLτ.
    """
    g::Vector{T}

    """
    Represents matrix-vector product Mᵀ⋅g.
    """
    Mᵀg::Vector{T}

    """
    Solution to Linear System M⋅v=g ==> v = M⁻¹⋅g.
    """
    M⁻¹g::Vector{T}

    """
    Constructor for GreensFunction.
    """
    function EstimateGreensFunction(holstein::HolsteinModel{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

        NLτ = holstein.nindices
        N   = holstein.nsites
        Lτ  = holstein.Lτ
        L1  = holstein.lattice.L1
        L2  = holstein.lattice.L2
        L3  = holstein.lattice.L3

        g    = zeros(T1,NLτ)
        Mᵀg  = zeros(T1,NLτ)
        M⁻¹g = zeros(T1,NLτ)

        new{T1}(NLτ,N,L1,L2,L3,Lτ,g,Mᵀg,M⁻¹g)
    end
end

"""
Updates the estimate of the Green's Function based on current phonon field configuration.
"""
function update!(Gr::EstimateGreensFunction{T1}, holstein::HolsteinModel{T1,T2}, tol::T1=1e-4) where {T1<:AbstractFloat,T2<:Number}

    # initialize random vector
    rand!(Gr.g,-1:2:1)

    # calculate Mᵀ⋅g
    mulMᵀ!(Gr.Mᵀg,holstein,Gr.g)

    # initialize to zero
    Gr.M⁻¹g .= 0.0

    # solve linear system to get M⁻¹⋅g
    cg!( Gr.M⁻¹g , holstein , Gr.Mᵀg , tol=tol , log=false , statevars=holstein.cg_state_vars , initially_zero=true )

    return nothing
end

"""
Returns ⟨cᵢ(τ₂)⋅c⁺ⱼ(τ₁)⟩ where 1⩽τ₁⩽Lτ, 1⩽τ₂⩽Lτ and 1⩽(i,j)⩽N.
Note that no time ordering is considered here.
"""
function estimate(Gr::EstimateGreensFunction{T},i::Int,j::Int,τ₂::Int,τ₁::Int)::T where {T<:AbstractFloat}
    
    l = get_index(τ₂,i,Gr.Lτ)
    r = get_index(τ₁,j,Gr.Lτ)
    return Gr.g[l] * Gr.M⁻¹g[r]
end

"""
Returns stochastic estmate of time-ordered Green's function
⟨T⋅cᵢ(τ₁+τ)⋅c⁺ⱼ(τ₁)⟩=⟨cᵢ(τ)⋅c⁺ⱼ(0)⟩=Gᵢⱼ(τ) where 1⩽τ₁⩽Lτ, 0⩽τ<Lτ and 1⩽(i,j)⩽N.
"""
function estimate_time_ordered(Gr::EstimateGreensFunction{T},i::Int,j::Int,τ::Int,τ₁::Int)::T where {T<:AbstractFloat}
    
    # getting τ₂, accounting for boundary conditions
    τ₂ = (τ₁+τ-1)%Gr.Lτ+1
    # getting green's function estimate
    gr = estimate(Gr,i,j,τ₂,τ₁)
    # correcting for antiperiodic boundary conditions if needed
    if τ₂<τ₁
        gr *= -1.0
    end
    return gr
end

end