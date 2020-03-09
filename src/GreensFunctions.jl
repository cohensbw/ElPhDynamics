module GreensFunctions

using IterativeSolvers
using Statistics
using Random
import LinearAlgebra: mul!, ldiv!

using ..HolsteinModels: HolsteinModel
using ..Utilities: get_index, get_site, get_τ

export EstimateGreensFunction
export update!, estimate

###########################################
## CODE FOR CALCULATING GREEN'S FUNCTION ##
###########################################

"""
A type for facilitating the stochastic estimation of the the Green's function Gr = ⟨cᵢ(τ)⋅cᵀⱼ(0)⟩.
"""
struct EstimateGreensFunction{T<:Number}

    """
    Total number of indices (sites) in D+1 dimensional lattice.
    """
    βN::Int

    """
    Number of sites in lattice.
    """
    N::Int

    """
    Length of imaginary time axis.
    """
    β::Int

    """
    Random vector of length βN.
    """
    g::Vector{T}

    """
    Solution to Linear System M⋅v=g ==> v = M⁻¹⋅g.
    """
    M⁻¹g::Vector{T}

    """
    Constructor for GreensFunction.
    """
    function EstimateGreensFunction(holstein::HolsteinModel{T1,T2}, is_complex::Bool=false) where {T1<:AbstractFloat,T2<:Number}

        βN  = holstein.nindices
        N   = holstein.nsites
        β   = holstein.Lτ

        g    = zeros(T1,βN)
        M⁻¹g = zeros(T1,βN)

        if is_complex
            new{Complex{T1}}(βN,N,β,g,M⁻¹g)
        else
            new{T1}(βN,N,β,g,M⁻¹g)
        end
    end
end

"""
Updates the estimate of the Green's Function based on current phonon field configuration.
"""
function update!(Gr::EstimateGreensFunction{T1}, holstein::HolsteinModel{T1,T2}, preconditioner=Identity()) where {T1<:AbstractFloat,T2<:Number}

    # initialize random vector
    # rand!(Gr.g,-1:2:1)
    randn!(Gr.g)

    # solve linear system to get M⁻¹⋅g
    iters = ldiv!(Gr.M⁻¹g,holstein,Gr.g,preconditioner)

    return nothing
end

"""
Returns ⟨cᵢ(τ₂)⋅c⁺ⱼ(τ₁)⟩ where 1⩽τ₁⩽β, 1⩽τ₂⩽β and 1⩽(i,j)⩽N.
Note that no time ordering is considered here.
"""
function estimate(Gr::EstimateGreensFunction{T},i::Int,j::Int,τ₂::Int,τ₁::Int)::T where {T<:Number}
    
    m = get_index(τ₁,j,Gr.β)
    n = get_index(τ₂,i,Gr.β)
    return real( conj(Gr.g[n]) * Gr.M⁻¹g[m] )
end

end