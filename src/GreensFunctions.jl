module GreensFunctions

using Statistics
using Random
using LinearAlgebra

using ..HolsteinModels: HolsteinModel, mulMᵀ!
using ..Utilities: get_index, get_site, get_τ

using ..BlockPreconditioners: setup!
# using ..SingleSitePreconditioners: setup!
# using ..DiagonalPreconditioners: setup!

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
function update!(Gr::EstimateGreensFunction{T1}, holstein::HolsteinModel{T1,T2}, preconditioner=I) where {T1<:AbstractFloat,T2<:Number}

    # initialize random vector
    randn!(Gr.g)

    # solve linear system to get M⁻¹⋅g
    iters = 0
    fill!(Gr.M⁻¹g,0.0)
    setup!(preconditioner) # setup block preconditioner
    if holstein.mul_by_M
        # solve M⋅x=g ==> x=M⁻¹⋅g
        iters = ldiv!(Gr.M⁻¹g,holstein,Gr.g,preconditioner)
    else
        # solve MᵀM⋅x=Mᵀg ==> x=[MᵀM]⁻¹⋅Mᵀg=M⁻¹⋅g
        mulMᵀ!(holstein.Mᵀg,holstein,Gr.g)
        iters = ldiv!(Gr.M⁻¹g,holstein,holstein.Mᵀg,preconditioner)
    end

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