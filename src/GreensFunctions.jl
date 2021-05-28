module GreensFunctions

using Statistics
using Random
using LinearAlgebra
using FFTW
using Parameters

using ..Models: AbstractModel, mulMᵀ!, mulM!, update_model!
using ..Utilities: get_index, get_site, get_τ, reshaped

using ..KPMPreconditioners

export EstimateGreensFunction, update!, measure_GΔ0, measure_GΔ0_G0Δ, measure_GΔ0_GΔ0, measure_GΔΔ_G00, estimate

###########################################
## CODE FOR CALCULATING GREEN'S FUNCTION ##
###########################################

"""
A type for facilitating the stochastic estimation needed to make measurements
"""
mutable struct EstimateGreensFunction{T<:AbstractFloat,Tfft<:AbstractFFTs.Plan,Tifft<:AbstractFFTs.Plan}

    """
    Number of random vectors to be use.
    """
    nᵥ::Int

    """
    First in pair of random vectors to convolve together.
    """
    n₁::Int

    """
    Second in pair of random vectors to convolve together.
    """
    n₂::Int
    
    """
    Number of degrees of freedom.
    """
    NL::Int

    """
    Length of imaginary time axis.
    """
    L::Int

    """
    Number of sites in physical lattice.
    """
    N::Int

    """
    Size of lattice in direction of third lattice vector.
    """
    L₃::Int

    """
    Size of lattice in direction of second lattice vector.
    """
    L₂::Int

    """
    Size of lattice in direction of first lattice vector.
    """
    L₁::Int

    """
    Number of sites/orbitals per unit cell.
    """
    nₛ::Int

    """
    All the random vectors.
    """
    R::Matrix{T}

    """
    All solutions to linear systems associated with random vectors.
    """
    M⁻¹R::Matrix{T}

    """
    First random vector of length NL.
    """
    r₁::Vector{T}

    """
    Second random Vector of length NL.
    """
    r₂::Vector{T}

    """
    M⁻¹⋅r₁ of length NL.
    """
    M⁻¹r₁::Vector{T}

    """
    M⁻¹⋅r₁ of length NL.
    """
    M⁻¹r₂::Vector{T}

    """
    Time ordered Green's function G[Δ,0]=Gᵣ(τ)=⟨T⋅cᵢ₊ᵣ(τ)⋅cᵀᵢ(0)⟩ where Δ=(τ,r)
    """
    GΔ0::Array{Complex{T},6}

    """
    G[Δ,Δ]⋅G[0,0]=⟨T⋅cᵢ₊ᵣ(τ)⋅cᵀᵢ₊ᵣ(τ)⟩⋅⟨T⋅cᵢ(0)⋅cᵀᵢ(0)⟩ where Δ=(τ,r)
    """
    GΔΔ_G00::Array{Complex{T},6}

    """
    G[Δ,0]⋅G[Δ,0]=⟨T⋅cᵢ₊ᵣ(τ)⋅cᵀᵢ(0)⟩⋅⟨T⋅cᵢ₊ᵣ(τ)⋅cᵀᵢ(0)⟩ where Δ=(τ,r)
    """
    GΔ0_GΔ0::Array{Complex{T},6}

    """
    G[Δ,0]⋅G[0,Δ] =⟨T⋅cᵢ₊ᵣ(τ)⋅cᵀᵢ(0)⟩⋅⟨T⋅cᵢ(0)⋅cᵀᵢ₊ᵣ(τ)⟩  where Δ=(τ,r)
    G[Δ,0]⋅G[-Δ,0]=⟨T⋅cᵢ₊ᵣ(τ)⋅cᵀᵢ(0)⟩⋅⟨T⋅cᵢ₋ᵣ(-τ)⋅cᵀᵢ(0)⟩ where Δ=(τ,r)
    """
    GΔ0_G0Δ::Array{Complex{T},6}

    """
    Forward FFT plan.
    """
    pfft::Tfft

    """
    Inverse FFT plan.
    """
    pifft::Tifft

    """
    Array of dimension [2L,n,L₁,L₂,L₃] used for performing convolution.
    """
    a::Array{Complex{T},5}

    """
    Array of dimension [2L,n,L₁,L₂,L₃] used for preforming convolution.
    """
    b::Array{Complex{T},5}

    """
    Array of dimension [2L,n,n,L₁,L₂,L₃] used for preforming convolution.
    """
    ab′::Array{Complex{T},6}

    """
    Array of dimension [2L,n,n,L₁,L₂,L₃] used for preforming convolution.
    """
    ab″::Array{Complex{T},6}

    """
    Temporary storage array of dimension [2L,n,L₁,L₂,L₃].
    """
    z::Array{Complex{T},5}

    """
    Constructor for GreensFunction.
    """
    function EstimateGreensFunction(model::AbstractModel{T},nᵥ::Int=2) where {T<:AbstractFloat}

        nᵥ      = max(2,nᵥ)
        NL      = model.Ndim
        N       = model.Nsites
        L       = model.Lτ
        L₃      = model.lattice.L3
        L₂      = model.lattice.L2
        L₁      = model.lattice.L1
        nₛ      = model.lattice.unit_cell.norbits
        R       = zeros(T,NL,nᵥ)
        M⁻¹R    = zeros(T,NL,nᵥ)
        r₁      = zeros(T,NL)
        r₂      = zeros(T,NL)
        M⁻¹r₁   = zeros(T,NL)
        M⁻¹r₂   = zeros(T,NL)
        GΔ0     = zeros(Complex{T},2L,nₛ,nₛ,L₁,L₂,L₃)
        GΔΔ_G00 = zeros(Complex{T},2L,nₛ,nₛ,L₁,L₂,L₃)
        GΔ0_GΔ0 = zeros(Complex{T},2L,nₛ,nₛ,L₁,L₂,L₃)
        GΔ0_G0Δ = zeros(Complex{T},2L,nₛ,nₛ,L₁,L₂,L₃)
        a       = zeros(Complex{T},2L,nₛ,L₁,L₂,L₃)
        b       = zeros(Complex{T},2L,nₛ,L₁,L₂,L₃)
        ab′     = zeros(Complex{T},2L,nₛ,nₛ,L₁,L₂,L₃)
        ab″     = zeros(Complex{T},2L,nₛ,nₛ,L₁,L₂,L₃)
        z       = zeros(Complex{T},2L,nₛ,L₁,L₂,L₃)
        pfft    = plan_fft(a, (1,3,4,5), flags=FFTW.PATIENT)
        pifft   = plan_ifft(ab′, (1,4,5,6), flags=FFTW.PATIENT)
        Tfft    = typeof(pfft)
        Tifft   = typeof(pifft)

        return new{T,Tfft,Tifft}(nᵥ,1,2,NL,L,N,L₃,L₂,L₁,nₛ,R,M⁻¹R,r₁,r₂,M⁻¹r₁,M⁻¹r₂,GΔ0,GΔΔ_G00,GΔ0_GΔ0,GΔ0_G0Δ,pfft,pifft,a,b,ab′,ab″,z)
    end
end

"""
Update Green's functions with new random vectors.
"""
function update!(estimator::EstimateGreensFunction, model::T, preconditioner=I) where {T<:AbstractModel}

    @unpack r₁, M⁻¹r₁, R, M⁻¹R = estimator
    
    # update preconditioner
    KPMPreconditioners.setup!(preconditioner)

    # iterate over number or random vectors
    for i in 1:estimator.nᵥ

        # initialize random vector with new random number
        randn!(r₁)
        fill!(M⁻¹r₁,0.0)

        # solve linear system
        if model.mul_by_M
            model.transposed = false
            # solve M⋅x=r₁ ==> x=M⁻¹⋅r₁
            iters, err = ldiv!(M⁻¹r₁, model, r₁, preconditioner)
        else
            model.transposed = false
            # solve MᵀM⋅x=Mᵀr₁ ==> x=[MᵀM]⁻¹⋅Mᵀr₁=M⁻¹r₁
            Mᵀr₁ = model.v″
            mulMᵀ!(Mᵀr₁, model, r₁)
            iters, err = ldiv!(M⁻¹r₁, model, Mᵀr₁, preconditioner)
        end

        # store result
        @views @. R[:,i]    = r₁
        @views @. M⁻¹R[:,i] = M⁻¹r₁
    end

    return nothing
end

"""
Setup Green's function convolutions with specified random vectors.
"""
function setup!(estimator::EstimateGreensFunction, n₁::Int, n₂::Int) where {T<:AbstractModel}

    # unpack variables
    @unpack nᵥ, r₁, r₂, M⁻¹r₁, M⁻¹r₂, R, M⁻¹R, a, b, L = estimator

    # assign current random vectors and solutions
    estimator.n₁    = n₁
    estimator.n₂    = n₂
    @views @. M⁻¹r₁ = M⁻¹R[:,n₁]
    @views @. r₁    = R[:,n₁]
    @views @. M⁻¹r₂ = M⁻¹R[:,n₂]
    @views @. r₂    = R[:,n₂]

    # initialize measured values to zero
    fill!(estimator.GΔ0,0.0)
    fill!(estimator.GΔ0_GΔ0,0.0)
    fill!(estimator.GΔ0_G0Δ,0.0)
    fill!(estimator.GΔΔ_G00,0.0)

    # vector to be convolved together
    a = estimator.a
    b = estimator.b

    # calcualte G[Δ,0]
    z = estimator.z
    antiperiodic_copy!(a,M⁻¹r₁,L)
    antiperiodic_copy!(z,M⁻¹r₂,L)
    @. a = (a+z)/sqrt(2.0)
    antiperiodic_copy!(b,r₁,L)
    antiperiodic_copy!(z,r₂,L)
    @. b = (b+z)/sqrt(2.0)
    convolve!(estimator.GΔ0,a,b,estimator)

    # calculate G[Δ,0]⋅G[Δ,0]
    periodic_product!(a,M⁻¹r₁,M⁻¹r₂,L)
    periodic_product!(b,r₁,r₂,L)
    convolve!(estimator.GΔ0_GΔ0,a,b,estimator)

    # calculate G[Δ,Δ]⋅G[0,0]
    periodic_product!(a,M⁻¹r₂,r₂,L)
    periodic_product!(b,M⁻¹r₁,r₁,L)
    convolve!(estimator.GΔΔ_G00,a,b,estimator)

    # calculate G[Δ,0]⋅G[0,Δ]
    periodic_product!(a,M⁻¹r₁,r₂,L)
    periodic_product!(b,M⁻¹r₂,r₁,L)
    convolve!(estimator.GΔ0_G0Δ,a,b,estimator)

    return nothing
end

"""
Measure time ordered Green's function G[Δ,0]=Gᵣ(τ)=⟨T⋅cᵢ₊ᵣ(τ)⋅cᵀᵢ(0)⟩ where Δ=(τ,r)
"""
function measure_GΔ0(estimator::EstimateGreensFunction,l₁::Int,l₂::Int,l₃::Int,o₁::Int,o₂::Int,τ::Int)

    L   = estimator.L
    GΔ0 = estimator.GΔ0
    return GΔ0[mod1(τ+1,2L),o₂,o₁,l₁+1,l₂+1,l₃+1]
end

"""
Measure G[Δ,0]⋅G[Δ,0]=⟨T⋅cᵢ₊ᵣ(τ)⋅cᵀᵢ(0)⟩⋅⟨T⋅cᵢ₊ᵣ(τ)⋅cᵀᵢ(0)⟩ where Δ=(τ,r)
"""
function measure_GΔ0_GΔ0(estimator::EstimateGreensFunction,l₁::Int,l₂::Int,l₃::Int,o₁::Int,o₂::Int,τ::Int)

    L       = estimator.L
    GΔ0_GΔ0 = estimator.GΔ0_GΔ0
    return GΔ0_GΔ0[mod1(τ+1,2L),o₂,o₁,l₁+1,l₂+1,l₃+1]
end

"""
Measure G[Δ,Δ]⋅G[0,0]=⟨T⋅cᵢ₊ᵣ(τ)⋅cᵀᵢ₊ᵣ(τ)⟩⋅⟨T⋅cᵢ(0)⋅cᵀᵢ(0)⟩ where Δ=(τ,r)
"""
function measure_GΔΔ_G00(estimator::EstimateGreensFunction,l₁::Int,l₂::Int,l₃::Int,o₁::Int,o₂::Int,τ::Int)

    L       = estimator.L
    GΔΔ_G00 = estimator.GΔΔ_G00
    return GΔΔ_G00[mod1(τ+1,2L),o₂,o₁,l₁+1,l₂+1,l₃+1]
end

"""
Measure G[Δ,0]⋅G[0,Δ] =⟨T⋅cᵢ₊ᵣ(τ)⋅cᵀᵢ(0)⟩⋅⟨T⋅cᵢ(0)⋅cᵀᵢ₊ᵣ(τ)⟩  where Δ=(τ,r)
        G[Δ,0]⋅G[-Δ,0]=⟨T⋅cᵢ₊ᵣ(τ)⋅cᵀᵢ(0)⟩⋅⟨T⋅cᵢ₋ᵣ(-τ)⋅cᵀᵢ(0)⟩ where Δ=(τ,r)
"""
function measure_GΔ0_G0Δ(estimator::EstimateGreensFunction,l₁::Int,l₂::Int,l₃::Int,o₁::Int,o₂::Int,τ::Int)

    L       = estimator.L
    GΔ0_G0Δ = estimator.GΔ0_G0Δ
    return GΔ0_G0Δ[mod1(τ+1,2L),o₂,o₁,l₁+1,l₂+1,l₃+1]
end

"""
Measure time ordered Green's function Gᵢ-ⱼ(τ₂-τ₁)=⟨T⋅cᵢ(τ₂)⋅cᵀⱼ(τ₁)⟩ where -β<(τ₂-τ₁)<β
"""
function estimate(estimator::EstimateGreensFunction,i::Int,j::Int,τ₂::Int,τ₁::Int,σ::Int)

    m = get_index(τ₁,j,estimator.L)
    n = get_index(τ₂,i,estimator.L)
    if σ==1
        Gᵢⱼτ₁τ₂ =  estimator.M⁻¹r₁[n] * estimator.r₁[m]
    elseif σ==2
        Gᵢⱼτ₁τ₂ =  estimator.M⁻¹r₂[n] * estimator.r₂[m]
    else
        throw(DomainError())
    end
    return Gᵢⱼτ₁τ₂
end

"""
Calculate convolution a⋆b.
"""
function convolve!(ab::AbstractArray,a::AbstractArray,b::AbstractArray,estimator::EstimateGreensFunction)

    b′  = estimator.z
    ab′ = estimator.ab′
    ab″ = estimator.ab″
    L   = estimator.L
    N   = estimator.N
    L₃  = estimator.L₃
    L₂  = estimator.L₂
    L₁  = estimator.L₁
    nₛ  = estimator.nₛ

    # forward FFT of a
    a′ = a
    mul!(b′,estimator.pfft,a)
    copyto!(a′,b′)
    
    # forward FFT of b
    mul!(b′,estimator.pfft,b)

    # normalization factor
    V = 2*L*N/nₛ

    # perform elementwise multiplication
    @fastmath @inbounds for k₃ in 1:L₃
        for k₂ in 1:L₂
            for k₁ in 1:L₁
                for s₁ in 1:nₛ
                    for s₂ in 1:nₛ
                        for ω in 1:2L
                            nk₃ = mod1(-k₃+2,L₃)
                            nk₂ = mod1(-k₂+2,L₂)
                            nk₁ = mod1(-k₁+2,L₁)
                            nω  = mod1(-ω+2,2L)
                            ab′[ω,s₂,s₁,k₁,k₂,k₃] = a′[ω,s₂,k₁,k₂,k₃] * b′[nω,s₁,nk₁,nk₂,nk₃] / V
                        end
                    end
                end
            end
        end
    end

    # perform inverse FFT
    mul!(ab″,estimator.pifft,ab′)

    # add result to output vector
    @. ab += ab″

    return nothing
end

"""
Copy so the vector x=[ x(1) , ... , x(τ) , ... , x(L) ] to a vector y so that
y=[ x(1) , ... , x(τ) , ... , x(L) , -x(1) , ... , -x(τ) , ... , -x(L) ]
"""
function antiperiodic_copy!(y::AbstractArray,x::AbstractArray,L::Int)

    N  = div(length(x),L)
    x′ = reshaped(x,L,N)
    y′ = reshaped(y,2L,N)
    @fastmath @inbounds for i in 1:N
        for τ in 1:L 
            y′[τ,i]   =  x′[τ,i]
            y′[τ+L,i] = -x′[τ,i]
        end
    end
    return nothing
end

"""
Multiply so that for x=[ x(1) , ... , x(τ) , ... , x(L) ] and y=[ y(1) , ... , y(τ) , ... , y(L) ] then
z=[ x(1)⋅y(1) , x(2)⋅y(2) , ... , x(L)⋅y(L) , x(1)⋅y(1) , ... , x(L)⋅y(L) , x(L)⋅y(L) ]
"""
function periodic_product!(z::AbstractArray,y::AbstractArray,x::AbstractArray,L::Int)

    N  = div(length(x),L)
    x′ = reshaped(x,L,N)
    y′ = reshaped(y,L,N)
    z′ = reshaped(z,2L,N)
    @fastmath @inbounds for i in 1:N
        for τ in 1:L
            val       = y′[τ,i] * x′[τ,i]
            z′[τ,i]   = val
            z′[τ+L,i] = val
        end
    end

    return nothing
end


end