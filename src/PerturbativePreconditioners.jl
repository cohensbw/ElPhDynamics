module PerturbativePreconditioners

using  UnsafeArrays
using  LinearAlgebra
import LinearAlgebra: ldiv!, mul!, transpose

using ..HolsteinModels: HolsteinModel
using ..Checkerboard: checkerboard_mul!, checkerboard_inverse_mul!
using ..Checkerboard: checkerboard_transpose_mul!, checkerboard_inverse_transpose_mul!
using ..TimeFreqFFTs: TimeFreqFFT, τ_to_ω!, ω_to_τ!
using ..TightBindingFFTs: TightBindingFFT, add_bond!, calc_basis!, r_to_k!, k_to_r!
using ..Utilities: get_index

export LeftPerturbativePreconditioner, setup!

abstract type PerturbativePreconditioner end

mutable struct LeftPerturbativePreconditioner{T1<:AbstractFloat,T2<:Number} <: PerturbativePreconditioner

    "current freuqncy mode ω to consider"
    ω::Int

    "holstein model."
    holstein::HolsteinModel{T1,T2}

    "TimeFreqFFT object for mapping between τ ⟷ ω"
    timefreqfft::TimeFreqFFT{T1}

    "TightBindingFFT"
    tightbindingfft::TightBindingFFT{T1}

    "Specifies whether to do a small V or small K expansion.
    If true  then do small V expansion away from non-interacting limit.
    If false then do small K expansion away from single-site limit."
    limit::Vector{Bool}

    "Order of perturbative expansion for each frequency mode ω"
    order::Vector{Int}

    "Regularization term applied to each frequency mode."
    η::Vector{T1}

    "z(ω) = exp{i⋅Δτ⋅2π/β⋅(ω+1/2)}"
    z::Vector{Complex{T1}}

    "exp{-Δτ⋅V̄} = (1/L)∑exp{-Δτ⋅V(τ)}"
    expnΔτV̄::Vector{Complex{T1}}

    "Green's functions evaluated in a specified limit.
    Gₛ(ω) = [-z⋅(I - zᵀ(ω)⋅Aₛ⁻¹ + i⋅Δτ⋅η(ω))]⁻¹ for a small K expansion away from single-site limit.
    G₀(ω) = [I - z(ω)⋅B₀ - i⋅Δτ⋅η(ω)]⁻¹ where B₀=exp{-Δτ(K-c₀)} for a small V expansion away from non-interacting limit.
    The elements of G₀ are computed assuming you are in the momentum space where K is diagonal."
    G::Matrix{Complex{T1}}

    "Define exp{-Δτ⋅cₛ} such that Aₛ = exp{-Δτ⋅(V̄+cₛ)}"
    expnΔτcₛ::T1

    "Define exp{-Δτ⋅c₀} such that A₀ = exp{-Δτ⋅(V̄+c₀)}"
    expnΔτc₀::T1

    "Temporary vector length NL."
    v1::AbstractVector{Complex{T1}}

    "Temporary vector length NL."
    v2::AbstractVector{Complex{T1}}

    "Temporary vector length N."
    v3::AbstractVector{Complex{T1}}

    "Temporary vector length N."
    v4::AbstractVector{Complex{T1}}

    function LeftPerturbativePreconditioner(holstein::HolsteinModel{T1,T2}, tightbindingfft::TightBindingFFT{T1}) where {T1<:AbstractFloat,T2<:Number}

        N   = holstein.nsites
        L   = holstein.Lτ
        NL  = N*L
        Lo2 = cld(L,2)

        timefreqfft = TimeFreqFFT(holstein.lattice,L)
        ω           = 1
        limit       = zeros(Bool,Lo2)
        order       = zeros(Int,Lo2)
        η           = ones(T1,Lo2)
        z           = [exp(2*π*im*((ω-1)+1/2)/L) for ω in 1:L] # z(ω) = exp{i⋅Δτ⋅2π/β⋅(ω+1/2)}
        expnΔτV̄     = zeros(Complex{T1},N)
        G           = zeros(Complex{T1},N,Lo2)
        expnΔτcₛ    = 1.0
        expnΔτc₀    = 0.0
        v1          = zeros(Complex{T1},NL)
        v2          = zeros(Complex{T1},NL)
        v3          = zeros(Complex{T1},N)
        v4          = zeros(Complex{T1},N)

        return new{T1,T2}(ω,holstein,timefreqfft,tightbindingfft,limit,order,η,z,expnΔτV̄,G,expnΔτcₛ,expnΔτc₀,v1,v2,v3,v4)
    end
end


function setup!(op::LeftPerturbativePreconditioner{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

    N  = op.holstein.nsites::Int
    L  = op.holstein.Lτ::Int
    Δτ = op.holstein.Δτ::T1

    # calulcate diagonal matrix exp{-Δτ⋅V̄} = (1/L)∑exp{-Δτ⋅V(τ)}
    expnΔτV = op.holstein.expnΔτV::Vector{T2}
    @fastmath @inbounds for i in 1:N
        op.expnΔτV̄[i] = 0.0
        for τ in 1:L
            op.expnΔτV̄[i] += expnΔτV[get_index(τ,i,L)]
        end
        op.expnΔτV̄[i] /= L
    end

    # set exp{-Δτ⋅c₀} value
    expnV̄₀      = sum(op.expnΔτV̄)/N # exp{-Δτ⋅V̄₀} = avg[exp{-Δτ⋅V̄}] 
    op.expnΔτc₀ = 1.0/expnV̄₀        # exp{-Δτ⋅c₀} where c₀ = -V̄₀

    # set exp{-Δτ⋅cₛ} value
    op.expnΔτcₛ = 1.0 # exp{-Δτ⋅cₛ}

    # λ are the eigenvalues of the K matrix.
    λ = op.tightbindingfft.λk::Array{T1,4}

    # iterate over frequency
    @fastmath @inbounds for ω in 1:cld(L,2)
        # i⋅Δτ⋅η(ω)
        iΔτη = im*Δτ*op.η[ω]
        # z(ω) = exp{i⋅Δτ⋅2π/β⋅(ω+1/2)}
        z = op.z[ω]
        # if doing small V expansion away from non-interacting limit
        if op.limit[ω]
            # iterate over momentum space k states
            for k in 1:N
                # G₀(ω) = [I - z(ω)⋅B₀ - i⋅Δτ⋅η(ω)]⁻¹ where B₀=exp{-Δτ(K-c₀)}=exp{-Δτ⋅K}⋅exp{+Δτ⋅c₀}
                B₀        = exp(-Δτ*λ[k])/op.expnΔτc₀
                op.G[k,ω] = 1.0/(1.0 - z*B₀ - iΔτη)
            end
        # if doing small K expansion away from single-site limit
        else
            # iterate over real space sites in lattice
            for i in 1:N
                # Gₛ(ω) = [-z⋅(I - zᵀ(ω)⋅Aₛ⁻¹ + i⋅Δτ⋅η(ω))]⁻¹ where Aₛ=exp{-Δτ⋅(V̄+cₛ)}=exp{-Δτ⋅V̄}⋅exp{-Δτ⋅cₛ}
                Aₛ        = op.expnΔτV̄[i]*op.expnΔτcₛ
                op.G[i,ω] = 1.0/(-z*( 1.0 - conj(z)/Aₛ + iΔτη))
            end
        end
    end

    return nothing
end

function setup!(op)

    return nothing
end


function ldiv!(vout::AbstractVector{T},op::PerturbativePreconditioner,vin::AbstractVector{T}) where {T<:AbstractFloat}


    N  = op.holstein.nsites::Int
    L  = op.holstein.Lτ::Int
    v1 = op.v1::Vector{Complex{T}}
    v2 = op.v2::Vector{Complex{T}}

    # 1. apply phase factor to go from (anit-periodic)⟶(periodic) in τ
    # 2. FFT from τ ⟶ ω
    τ_to_ω!(v2,op.timefreqfft,vin)

    @uviews v1 v2 begin

        a1  = reshape(v1,(L,N))
        a1T = reshape(v1,(N,L))
        a2  = reshape(v2,(L,N))
        a2T = reshape(v2,(N,L))

        transpose!(a1T,a2)
        fill!(v2,0.0)

        # iterating over half the range of frequencies
        for ω in 1:cld(L,2)

            # input vector
            u1 = @view a1T[:,ω]

            # output vector
            u2 = @view a2T[:,ω]

            # set frequency
            op.ω = ω

            if op.limit[ω]==true
                # apply small V expansion away from non-interacting limit
                mul_invM0!(u2,op,u1)
            else
                # apply small K expansion away from single-site limit
                mul_invMs!(u2,op,u1)
            end

            # accounting for symmetry
            for i in 1:N
                a2T[i,L-ω+1] = conj(a2T[i,ω])
            end
        end

        transpose!(a1,a2T)
    end

    # 1. iFFT from ω ⟶ τ
    # 2. apply inverse phase factor to go from (periodic)⟶(anti-periodic) in τ
    ω_to_τ!(vout,op.timefreqfft,v1)

    return nothing
end

function ldiv!(op::PerturbativePreconditioner,v::AbstractVector)

    ldiv!(v,op,v)
    return nothing
end


"""
Calculate v′=[1+G₀Γ₀+(G₀Γ₀)²+...]⋅G₀⋅A₀⁻¹⋅v product arrived at perturbing away from the non-interacting limit.
"""
function mul_invM0!(v′::AbstractVector{Complex{T1}},op::LeftPerturbativePreconditioner{T1,T2},v::AbstractVector{Complex{T1}}) where {T1<:AbstractFloat,T2<:Number}

    Δτ   = op.holstein.Δτ::T1
    η    = op.η[op.ω]
    iΔτη = im*Δτ*η
    G    = op.G::Matrix{Complex{T1}}
    v₀   = op.v3::Vector{Complex{T1}}
    vk   = op.v4::Vector{Complex{T1}}

    @uviews G begin

        # v′= A₀⁻¹⋅v where A₀ = exp{-Δτ⋅(V̄+c₀)} = exp{-Δτ⋅V̄}⋅exp{-Δτ⋅c₀}
        @. v′ = inv(op.expnΔτV̄*op.expnΔτc₀) * v

        # G₀(ω) = [I - z(ω)⋅B₀ - i⋅Δτ⋅η(ω)]⁻¹ where B₀=exp{-Δτ(K-c₀)}
        G₀ = @view G[:,op.ω]

        # v′= G₀⋅A₀⁻¹⋅v 
        r_to_k!(vk,op.tightbindingfft,v′)
        @. vk *= G₀
        k_to_r!(v′,op.tightbindingfft,vk)

        # v′=[1 + G₀Γ₀ + (G₀Γ₀)² + ...]⋅G₀⋅A₀⁻¹⋅v where Γ₀ = I - A₀⁻¹ - Δτ⋅(iη)
        # Apply recursive algorithm to execute the multiplication by the expansion
        vₙ = v′
        copyto!(v₀,vₙ)
        @fastmath @inbounds for n in 1:op.order[op.ω]
            # vₙ = Γ₀⋅vₙ₋₁
            @. vₙ *= (1.0 - inv(op.expnΔτV̄*op.expnΔτc₀) - iΔτη)
            # vₙ = G₀⋅Γ₀⋅vₙ₋₁
            r_to_k!(vk,op.tightbindingfft,vₙ)
            @. vk *= G₀
            k_to_r!(vₙ,op.tightbindingfft,vk)
            # vₙ = v₀ + G₀⋅Γ₀⋅vₙ₋₁
            @. vₙ += v₀
        end
    end

    return nothing
end


"""
Calculate v′=[1+GₛΓₛ+(GₛΓₛ)²+...]⋅Gₛ⋅Aₛ⁻¹⋅v product arrived at given by perturbing away from the single-site limit.
"""
function mul_invMs!(v′::AbstractVector{Complex{T1}},op::LeftPerturbativePreconditioner{T1,T2},v::AbstractVector{Complex{T1}}) where {T1<:AbstractFloat,T2<:Number}

    Δτ   = op.holstein.Δτ::T1
    η    = op.η[op.ω]
    iΔτη = im*Δτ*η
    z    = op.z[op.ω]
    G    = op.G::Matrix{Complex{T1}}
    v₀   = op.v3::Vector{Complex{T1}}
    vₙ′  = op.v4::Vector{Complex{T1}}

    neighbor_table_tij = op.holstein.neighbor_table_tij::Matrix{Int}
    coshtij = op.holstein.coshtij::Vector{T2}
    sinhtij = op.holstein.sinhtij::Vector{T2}

    @uviews G begin
        
        # Gₛ(ω) = [-z⋅(I - zᵀ(ω)⋅Aₛ⁻¹ + i⋅Δτ⋅η(ω))]⁻¹
        Gₛ = @view G[:,op.ω]

        # v′= Aₛ⁻¹⋅v where Aₛ = exp{-Δτ⋅(V̄+cₛ)} = exp{-Δτ⋅V̄}⋅exp{-Δτ⋅cₛ}
        @. v′ = inv(op.expnΔτV̄*op.expnΔτc₀) * v

        # v′= Gₛ⋅Aₛ⁻¹⋅v 
        @. v′ *= Gₛ

        # v′=[1 + GₛΓₛ + (GₛΓₛ)² + ...]⋅Gₛ⋅Aₛ⁻¹⋅v where Γₛ = -z⋅[I-Bₛ+Δτ(iη)] where Bₛ=exp{-Δτ·(K-cₛ)}.
        # Apply recursive algorithm to execute the multiplication by the expansion
        vₙ = v′
        copyto!(v₀,vₙ)
        @fastmath @inbounds for n in 1:op.order[op.ω]
            # vₙ = Γₛ⋅vₙ₋₁
            @. vₙ′ = inv(op.expnΔτcₛ) * vₙ # vₙ′ = exp{+Δτ⋅cₛ}⋅vₙ₋₁
            checkerboard_mul!(vₙ′,neighbor_table_tij,coshtij,sinhtij) # vₙ′= Bₛ⋅vₙ₋₁= exp{-Δτ·(K-cₛ)}⋅vₙ₋₁
            @. vₙ = -z * (vₙ + iΔτη*vₙ - vₙ′)
            # vₙ = Gₛ⋅Γₛ⋅vₙ₋₁
            @. vₙ *= Gₛ
            # vₙ = v₀ + Gₛ⋅Γₛ⋅vₙ₋₁
            @. vₙ += v₀
        end
    end

    return nothing
end


end