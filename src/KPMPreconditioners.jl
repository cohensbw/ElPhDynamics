module KPMPreconditioners

using LinearAlgebra
using Random
using FFTW
using Logging
using Printf

import LinearAlgebra: ldiv!, mul!, transpose

using ..Models: HolsteinModel, SSHModel, AbstractModel, Continuous, update_model!
using ..Checkerboard: checkerboard_mul!, checkerboard_transpose_mul!, checkerboard_inverse_mul!
using ..TimeFreqFFTs: TimeFreqFFT, τ_to_ω!, ω_to_τ!
using ..Utilities: get_index, reshaped

export LeftRightKPMPreconditioner, LeftKPMPreconditioner, RightKPMPreconditioner, SymmetricKPMPreconditioner, setup!, construct_Bbar

"""
Object to represent Kenerl Polynomial Expansion.
"""
mutable struct KPMExpansion{T1<:AbstractFloat,T2<:Continuous,T3<:AbstractModel}

    "If true apply preconditioner, else default to identity operator."
    active::Bool

    "Current frequency."
    ω::Int

    "Dimension of Krylov subspace used to approximate eigenvalues"
    n::Int

    "Used by Arnoldi method, columns form orthonormal basis of Krylov subspace"
    Q::Matrix{T1}

    "Used by Arnoldi method, A on Q basis, and t is upper Hessenberg"
    h::Matrix{T1}

    "amount by which to buffer min/max eigenvalues to get λ_lo/λ_hi"
    buf::T1

    "Min egeinvalues of A"
    λ_lo::T1

    "Max egeinvalues of A"
    λ_hi::T1

    "λ_avg = (λ_hi+λ_lo)/2"
    λ_avg::T1

    "λ_mag = (λ_hi-λ_lo)/2"
    λ_mag::T1

    "order = c1/phi + c2"
    c1::T1

    "order = c1/phi + c2"
    c2::T1

    "model."
    model::T3

    "TimeFreqFFT object for mapping between τ ⟷ ω"
    timefreqfft::TimeFreqFFT{T1}

    "exp{-Δτ⋅V̄} = (1/L)∑exp{-Δτ⋅V(τ)}"
    expnΔτV̄::Vector{T1}

    "For checkerboard representation of exp{-Δτ⋅K̄}=(1/L)∑exp{-Δτ⋅K(τ)}"
    cosht̄::Vector{T2}

    "For checkerboard representation of exp{-Δτ⋅K̄}=(1/L)∑exp{-Δτ⋅K(τ)}"
    sinht̄::Vector{T2}

    "ϕ=2π/L⋅(ω+1/2)"
    ϕs::Vector{T1}

    "Chebyshev Coefficients."
    coeff::Vector{Vector{Complex{T1}}}

    "Polynomial order of Chebyshev expansion for each ω value."
    order::Vector{Int}

    "Temporary Vector"
    v1::Vector{Complex{T1}}

    "Temporary Vector"
    v2::Vector{Complex{T1}}

    "Temporary Vector"
    v3::Vector{Complex{T1}}

    "Temporary Vector"
    v4::Vector{Complex{T1}}

    "Temporary Vector"
    v5::Vector{Complex{T1}}

    "Count total checkerboard multiplies."
    checkerboard_count::Int

    function KPMExpansion(model::AbstractModel{T1,T2}, n::Int, buf::T1, c1::T1, c2::T1) where {T1,T2}

        N   = model.Nsites
        L   = model.Lτ
        NL  = model.Ndim
        Lo2 = cld(L,2)

        timefreqfft = TimeFreqFFT(model.lattice,L)

        λ_lo      = 0.0
        λ_hi      = 2.0
        λ_avg     = (λ_hi+λ_lo)/2
        λ_mag     = (λ_hi-λ_lo)/2
        expnΔτV̄   = zeros(T1,N)
        cosht̄     = zeros(T2,model.Nbonds)
        sinht̄     = zeros(T2,model.Nbonds)
        ϕs        = [2*π/L*(ω+1/2) for ω in 0:Lo2-1]
        order     = ones(Int,Lo2)
        v1        = zeros(Complex{T1},NL)
        v2        = zeros(Complex{T1},NL)
        v3        = zeros(Complex{T1},N)
        v4        = zeros(Complex{T1},N)
        v5        = zeros(Complex{T1},N)

        if typeof(model) <: HolsteinModel
            cosht̄ .= model.cosht
            sinht̄ .= model.sinht
        elseif typeof(model) <: SSHModel
            expnΔτV̄ .= model.expΔτμ
        else
            throw(TypeError())
        end

        # size of krylov subspace
        n = min(n,N)

        # matrices for arnolid method
        Q = zeros(T1,N,n+1)
        h = zeros(T1,n+1,n)

        # construct expansion of the function f(x)=1.0-exp{i⋅ϕ⋅x} for each ϕ value
        coeff = [zeros(Complex{T1},1) for ω in 1:Lo2]

        return new{T1,T2,typeof(model)}(true,1,n,Q,h,buf,λ_lo,λ_hi,λ_avg,λ_mag,c1,c2,model,timefreqfft,expnΔτV̄,cosht̄,sinht̄,ϕs,coeff,order,v1,v2,v3,v4,v5,0)
    end
end


"""
Abstract type to reprepresent preconditioners based the Kernel Polynomial Method
that uses Chebyshev Polynomials to approximate M⁻¹[ω,ω].
"""
abstract type KPMPreconditioner{T1<:AbstractFloat,T2<:Continuous,T3<:AbstractModel} end


"""
For preconditioning M⋅x=b
"""
mutable struct LeftKPMPreconditioner{T1,T2,T3} <: KPMPreconditioner{T1,T2,T3}

    expansion::KPMExpansion{T1,T2,T3}

    function LeftKPMPreconditioner(model::AbstractModel{T1,T2}, n::Int, buf::T1, c1::T1, c2::T1) where {T1,T2}
        expansion = KPMExpansion(model,n,buf,c1,c2)
        return new{T1,T2,typeof(model)}(expansion)
    end

    function LeftKPMPreconditioner(expansion::KPMExpansion{T1,T2,T3}) where {T1,T2,T3}

        return new{T1,T2,T3}(expansion)
    end
end


"""
For preconditioning Mᵀ⋅x=b
"""
mutable struct RightKPMPreconditioner{T1,T2,T3} <: KPMPreconditioner{T1,T2,T3}

    expansion::KPMExpansion{T1,T2,T3}

    function RightKPMPreconditioner(model::AbstractModel{T1,T2}, n::Int, buf::T1, c1::T1, c2::T1) where {T1,T2}

        expansion = KPMExpansion(model,n,buf,c1,c2)
        T3        = typeof(model)
        return new{T1,T2,T3}(expansion)
    end

    function RightKPMPreconditioner(expansion::KPMExpansion{T1,T2,T3}) where {T1,T2,T3}

        return new{T1,T2,T3}(expansion)
    end
end


"""
For preconditioning M⋅x=b or Mᵀ⋅x=b
"""
mutable struct LeftRightKPMPreconditioner{T1,T2,T3} <: KPMPreconditioner{T1,T2,T3}

    lkpm::LeftKPMPreconditioner{T1,T2,T3}
    rkpm::RightKPMPreconditioner{T1,T2,T3}
    expansion::KPMExpansion{T1,T2,T3}

    function LeftRightKPMPreconditioner(model::AbstractModel{T1,T2}, n::Int, buf::T1, c1::T1, c2::T1) where {T1<:AbstractFloat,T2<:Number}

        lkpm = LeftKPMPreconditioner(model,n,buf,c1,c2)
        rkpm = transpose(lkpm)
        expansion = lkpm.expansion

        return new{T1,T2,typeof(model)}(lkpm,rkpm,expansion)
    end
end


"""
For preconditioning MᵀM⋅x=b
"""
mutable struct SymmetricKPMPreconditioner{T1,T2,T3} <: KPMPreconditioner{T1,T2,T3}

    expansion::KPMExpansion{T1,T2,T3}
    transposed::Bool

    function SymmetricKPMPreconditioner(model::AbstractModel{T1,T2}, n::Int, buf::T1, c1::T1, c2::T1) where {T1,T2}

        expansion = KPMExpansion(model,n,buf,c1,c2)
        T3        = typeof(model)
        return new{T1,T2,T3}(expansion,false)
    end

    function SymmetricKPMPreconditioner(expansion::KPMExpansion{T1,T2,T3}) where {T1,T2,T3}

        return new{T1,T2,T3}(expansion,false)
    end
end


"""
Return RightKPMPreconditioner given LeftKPMPreconditioner.
"""
function transpose(P::LeftKPMPreconditioner)

    return RightKPMPreconditioner(P.expansion)
end


"""
Return LeftKPMPreconditioner given RightKPMPreconditioner.
"""
function transpose(P::RightKPMPreconditioner)

    return LeftKPMPreconditioner(P.expansion)
end


"""
Set up the KPMPreconditioner i.e. calculate exp{-Δτ⋅V̄}.
"""
function setup!(op::KPMPreconditioner)

    setup!(op.expansion)
    return nothing
end


"""
Update the preconditioner.
"""
function setup!(op::KPMExpansion{T1,T2,T3}) where {T1,T2,T3}

    # update exp{-Δτ⋅V̄} and exp{-Δτ⋅K̄}
    update_A!(op)

    # approximate min/max eigenvalue of A = exp{-Δτ⋅V̄}⋅exp{-Δτ⋅K̄}
    e_min, e_max = arnoldi_eigenvalue_bounds!(op, op.Q, op.h, op.v3, op.v4, op.model.rng)
    @printf "[ %.6f , %.6f ]\n" e_min e_max

    # preconditioner can only be setup and active if e_min and e_max are reasonable
    # if isfinite(e_min) && isfinite(e_max)
    if (0.0 < e_min < 1.0) && (1.0 < e_max) && (e_max-e_min)<2.0

        # compute λ_lo and λ_hi
        λ_lo = max(0.0 , (1-2*op.buf)*e_min)
        λ_hi = (1+2*op.buf)*e_max

        # if λ_lo or λ_hi has changed by a factor of more than op.buf,
        # recompute expansion coefficients
        if !isapprox(λ_lo, op.λ_lo, rtol=op.buf) || !isapprox(λ_hi, op.λ_hi, rtol=op.buf)

            op.λ_lo  = λ_lo
            op.λ_hi  = λ_hi
            op.λ_avg = (op.λ_hi+op.λ_lo)/2
            op.λ_mag = (op.λ_hi-op.λ_lo)/2

            # update expansions
            for ω in 1:length(op.ϕs)
                # calculate order of expansion
                coeff       = op.coeff[ω]
                ϕ           = op.ϕs[ω]
                # order       = round(Int, op.c1/ϕ + op.c2)
                order       = floor(Int, (op.λ_hi-op.λ_lo)*(op.c1/ϕ + op.c2))
                order       = max(1,order)
                op.order[ω] = order
                # resize vector containing expansion coefficients
                resize!(coeff,order)
                # calculate expansion coefficients
                kpm_coefficients!(coeff, order, op.λ_lo, op.λ_hi, ϕ) 
            end
        end

        # set preconditioner to being active
        op.active = true
    else

        # deactivate preconditioner
        # println("deactive")
        op.active = false
    end

    return nothing
end

function setup!(op)

    return nothing
end


"""
Calculate exp{-Δτ⋅V̄} and exp{-Δτ⋅K̄}
"""
function update_A!(op::KPMExpansion{T1,T2,T3}) where {T1,T2,T3<:HolsteinModel}

    N  = op.model.Nsites::Int
    L  = op.model.Lτ::Int
    Δτ = op.model.Δτ

    # calulcate diagonal matrix exp{-Δτ⋅V̄}
    expnΔτV = op.model.expnΔτV::Vector{T2}
    @fastmath @inbounds for i in 1:N
        op.expnΔτV̄[i] = 0.0
        for τ in 1:L
            op.expnΔτV̄[i] += expnΔτV[get_index(τ,i,L)]
        end
        op.expnΔτV̄[i] /= L
    end

    return nothing
end


"""
Update the matrix A=exp{-Δτ⋅V̄}⋅exp{-Δτ⋅K̄}
"""
function update_A!(op::KPMExpansion{T1,T2,T3}) where {T1,T2,T3<:SSHModel}

    N  = op.model.Nbonds::Int
    L  = op.model.Lτ::Int
    Δτ = op.model.Δτ

    # calulcate checkerboard representation of matrix exp{-Δτ⋅K̄}
    cosht       = op.model.cosht::Matrix{T2}
    sinht       = op.model.sinht::Matrix{T2}
    t′          = op.model.t′::Matrix{T2}
    chkbrd_perm = op.model.checkerboard_perm::Vector{Int}
    @fastmath @inbounds for i in 1:N
        op.cosht̄[i] = 0.0
        op.sinht̄[i] = 0.0
        for τ in 1:L
            op.cosht̄[i] += cosht[τ,i]
            op.sinht̄[i] += sinht[τ,i]
        end
        op.cosht̄[i] /= L
        op.sinht̄[i] /= L
    end

    # calulcate diagonal matrix exp{-Δτ⋅V̄}
    copyto!(op.expnΔτV̄,op.model.expΔτμ)
    
    return nothing
end


"""
Perform A⋅v where A=exp{-Δτ⋅K̄}⋅exp{-Δτ⋅V̄}
"""
function mul!(v′::AbstractVector{T4},op::KPMExpansion{T1,T2,T3},v::AbstractVector{T4}) where {T1,T2,T3,T4<:Continuous}

    expnΔτV̄ = op.expnΔτV̄::Vector{T1}
    neighbor_table = op.model.neighbor_table::Matrix{Int}
    cosht̄ = op.cosht̄::Vector{T2}
    sinht̄ = op.sinht̄::Vector{T2}

    @. v′ = expnΔτV̄ * v
    checkerboard_mul!(v′,neighbor_table,cosht̄,sinht̄)

    op.checkerboard_count += 1

    return nothing
end


"""
Perform A⁻¹⋅v where A = exp{-Δτ⋅K̄}⋅exp{-Δτ⋅V̄}
"""
function ldiv!(v′::AbstractVector{T4},op::KPMExpansion{T1,T2,T3},v::AbstractVector{T4}) where {T1,T2,T3,T4<:Continuous}

    expnΔτV̄ = op.expnΔτV̄::Vector{T1}
    neighbor_table = op.model.neighbor_table::Matrix{Int}
    cosht̄ = op.cosht̄::Vector{T2}
    sinht̄ = op.sinht̄::Vector{T2}

    copyto!(v′,v)
    checkerboard_inverse_mul!(v′,neighbor_table,cosht̄,sinht̄)
    @. v′ /= expnΔτV̄

    op.checkerboard_count += 1

    return nothing
end


"""
Apply Preconditioner.
"""
function ldiv!(vout::AbstractVector{T},P::KPMPreconditioner,vin::AbstractVector{T}) where {T<:AbstractFloat}

    op = P.expansion::KPMExpansion
    N  = op.model.Nsites::Int
    L  = op.model.Lτ::Int
    v1 = op.v1::Vector{Complex{T}}
    v2 = op.v2::Vector{Complex{T}}
    op.checkerboard_count = 0

    if op.active # apply preconditioner if active

        # 1. apply phase factor to go from (anit-periodic)⟶(periodic) in τ
        # 2. FFT from τ ⟶ ω
        τ_to_ω!(v2,op.timefreqfft,vin)

        a1  = reshaped(v1,(L,N))
        a1T = reshaped(v1,(N,L))
        a2  = reshaped(v2,(L,N))
        a2T = reshaped(v2,(N,L))

        transpose!(a1T,a2)

        # iterating over half the range of frequencies
        @fastmath @inbounds for ω in 1:cld(L,2)

            # input vector
            u1 = @view a1T[:,ω]

            # output vector
            u2 = @view a2T[:,ω]

            # set frequency
            op.ω = ω

            # multiply by KPM approximation to M⁻¹[ω,ω]
            mul!(u2,P,u1)

            # accounting for symmetry
            for i in 1:N
                a2T[i,L-ω+1] = conj(a2T[i,ω])
            end
        end

        transpose!(a1,a2T)

        # 1. iFFT from ω ⟶ τ
        # 2. apply inverse phase factor to go from (periodic)⟶(anti-periodic) in τ
        ω_to_τ!(vout,op.timefreqfft,v1)

    else # if preconditioner inactive then behave like identity matrix

        copyto!(vout,vin)
    end

    return nothing
end


function ldiv!(op::KPMPreconditioner,v::AbstractVector)

    ldiv!(v,op,v)
    return nothing
end


"""
Multiply by KPM approximation for M⁻¹[ω,ω] or M⁻ᵀ[ω,ω]
"""
function mul!(v′::AbstractVector{Complex{T1}},P::LeftRightKPMPreconditioner{T1,T2,T3},v::AbstractVector{Complex{T1}}) where {T1,T2,T3}

    op    = P.expansion::KPMExpansion{T1,T2,T3}
    model = op.model::T3
    lkpm  = P.lkpm::LeftKPMPreconditioner{T1,T2,T3}
    rkpm  = P.rkpm::RightKPMPreconditioner{T1,T2,T3}

    if model.transposed
        mul!(v′,rkpm,v)
    else
        mul!(v′,lkpm,v)
    end

    return nothing
end


"""
Multiply by KPM approximation for M⁻¹[ω,ω]
"""
function mul!(v′::AbstractVector{Complex{T1}},P::LeftKPMPreconditioner{T1,T2,T3},v::AbstractVector{Complex{T1}}) where {T1,T2,T3}

    op    = P.expansion::KPMExpansion{T1,T2,T3}
    ω     = op.ω # current frequency
    order = op.order[ω] # order of expansion
    coeff = op.coeff::Vector{Vector{Complex{T1}}}
    c     = coeff[ω]::Vector{Complex{T1}}

    # Recursively build `uₙ = Tₙ(A)⋅v`.
    uₙ₋₁ = op.v3
    uₙ   = op.v4
    uₙ₊₁ = op.v5

    # v′ = c₁⋅u₁ = c₁⋅T₁(A)⋅v
    @. v′ = c[1] * v
    if order>1
        n = 1 # current order
        copyto!(uₙ,v)     # u₁ = v
        mulA′!(uₙ₊₁,P,uₙ) # u₂ = A′⋅u₁
        @fastmath @inbounds while true
            n += 1 # increment order counter
            # uₙ₋₁ = uₙ
            # uₙ   = uₙ₊₁
            temp = uₙ₋₁
            uₙ₋₁ = uₙ
            uₙ   = uₙ₊₁
            uₙ₊₁ = temp
            # v′ = v′ + cₙ⋅uₙ = v′ + cₙ⋅Tₙ(A)⋅v
            @. v′ += c[n] * uₙ
            if n==order
                break
            end
            # uₙ₊₁ = A′⋅uₙ
            mulA′!(uₙ₊₁,P,uₙ)
            # uₙ₊₁ = 2⋅A′⋅uₙ - uₙ₋₁
            @. uₙ₊₁ = 2 * uₙ₊₁ - uₙ₋₁
        end
    end

    return nothing
end


"""
Multiply by KPM approximation for M⁻ᵀ[ω,ω]
"""
function mul!(v′::AbstractVector{Complex{T1}},P::RightKPMPreconditioner{T1,T2},v::AbstractVector{Complex{T1}}) where {T1<:AbstractFloat,T2<:Number}

    op    = P.expansion::KPMExpansion{T1,T2}
    ω     = op.ω # current frequency
    order = op.order[ω] # order of expansion
    coeff = op.coeff::Vector{Vector{Complex{T1}}}
    c     = coeff[ω]::Vector{Complex{T1}}

    # Recursively build `uₙ = Tₙ(A)⋅v`.
    uₙ₋₁ = op.v3
    uₙ   = op.v4
    uₙ₊₁ = op.v5

    # v′ = c₁⋅u₁ = c₁⋅T₁(A)⋅v
    @. v′ = conj(c[1]) * v
    if order>1
        n = 1 # current order
        copyto!(uₙ,v)     # u₁ = v
        mulA′!(uₙ₊₁,P,uₙ) # u₂ = A′⋅u₁
        @fastmath @inbounds while true
            n += 1 # increment order counter
            # uₙ₋₁ = uₙ
            # uₙ   = uₙ₊₁
            temp = uₙ₋₁
            uₙ₋₁ = uₙ
            uₙ   = uₙ₊₁
            uₙ₊₁ = temp
            # v′ = v′ + cₙ⋅uₙ = v′ + cₙ⋅Tₙ(A)⋅v
            @. v′ += conj(c[n]) * uₙ
            if n==order
                break
            end
            # uₙ₊₁ = A′⋅uₙ
            mulA′!(uₙ₊₁,P,uₙ)
            # uₙ₊₁ = 2⋅A′⋅uₙ - uₙ₋₁
            @. uₙ₊₁ = 2 * uₙ₊₁ - uₙ₋₁
        end
    end

    return nothing
end


"""
Multiply by KPM approximation for M⁻¹[ω,ω]⋅M⁻ᵀ[ω,ω]
"""
function mul!(v′::AbstractVector{Complex{T1}},P::SymmetricKPMPreconditioner{T1,T2},v::AbstractVector{Complex{T1}}) where {T1<:AbstractFloat,T2<:Number}

    op    = P.expansion::KPMExpansion{T1,T2}
    ω     = op.ω # current frequency
    order = op.order[ω] # order of expansion
    coeff = op.coeff::Vector{Vector{Complex{T1}}}
    c     = coeff[ω]::Vector{Complex{T1}}

    # Recursively build `uₙ = Tₙ(A)⋅v`.
    uₙ₋₁ = op.v3
    uₙ   = op.v4
    uₙ₊₁ = op.v5

    # multiply by M⁻ᵀ[ω,ω]
    P.transposed = true

    # v′ = c₁⋅u₁ = c₁⋅T₁(A)⋅v
    @. v′ = conj(c[1]) * v
    if order>1
        n = 1 # current order
        copyto!(uₙ,v)     # u₁ = v
        mulA′!(uₙ₊₁,P,uₙ) # u₂ = A′⋅u₁
        @fastmath @inbounds while true
            n += 1 # increment order counter
            # uₙ₋₁ = uₙ
            # uₙ   = uₙ₊₁
            temp = uₙ₋₁
            uₙ₋₁ = uₙ
            uₙ   = uₙ₊₁
            uₙ₊₁ = temp
            # v′ = v′ + cₙ⋅uₙ = v′ + cₙ⋅Tₙ(A)⋅v
            @. v′ += conj(c[n]) * uₙ
            if n==order
                break
            end
            # uₙ₊₁ = A′⋅uₙ
            mulA′!(uₙ₊₁,P,uₙ)
            # uₙ₊₁ = 2⋅A′⋅uₙ - uₙ₋₁
            @. uₙ₊₁ = 2 * uₙ₊₁ - uₙ₋₁
        end
    end

    # Multiply by M⁻¹[ω,ω]
    P.transposed = false

    if order==1
        @. v′ *= c[1] # v′ = c₁⋅u₁ = c₁⋅T₁(A)⋅v
    else
        copyto!(uₙ,v′)    # u₁ = v
        @. v′ = c[1] * v′ # v′ = c₁⋅u₁ = c₁⋅T₁(A)⋅v
        n = 1 # current order
        mulA′!(uₙ₊₁,P,uₙ) # u₂ = A′⋅u₁
        @fastmath @inbounds while true
            n += 1 # increment order counter
            # uₙ₋₁ = uₙ
            # uₙ   = uₙ₊₁
            temp = uₙ₋₁
            uₙ₋₁ = uₙ
            uₙ   = uₙ₊₁
            uₙ₊₁ = temp
            # v′ = v′ + cₙ⋅uₙ = v′ + cₙ⋅Tₙ(A)⋅v
            @. v′ += c[n] * uₙ
            if n==order
                break
            end
            # uₙ₊₁ = A′⋅uₙ
            mulA′!(uₙ₊₁,P,uₙ)
            # uₙ₊₁ = 2⋅A′⋅uₙ - uₙ₋₁
            @. uₙ₊₁ = 2 * uₙ₊₁ - uₙ₋₁
        end
    end

    return nothing
end


"""
Perform A′⋅v where A′= 2⋅(A-λ_lo)/(λ_hi-λ_lo)-I
"""
function mulA′!(v′::AbstractVector{Complex{T}},P::KPMPreconditioner,v::AbstractVector{Complex{T}}) where {T<:AbstractFloat}

    λ_avg = P.expansion.λ_avg::T
    λ_mag = P.expansion.λ_mag::T
    mulA!(v′,P,v)
    @. v′ = (1/λ_mag)*v′ - (λ_avg/λ_mag)*v

    return nothing
end


"""
Perform A⋅v where A=exp{-Δτ⋅V̄}⋅exp{-Δτ⋅K̄} or A=exp{-Δτ⋅K}⋅exp{-Δτ⋅V̄}
"""
function mulA!(v′::AbstractVector{Complex{T1}},P::LeftRightKPMPreconditioner{T1,T2},v::AbstractVector{Complex{T1}}) where {T1<:AbstractFloat,T2<:Number}

    op    = P.expansion::KPMExpansion{T1,T2}
    model = op.model::T3
    lkpm  = P.lkpm::LeftKPMPreconditioner{T1,T2}
    rkpm  = P.rkpm::RightKPMPreconditioner{T1,T2}

    if model.transposed
        mul!(v′,rkpm,v)
    else
        mul!(v′,lkpm,v)
    end

    return nothing
end

"""
Perform A⋅v where A=exp{-Δτ⋅V̄}⋅exp{-Δτ⋅K̄}
"""
function mulA!(v′::AbstractVector{Complex{T1}},P::LeftKPMPreconditioner{T1,T2,T3},v::AbstractVector{Complex{T1}}) where {T1,T2,T3}

    op      = P.expansion::KPMExpansion{T1,T2,T3}
    expnΔτV̄ = op.expnΔτV̄::Vector{T1}
    neighbor_table = op.model.neighbor_table::Matrix{Int}
    cosht̄ = op.cosht̄::Vector{T2}
    sinht̄ = op.sinht̄::Vector{T2}

    @. v′ = expnΔτV̄ * v
    checkerboard_mul!(v′,neighbor_table,cosht̄,sinht̄)

    op.checkerboard_count += 1

    return nothing
end

"""
Perform A⋅v where A=exp{-Δτ⋅K̄}⋅exp{-Δτ⋅V̄}
"""
function mulA!(v′::AbstractVector{Complex{T1}},P::RightKPMPreconditioner{T1,T2,T3},v::AbstractVector{Complex{T1}}) where {T1,T2,T3}

    op      = P.expansion::KPMExpansion{T1,T2,T3}
    expnΔτV̄ = op.expnΔτV̄::Vector{T1}
    neighbor_table = op.model.neighbor_table::Matrix{Int}
    cosht̄ = op.cosht̄::Vector{T2}
    sinht̄ = op.sinht̄::Vector{T2}

    copyto!(v′,v)
    checkerboard_transpose_mul!(v′,neighbor_table,cosht̄,sinht̄)
    @. v′ *= expnΔτV̄

    op.checkerboard_count += 1

    return nothing
end


"""
Perform A⋅v where A=exp{-Δτ⋅V̄}⋅exp{-Δτ⋅K̄} or A=exp{-Δτ⋅K̄}⋅exp{-Δτ⋅V̄}
"""
function mulA!(v′::AbstractVector{Complex{T1}},P::SymmetricKPMPreconditioner{T1,T2,T3},v::AbstractVector{Complex{T1}}) where {T1,T2,T3}

    op      = P.expansion::KPMExpansion{T1,T2,T3}
    expnΔτV̄ = op.expnΔτV̄::Vector{T1}
    neighbor_table = op.model.neighbor_table::Matrix{Int}
    cosht̄ = op.cosht̄::Vector{T2}
    sinht̄ = op.sinht̄::Vector{T2}

    if P.transposed
        copyto!(v′,v)
        checkerboard_transpose_mul!(v′,neighbor_table,cosht̄,sinht̄)
        @. v′ *= expnΔτV̄
    else
        @. v′ = expnΔτV̄ * v
        checkerboard_mul!(v′,neighbor_table,cosht̄,sinht̄)
    end

    op.checkerboard_count += 1

    return nothing
end


"""
Calculate coefficients c_m of a polynomial approximation,
    M⁻¹[ω,ω] ~ sum_m c_m T_m(x)
valid for all x in the range
    λ_lo < x < λ_hi.
A Chebyshev approximation naturally lies in the range -1 < x < 1, so some rescaling
factors are necessary.
"""
function kpm_coefficients!(c::AbstractVector{T}, order::Int, λ_lo, λ_hi, ϕ) where {T<:Complex}
    
    M     = order
    N_M   = 2*M
    λ_avg = (λ_hi+λ_lo)/2
    λ_mag = (λ_hi-λ_lo)/2

    # initialize coefficients to zero
    fill!(c,0)

    # declare temporary array
    c′ = zeros(T,N_M)

    #####################################
    ## Real Part Exapnsion Coefficient ##
    #####################################

    for n in 0:(N_M-1)
        c′[n+1] = real(scalar_invM(λ_mag*cos(π*(n+0.5)/N_M)+λ_avg,ϕ))
    end
    FFTW.dct!(c′)
    
    # FFTW uses the "unitary" normalization. Undo that.
    c′    *= sqrt(2*N_M)/2
    c′[1] *= sqrt(2)
    
    for m in 0:(M-1)
        q_m = π / (m == 0 ? 1 : 2)
        c[m+1] += (π * c′[m+1]) / (N_M * q_m)
    end

    ##########################################
    ## Imaginary Part Exapnsion Coefficient ##
    ##########################################

    for n in 0:(N_M-1)
        c′[n+1] = imag(scalar_invM(λ_mag*cos(π*(n+0.5)/N_M)+λ_avg,ϕ))
    end
    FFTW.dct!(c′)
    
    # FFTW uses the "unitary" normalization. Undo that.
    c′    *= sqrt(2*N_M)/2
    c′[1] *= sqrt(2)
    
    for m in 0:(M-1)
        q_m = π / (m == 0 ? 1 : 2)
        c[m+1] += im * (π * c′[m+1]) / (N_M * q_m)
    end

    return nothing
end

"""
Computes a basis of the (n + 1)-Krylov subspace of A: the space spanned by {b, Ab, ..., A^n b}.
Then use this to approximate the min and max eigenvalues of A.
"""
function arnoldi_eigenvalue_bounds!(A, Q::AbstractMatrix{T1}, h::AbstractMatrix{T1}, b::AbstractVector{T2}, v::AbstractVector{T2}, rng::AbstractRNG) where {T1<:AbstractFloat,T2<:Continuous}

    # dimension of Krylov subspace, must be >= 1
    n = size(h,2)

    # dimension of A matrix
    m = size(Q,1)

    ################################
    ## Arnoldi for Max Eigenvalue ##
    ################################

    # randomize input vector using for loop because
    # b vector be of complex valued data type
    for i in 1:m
        b[i] = randn(rng,T1)
    end

    # normalize input vector
    normalize!(b)

    # Use it as the first Krylov vector
    @. Q[:,1] = real(b)

    l = n
    for k in 1:n
        mul!(v,A,b) # Generate a new candidate vector
        for j in 1:k # Subtract the projections on previous vectors
            Qj      = @view Q[:, j]
            h[j, k] = real(dot(Qj, v))
            @. v   -= h[j, k] * Qj
        end
        h[k+1, k] = norm(v)
        # Add the produced vector to the list, unless the zero vector is produced
        if h[k+1, k] > 1e-12
            @. b = v / h[k + 1, k]
            @. Q[:,k+1] = real(b)
        else  # If that happens, stop iterating.
            l = k
            break
        end
    end

    # calulcate min and max eigenvalues
    h′       = @view h[1:l,1:l]
    if all(i->isfinite(i), h′)
        eigvs    = eigvals!(h′)
        e_max    = maximum(real, eigvs)
    else
        e_max = Inf
    end

    ################################
    ## Arnoldi for Min Eigenvalue ##
    ################################

    # randomize input vector
    for i in 1:m
        b[i] = randn(rng,T1)
    end

    # normalize input vector
    normalize!(b)

    # Use it as the first Krylov vector
    Q1 = @view Q[:,1]
    @. Q1 = real(b)

    l = n
    for k in 1:n
        ldiv!(v,A,b) # Generate a new candidate vector
        for j in 1:k # Subtract the projections on previous vectors
            Qj      = @view Q[:, j]
            h[j, k] = real(dot(Qj, v))
            @. v   -= h[j, k] * Qj
        end
        h[k+1, k] = norm(v)
        # Add the produced vector to the list, unless the zero vector is produced
        if h[k+1, k] > 1e-12
            @. b = v / h[k + 1, k]
            @. Q[:,k+1] = real(b)
        else  # If that happens, stop iterating.
            l = k
            break
        end
    end

    # calulcate min and max eigenvalues
    h′     = @view h[1:l,1:l]
    if all(i->isfinite(i), h′)
        eigvs = eigvals!(h′)
        e_min = 1/maximum(real, eigvs)
    else
        e_min = -Inf
    end

    return e_min, e_max
end

"""
Scalar function of x where x has replaced A=exp{-Δτ⋅V̄}⋅exp{-Δτ⋅K} in
the expresion M⁻¹[ω,ω] = (I - exp{-i⋅ϕ(ω)}⋅A)⁻¹
"""
function scalar_invM(x,ϕ)

    return inv(1.0 - exp(-im * ϕ) * x)
end

"""
Function to construct Bbar (also called A) matrix.
"""
function construct_Bbar(op::KPMPreconditioner{T1,T2,T3};threshold::T1=1e-10) where {T1,T2,T3}

    expansion = op.expansion
    N         = expansion.model.Nsites

    # store matrix elements
    rows     = Int[]
    cols     = Int[]
    elements = T1[]

    # stores columns vector
    v  = zeros(T1,N)
    Av = zeros(T1,N)

    # iterating over rows
    for col in 1:N
        # initialize column vector as unit vector
        v      .= 0.0
        v[col]  = 1.0
        # doing matrix vector multiply
        mul!(Av,expansion,v)
        # iterate of column vecto
        for row in 1:N
            # if nonzero
            if abs(Av[row])>threshold
                # save matrix element
                append!(rows,row)
                append!(cols,col)
                append!(elements,Av[row])
            end
        end
    end
    return rows, cols, elements

    return nothing
end

end