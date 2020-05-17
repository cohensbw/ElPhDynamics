module KPMPreconditioners

using LinearAlgebra
using FFTW
using UnsafeArrays

import LinearAlgebra: ldiv!, mul!, transpose

using ..HolsteinModels: HolsteinModel
using ..Checkerboard: checkerboard_mul!, checkerboard_transpose_mul!
using ..TimeFreqFFTs: TimeFreqFFT, τ_to_ω!, ω_to_τ!
using ..Utilities: get_index

export LeftKPMPreconditioner, RightKPMPreconditioner, SymmetricKPMPreconditioner

"""
Object to represent Kenerl Polynomial Expansion.
"""
mutable struct KPMExpansion{T1<:AbstractFloat,T2<:Number}

    "Current frequency."
    ω::Int

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
    c2::T2

    "minimum order expansion"
    order_min::Int

    "maximum order expansion"
    order_max::Int

    "number of order 1 expansions."
    n_order_1::Int

    "holstein model."
    holstein::HolsteinModel{T1,T2}

    "TimeFreqFFT object for mapping between τ ⟷ ω"
    timefreqfft::TimeFreqFFT{T1}

    "exp{-Δτ⋅V̄} = (1/L)∑exp{-Δτ⋅V(τ)}"
    expnΔτV̄::Vector{T1}

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

    function KPMExpansion(holstein::HolsteinModel{T1,T2},
                          λ_lo::T1, λ_hi::T1, c1::T1, c2::T2,
                          jackson_kernel::Bool) where {T1<:AbstractFloat,T2<:Number}

        N   = holstein.nsites
        L   = holstein.Lτ
        NL  = N*L
        Lo2 = cld(L,2)

        timefreqfft = TimeFreqFFT(holstein.lattice,L)

        λ_avg   = (λ_hi+λ_lo)/2
        λ_mag   = (λ_hi-λ_lo)/2
        expnΔτV̄ = zeros(T1,N)
        ϕs        = [2*π/L*(ω+1/2) for ω in 0:Lo2-1]
        order     = [max(1,round(Int,c1/ϕ + c2)) for ϕ in ϕs]
        order_max = maximum(order)
        order_min = minimum(order)
        n_order_1 = count(i->i==1,order)
        v1      = zeros(Complex{T1},NL)
        v2      = zeros(Complex{T1},NL)
        v3      = zeros(Complex{T1},N)
        v4      = zeros(Complex{T1},N)
        v5      = zeros(Complex{T1},N)

        # construct expansion of the function f(x)=1.0-exp{i⋅ϕ⋅x} for each ϕ value
        coeff   = Vector{Vector{Complex{T1}}}()
        for ω in 1:cld(L,2)
            n = order[ω]
            ϕ = ϕs[ω]
            # calcualte real part of coefficient
            coeff_re = kpm_coefficients(x->real(scalar_invM(x,ϕ)), n, λ_lo=λ_lo, λ_hi=λ_hi, jackson_kernel=jackson_kernel)
            # calcualte imaginary part of coefficient
            coeff_im = kpm_coefficients(x->imag(scalar_invM(x,ϕ)), n, λ_lo=λ_lo, λ_hi=λ_hi, jackson_kernel=jackson_kernel)
            # record expansion coefficients
            push!(coeff,coeff_re+im*coeff_im)
        end

        return new{T1,T2}(1,λ_lo,λ_hi,λ_avg,λ_mag,c1,c2,order_min,order_max,n_order_1,holstein,timefreqfft,expnΔτV̄,ϕs,coeff,order,v1,v2,v3,v4,v5,0)
    end
end


"""
Abstract type to reprepresent preconditioners based the Kernel Polynomial Method
that uses Chebyshev Polynomials to approximate M⁻¹[ω,ω].
"""
abstract type KPMPreconditioner end

mutable struct LeftKPMPreconditioner{T1<:AbstractFloat,T2<:Number} <: KPMPreconditioner

    expansion::KPMExpansion{T1,T2}

    function LeftKPMPreconditioner(holstein::HolsteinModel{T1,T2},
                                   λ_lo::T1, λ_hi::T1, c1::T1, c2::T1,
                                   jackson_kernel::Bool) where {T1<:AbstractFloat,T2<:Number}

        expansion = KPMExpansion(holstein,λ_lo,λ_hi,c1,c2,jackson_kernel)
        return new{T1,T2}(expansion)
    end

    function LeftKPMPreconditioner(expansion::KPMExpansion{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

        return new{T1,T2}(expansion)
    end
end

mutable struct RightKPMPreconditioner{T1<:AbstractFloat,T2<:Number} <: KPMPreconditioner

    expansion::KPMExpansion{T1,T2}

    function RightKPMPreconditioner(holstein::HolsteinModel{T1,T2},
                                   λ_lo::T1, λ_hi::T1, c1::T1, c2::T1,
                                   jackson_kernel::Bool) where {T1<:AbstractFloat,T2<:Number}

        expansion = KPMExpansion(holstein,λ_lo,λ_hi,c1,c2,jackson_kernel)
        return new{T1,T2}(expansion)
    end

    function RightKPMPreconditioner(expansion::KPMExpansion{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

        return new{T1,T2}(expansion)
    end
end


mutable struct SymmetricKPMPreconditioner{T1<:AbstractFloat,T2<:Number} <: KPMPreconditioner

    expansion::KPMExpansion{T1,T2}
    transposed::Bool

    function SymmetricKPMPreconditioner(holstein::HolsteinModel{T1,T2},
                                        λ_lo::T1, λ_hi::T1, c1::T1, c2::T1,
                                        jackson_kernel::Bool) where {T1<:AbstractFloat,T2<:Number}

        expansion = KPMExpansion(holstein,λ_lo,λ_hi,c1,c2,jackson_kernel)
        return new{T1,T2}(expansion,false)
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

function setup!(op::KPMExpansion{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

    N  = op.holstein.nsites::Int
    L  = op.holstein.Lτ::Int
    Δτ = op.holstein.Δτ

    # calulcate diagonal matrix exp{-Δτ⋅V̄} = (1/L)∑exp{-Δτ⋅V(τ)}
    expnΔτV = op.holstein.expnΔτV::Vector{T2}
    @fastmath @inbounds for i in 1:N
        op.expnΔτV̄[i] = 0.0
        for τ in 1:L
            op.expnΔτV̄[i] += expnΔτV[get_index(τ,i,L)]
        end
        op.expnΔτV̄[i] /= L
    end
    return nothing
end

function setup!(op)

    return nothing
end


"""
Apply Preconditioner.
"""
function ldiv!(vout::AbstractVector{T},P::KPMPreconditioner,vin::AbstractVector{T}) where {T<:AbstractFloat}

    op = P.expansion::KPMExpansion
    N  = op.holstein.nsites::Int
    L  = op.holstein.Lτ::Int
    v1 = op.v1::Vector{Complex{T}}
    v2 = op.v2::Vector{Complex{T}}
    op.checkerboard_count = 0

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

            # multiply by KPM approximation to M⁻¹[ω,ω]
            mul!(u2,P,u1)

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

function ldiv!(op::KPMPreconditioner,v::AbstractVector)

    ldiv!(v,op,v)
    return nothing
end


"""
Multiply by KPM approximation to M⁻¹[ω,ω]
"""
function mul!(v′::AbstractVector{Complex{T1}},P::LeftKPMPreconditioner{T1,T2},v::AbstractVector{Complex{T1}}) where {T1<:AbstractFloat,T2<:Number}

    op    = P.expansion::KPMExpansion{T1,T2}
    ω     = op.ω # current frequency
    order = op.order[ω] # order of expansion
    coeff = op.coeff::Vector{Vector{Complex{T1}}}
    c     = coeff[ω]::Vector{Complex{T1}}

    # Recursively build `uₙ = Tₙ(A)⋅v`.
    fill!(v′,0.0)
    uₙ₋₁ = op.v3
    uₙ   = op.v4
    uₙ₊₁ = op.v5

    copyto!(uₙ,v) # u₁ = v
    @. v′ += c[1] * uₙ # v′ = c₁⋅u₁ = c₁⋅T₁(A)⋅v
    if order>1
        n = 1 # current order
        mulA′!(uₙ₊₁,P,uₙ) # u₂ = A′⋅u₁
        while true
            n += 1 # increment order counter
            copyto!(uₙ₋₁,uₙ) # uₙ₋₁ = uₙ
            copyto!(uₙ,uₙ₊₁) # uₙ   = uₙ₊₁
            @. v′ += c[n] * uₙ # v′ = v′ + cₙ⋅uₙ = v′ + cₙ⋅Tₙ(A)⋅v
            if n==order
                break
            end
            mulA′!(uₙ₊₁,P,uₙ)         # uₙ₊₁ = A′⋅uₙ
            @. uₙ₊₁ = 2 * uₙ₊₁ - uₙ₋₁ # uₙ₊₁ = 2⋅A′⋅uₙ - uₙ₋₁
        end
    end

    return nothing
end


"""
Multiply by KPM approximation to M⁻ᵀ[ω,ω]
"""
function mul!(v′::AbstractVector{Complex{T1}},P::RightKPMPreconditioner{T1,T2},v::AbstractVector{Complex{T1}}) where {T1<:AbstractFloat,T2<:Number}

    op    = P.expansion::KPMExpansion{T1,T2}
    ω     = op.ω # current frequency
    order = op.order[ω] # order of expansion
    coeff = op.coeff::Vector{Vector{Complex{T1}}}
    c     = coeff[ω]::Vector{Complex{T1}}

    # Recursively build `uₙ = Tₙ(A)⋅v`.
    fill!(v′,0.0)
    uₙ₋₁ = op.v3
    uₙ   = op.v4
    uₙ₊₁ = op.v5

    copyto!(uₙ,v) # u₁ = v
    @. v′ += conj(c[1]) * uₙ # v′ = c₁⋅u₁ = c₁⋅T₁(A)⋅v
    if order>1
        n = 1 # current order
        mulA′!(uₙ₊₁,P,uₙ) # u₂ = A′⋅u₁
        while true
            n += 1 # increment order counter
            copyto!(uₙ₋₁,uₙ) # uₙ₋₁ = uₙ
            copyto!(uₙ,uₙ₊₁) # uₙ   = uₙ₊₁
            @. v′ += conj(c[n]) * uₙ # v′ = v′ + cₙ⋅uₙ = v′ + cₙ⋅Tₙ(A)⋅v
            if n==order
                break
            end
            mulA′!(uₙ₊₁,P,uₙ)         # uₙ₊₁ = A′⋅uₙ
            @. uₙ₊₁ = 2 * uₙ₊₁ - uₙ₋₁ # uₙ₊₁ = 2⋅A′⋅uₙ - uₙ₋₁
        end
    end

    return nothing
end


"""
Multiply by KPM approximation to M⁻¹[ω,ω]⋅M⁻ᵀ[ω,ω]
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

    ##########################
    ## Multiply by M⁻ᵀ[ω,ω] ##
    ##########################

    P.transposed = true
    copyto!(uₙ,v) # u₁ = v
    fill!(v′,0.0)
    @. v′ += conj(c[1]) * uₙ # v′ = c₁⋅u₁ = c₁⋅T₁(A)⋅v
    if order>1
        n = 1 # current order
        mulA′!(uₙ₊₁,P,uₙ) # u₂ = A′⋅u₁
        while true
            n += 1 # increment order counter
            copyto!(uₙ₋₁,uₙ) # uₙ₋₁ = uₙ
            copyto!(uₙ,uₙ₊₁) # uₙ   = uₙ₊₁
            @. v′ += conj(c[n]) * uₙ # v′ = v′ + cₙ⋅uₙ = v′ + cₙ⋅Tₙ(A)⋅v
            if n==order
                break
            end
            mulA′!(uₙ₊₁,P,uₙ)         # uₙ₊₁ = A′⋅uₙ
            @. uₙ₊₁ = 2 * uₙ₊₁ - uₙ₋₁ # uₙ₊₁ = 2⋅A′⋅uₙ - uₙ₋₁
        end
    end

    ##########################
    ## Multiply by M⁻¹[ω,ω] ##
    ##########################

    P.transposed = false
    copyto!(uₙ,v′) # u₁ = v
    fill!(v′,0.0)
    @. v′ += c[1] * uₙ # v′ = c₁⋅u₁ = c₁⋅T₁(A)⋅v
    if order>1
        n = 1 # current order
        mulA′!(uₙ₊₁,P,uₙ) # u₂ = A′⋅u₁
        while true
            n += 1 # increment order counter
            copyto!(uₙ₋₁,uₙ) # uₙ₋₁ = uₙ
            copyto!(uₙ,uₙ₊₁) # uₙ   = uₙ₊₁
            @. v′ += c[n] * uₙ # v′ = v′ + cₙ⋅uₙ = v′ + cₙ⋅Tₙ(A)⋅v
            if n==order
                break
            end
            mulA′!(uₙ₊₁,P,uₙ)         # uₙ₊₁ = A′⋅uₙ
            @. uₙ₊₁ = 2 * uₙ₊₁ - uₙ₋₁ # uₙ₊₁ = 2⋅A′⋅uₙ - uₙ₋₁
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
Perform A⋅v where A=exp{-Δτ⋅V̄}⋅exp{-Δτ⋅K}
"""
function mulA!(v′::AbstractVector{Complex{T1}},P::LeftKPMPreconditioner{T1,T2},v::AbstractVector{Complex{T1}}) where {T1<:AbstractFloat,T2<:Number}

    op      = P.expansion::KPMExpansion{T1,T2}
    expnΔτV̄ = op.expnΔτV̄::Vector{T1}

    neighbor_table_tij = op.holstein.neighbor_table_tij::Matrix{Int}
    coshtij = op.holstein.coshtij::Vector{T2}
    sinhtij = op.holstein.sinhtij::Vector{T2}

    copyto!(v′,v)
    checkerboard_mul!(v′,neighbor_table_tij,coshtij,sinhtij)
    @. v′ *= expnΔτV̄

    op.checkerboard_count += 1

    return nothing
end

"""
Perform A⋅v where A=exp{-Δτ⋅K}⋅exp{-Δτ⋅V̄}
"""
function mulA!(v′::AbstractVector{Complex{T1}},P::RightKPMPreconditioner{T1,T2},v::AbstractVector{Complex{T1}}) where {T1<:AbstractFloat,T2<:Number}

    op      = P.expansion::KPMExpansion{T1,T2}
    expnΔτV̄ = op.expnΔτV̄::Vector{T1}

    neighbor_table_tij = op.holstein.neighbor_table_tij::Matrix{Int}
    coshtij = op.holstein.coshtij::Vector{T2}
    sinhtij = op.holstein.sinhtij::Vector{T2}

    @. v′ = expnΔτV̄ * v
    checkerboard_transpose_mul!(v′,neighbor_table_tij,coshtij,sinhtij)

    op.checkerboard_count += 1

    return nothing
end

"""
Perform A⋅v where A=exp{-Δτ⋅K}⋅exp{-Δτ⋅V̄} OR A=exp{-Δτ⋅V̄}⋅exp{-Δτ⋅K} depending on whether the
SymmetricKPMPreconditioner is transposed or not.
"""
function mulA!(v′::AbstractVector{Complex{T1}},P::SymmetricKPMPreconditioner{T1,T2},v::AbstractVector{Complex{T1}}) where {T1<:AbstractFloat,T2<:Number}

    op      = P.expansion::KPMExpansion{T1,T2}
    expnΔτV̄ = op.expnΔτV̄::Vector{T1}

    neighbor_table_tij = op.holstein.neighbor_table_tij::Matrix{Int}
    coshtij = op.holstein.coshtij::Vector{T2}
    sinhtij = op.holstein.sinhtij::Vector{T2}

    if P.transposed
        @. v′ = expnΔτV̄ * v
        checkerboard_transpose_mul!(v′,neighbor_table_tij,coshtij,sinhtij)
    else
        copyto!(v′,v)
        checkerboard_mul!(v′,neighbor_table_tij,coshtij,sinhtij)
        @. v′ *= expnΔτV̄
    end

    op.checkerboard_count += 1

    return nothing
end


"""
This subroutine is completely general. Given a scalar function f(x), the goal is
to calculate coefficients c_m of a polynomial approximation,
   f(x) ~ sum_m c_m T_m(x)
valid for all x in the range
   λ_lo < x < λ_hi

A Chebyshev approximation naturally lies in the range -1 < x < 1, so some rescaling
factors are necessary.
"""
function kpm_coefficients(f::Function, order::Int; λ_lo::T, λ_hi::T, jackson_kernel::Bool=false) where {T<:AbstractFloat}    
    
    M = order
    λ_avg = (λ_hi + λ_lo) / 2.0
    λ_mag = (λ_hi - λ_lo) / 2.0

    c = zeros(T, M)
    N_M = 2*M
    f_orig = [f(λ_mag*cos(π*(n+0.5)/N_M)+λ_avg) for n in 0:(N_M-1)]
    f_hat = FFTW.dct(f_orig) / 2
    
    # FFTW uses the "unitary" normalization. Undo that.
    f_hat[1] *= sqrt(4*N_M)
    @. f_hat[2:N_M] *= sqrt(2*N_M)
    
    for m in 0:(M-1)
        if jackson_kernel
            # Lorentz kernel
            # λ = 0.1
            # g_m = sinh(λ * (1 - m/M)) / sinh(λ)
            
            # Jackson kernel
            g_m = ((M-m+1) * cos(π*m/(M+1)) + sin(π*m/(M+1)) / tan(π/(M+1))) / (M+1)
        else
            g_m = 1
        end
        q_m = π / (m == 0 ? 1 : 2)
        c[m+1] = (π * g_m * f_hat[m+1]) / (N_M * q_m)
    end
    return c
end


"""
Scalar function of x where x has replaced A=exp{-Δτ⋅V̄}⋅exp{-Δτ⋅K} in
the expresion M⁻¹[ω,ω] = I - exp{i⋅ϕ(ω)}⋅A
"""
function scalar_invM(x,ϕ)

    return inv(1.0 - exp(im * ϕ) * x)
end

end