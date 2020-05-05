module DiagonalPreconditioners

using  UnsafeArrays
using  LinearAlgebra
import LinearAlgebra: ldiv!, mul!, transpose

using ..HolsteinModels: HolsteinModel
using ..Checkerboard:   checkerboard_matrix, checkerboard_mul!, checkerboard_transpose_mul!
using ..TimeFreqFFTs:   TimeFreqFFT, τ_to_ω!, ω_to_τ!
using ..Utilities:      get_index

export LeftDiagonalPreconditioner, RightDiagonalPreconditioner, SplitDiagonalPreconditioner, setup!

abstract type DiagonalPreconditioner end

"""
Split diagonal preconditioner.
"""
mutable struct SplitDiagonalPreconditioner{T1<:AbstractFloat,T2<:Number} <: DiagonalPreconditioner

    "holstein model."
    holstein::HolsteinModel{T1,T2}

    "TimeFreqFFT object for mapping between τ ⟷ ω"
    timefreqfft::TimeFreqFFT{T1}

    "Represents (1/L)⋅∑[exp(-Δτ⋅λ⋅x(τ))]"
    expnΔτV_bar::Vector{T2}

    "Array of complex phase factor need to multiply by M[ω,ω] matrices."
    ω_phases::Vector{Complex{T1}}

    "Block Diagonal of M matrix in ω space evaluated in the single-site limit."
    M0ᵀωM0ω::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z1::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z2::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z3::Vector{Complex{T1}}

    function SplitDiagonalPreconditioner(holstein::HolsteinModel{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

        N                  = holstein.nsites
        L                  = holstein.Lτ
        Δτ                 = holstein.Δτ
        tij                = holstein.tij
        neighbor_table_tij = holstein.neighbor_table_tij

        timefreqfft  = TimeFreqFFT(holstein.lattice,L)
        expnΔτV_bar  = zeros(T1,N)
        ω_phases     = [exp(2*π*im*((ω-1)+1/2)/L) for ω = 1:L]
        M0ᵀωM0ω      = zeros(Complex{T1},N*L)
        z1           = zeros(Complex{T1},N*L)
        z2           = zeros(Complex{T1},N*L)
        z3           = zeros(Complex{T1},N*L)

        return new{T1,T2}(holstein, timefreqfft, expnΔτV_bar, ω_phases, M0ᵀωM0ω, z1, z2, z3)
    end
end

"""
Left diagonal preconditioner.
"""
mutable struct LeftDiagonalPreconditioner{T1<:AbstractFloat,T2<:Number} <: DiagonalPreconditioner

    "holstein model."
    holstein::HolsteinModel{T1,T2}

    "TimeFreqFFT object for mapping between τ ⟷ ω"
    timefreqfft::TimeFreqFFT{T1}

    "Represents (1/L)⋅∑[exp(-Δτ⋅λ⋅x(τ))]"
    expnΔτV_bar::Vector{T2}

    "Array of complex phase factor need to multiply by M[ω,ω] matrices."
    ω_phases::Vector{Complex{T1}}

    "Block Diagonal of M matrix in ω space evaluated in the single-site limit."
    M0ω::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z1::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z2::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z3::Vector{Complex{T1}}

    function LeftDiagonalPreconditioner(holstein::HolsteinModel{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

        N                  = holstein.nsites
        L                  = holstein.Lτ
        NL                 = N*L
        Δτ                 = holstein.Δτ
        tij                = holstein.tij
        neighbor_table_tij = holstein.neighbor_table_tij

        timefreqfft  = TimeFreqFFT(holstein.lattice,L)
        expnΔτV_bar  = zeros(T1,N)
        ω_phases     = [exp(2*π*im*((ω-1)+1/2)/L) for ω = 1:L]
        M0ω          = zeros(Complex{T1},NL)
        z1           = zeros(Complex{T1},NL)
        z2           = zeros(Complex{T1},NL)
        z3           = zeros(Complex{T1},NL)

        return new{T1,T2}(holstein, timefreqfft, expnΔτV_bar, ω_phases, M0ω, z1, z2, z3)
    end
end


"""
Right diagonal preconditioner. Always constructed from an instance of LeftDiagonalPreconditioner.
"""
mutable struct RightDiagonalPreconditioner{T1<:AbstractFloat,T2<:Number} <: DiagonalPreconditioner

    "holstein model."
    holstein::HolsteinModel{T1,T2}

    "TimeFreqFFT object for mapping between τ ⟷ ω"
    timefreqfft::TimeFreqFFT{T1}

    "Represents (1/L)⋅∑[exp(-Δτ⋅λ⋅x(τ))]"
    expnΔτV_bar::Vector{T2}

    "Array of complex phase factor need to multiply by M[ω,ω] matrices."
    ω_phases::Vector{Complex{T1}}

    "Diagonal of M matrix in ω space."
    M0ω::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z1::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z2::Vector{Complex{T1}}

    "Temporary storage vector of length NL."
    z3::Vector{Complex{T1}}

    function RightDiagonalPreconditioner(op::LeftDiagonalPreconditioner{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

        return new{T1,T2}(op.holstein, op.timefreqfft, op.expnΔτV_bar, op.ω_phases, op.M0ω, op.z1, op.z2, op.z3)
    end
end

"""
Get diagonal of M matrix in ω space.
"""
function setup!(op::DiagonalPreconditioner)

    N = op.holstein.nsites
    L = op.holstein.Lτ

    # calculating expnΔτV_bar
    @fastmath @inbounds for i in 1:N
        op.expnΔτV_bar[i] = 0.0
        for τ in 1:L
            op.expnΔτV_bar[i] += op.holstein.expnΔτV[get_index(τ,i,L)]
        end
        op.expnΔτV_bar[i] /= L
    end
    
    # constructing all Diag(M[ω,ω])
    @fastmath @inbounds for i in 1:N
        Bi = op.expnΔτV_bar[i]
        for ω in 1:L
            op.M0ω[get_index(ω,i,L)] = 1.0 - op.ω_phases[ω]*Bi
        end
    end

    return nothing
end

function setup!(op::SplitDiagonalPreconditioner)

    N = op.holstein.nsites
    L = op.holstein.Lτ

    # calculating expnΔτV_bar
    @fastmath @inbounds for i in 1:N
        op.expnΔτV_bar[i] = 0.0
        for τ in 1:L
            op.expnΔτV_bar[i] += op.holstein.expnΔτV[get_index(τ,i,L)]
        end
        op.expnΔτV_bar[i] /= L
    end
    
    # constructing all Diag(M[ω,ω])
    @fastmath @inbounds for i in 1:N
        Bi = op.expnΔτV_bar[i]
        for ω in 1:L
            tmp = 1.0 - op.ω_phases[ω]*Bi
            op.M0ᵀωM0ω[get_index(ω,i,L)] = tmp * conj(tmp)
        end
    end

    return nothing
end


"""
Transpose LeftDiagonalPreconditioner and return RightDiagonalPreconditioner.
"""
function transpose(op::LeftDiagonalPreconditioner)::RightDiagonalPreconditioner

    return RightDiagonalPreconditioner(op)
end

"""
Apply split diagonal preconditioner.
"""
function ldiv!(vout::AbstractVector{T},op::SplitDiagonalPreconditioner,vin::AbstractVector{T}) where {T<:AbstractFloat}

    N  = op.holstein.nsites
    L  = op.holstein.Lτ
    z1 = op.z1
    z2 = op.z2

    # 1. apply phase factor to go from (anit-periodic)⟶(periodic) in τ
    # 2. FFT from τ ⟶ ω
    τ_to_ω!(z1,op.timefreqfft,vin)

    # calculate [2⋅P⁻¹-P⁻¹⋅MᵀM⋅P⁻¹]⋅z where P⁻¹=M0ω⁻¹⋅M0ω⁻ᵀ is a diagonal matrix
    @. z1 /= op.M0ᵀωM0ω
    mul!(z2,op,z1)
    @. z2 /= op.M0ᵀωM0ω
    @. z1  = 2*z1 - z2

    # 1. iFFT from ω ⟶ τ
    # 2. apply inverse phase factor to go from (periodic)⟶(anti-periodic) in τ
    ω_to_τ!(vout,op.timefreqfft,z1)

    return nothing
end


"""
Apply left diagonal preconditioner.
"""
function ldiv!(vout::AbstractVector{T},op::LeftDiagonalPreconditioner,vin::AbstractVector{T}) where {T<:AbstractFloat}

    N  = op.holstein.nsites
    L  = op.holstein.Lτ
    z1 = op.z1
    z2 = op.z2

    # 1. apply phase factor to go from (anit-periodic)⟶(periodic) in τ
    # 2. FFT from τ ⟶ ω
    τ_to_ω!(z1,op.timefreqfft,vin)

    # calculate [2⋅P⁻¹-P⁻¹⋅M⋅P⁻¹]⋅z where P=M0ω is a diagonal matrix
    @. z1 /= op.M0ω
    mul!(z2,op,z1)
    @. z2 /= op.M0ω
    @. z1  = 2*z1 - z2

    # 1. iFFT from ω ⟶ τ
    # 2. apply inverse phase factor to go from (periodic)⟶(anti-periodic) in τ
    ω_to_τ!(vout,op.timefreqfft,z1)

    return nothing
end


"""
Apply right diagonal preconditioner.
"""
function ldiv!(vout::AbstractVector{T},op::RightDiagonalPreconditioner,vin::AbstractVector{T}) where {T<:AbstractFloat}

    N  = op.holstein.nsites
    L  = op.holstein.Lτ
    z1 = op.z1
    z2 = op.z2

    # 1. apply phase factor to go from (anit-periodic)⟶(periodic) in τ
    # 2. FFT from τ ⟶ ω
    τ_to_ω!(z1,op.timefreqfft,vin)

    # apply [2⋅P⁻ᵀ - P⁻ᵀ⋅Mᵀ⋅P⁻ᵀ] where P = M0ω
    @. z1 /= conj(op.M0ω)
    mul!(z2,op,z1)
    @. z2 /= conj(op.M0ω)
    @. z1  = 2*z1 - z2

    # 1. iFFT from ω ⟶ τ
    # 2. apply inverse phase factor to go from (periodic)⟶(anti-periodic) in τ
    ω_to_τ!(vout,op.timefreqfft,z1)

    return nothing
end


function ldiv!(op::DiagonalPreconditioner,v::AbstractVector{T}) where {T<:AbstractFloat}

    ldiv!(v,op,v)
    return nothing
end


"""
Multiply a vector by the block diagonal matrices Mᵀ[ω,ω]⋅M[ω,ω]
"""
function mul!(zout::AbstractVector{T},op::SplitDiagonalPreconditioner,zin::AbstractVector{T}) where {T<:Complex}

    N = op.holstein.nsites
    L = op.holstein.Lτ

    # Multiply by M[ω,ω]
    copyto!(op.z3,zin)

    # multiply by exp{-Δτ⋅K}
    checkerboard_mul!(op.z3, op.holstein.neighbor_table_tij, op.holstein.coshtij, op.holstein.sinhtij)
    
    # multiply by exp{iϕ(ω)}⋅exp{-Δτ⋅V}_bar
    @fastmath @inbounds for i in 1:N
        for ω in 1:L
            op.z3[get_index(ω,i,L)] *= op.ω_phases[ω] * op.expnΔτV_bar[i]
        end
    end

    # get final result
    @. zout = zin - op.z3

    # Multiply by Mᵀ[ω,ω]

    # multiply by exp{-Δτ⋅V}ᵀ_bar⋅exp{iϕ(ω)}ᵀ
    @fastmath @inbounds for i in 1:N
        for ω in 1:L
            indx = get_index(ω,i,L)
            op.z3[indx] = conj(op.expnΔτV_bar[i]) * conj(op.ω_phases[ω]) * zout[indx]
        end
    end

    # multiply by exp{-Δτ⋅K}
    checkerboard_transpose_mul!(op.z3, op.holstein.neighbor_table_tij, op.holstein.coshtij, op.holstein.sinhtij)

    # get final result
    @. zout = zout - op.z3

    return nothing
end


"""
Multiply a vector by the block diagonal matrices M[ω,ω]
"""
function mul!(zout::AbstractVector{T},op::LeftDiagonalPreconditioner,zin::AbstractVector{T}) where {T<:Complex}

    N = op.holstein.nsites
    L = op.holstein.Lτ

    copyto!(op.z3,zin)

    # multiply by exp{-Δτ⋅K}
    checkerboard_mul!(op.z3, op.holstein.neighbor_table_tij, op.holstein.coshtij, op.holstein.sinhtij)
    
    # multiply by exp{iϕ(ω)}⋅exp{-Δτ⋅V}_bar
    @fastmath @inbounds for i in 1:N
        for ω in 1:L
            op.z3[get_index(ω,i,L)] *= op.ω_phases[ω] * op.expnΔτV_bar[i]
        end
    end

    # get final result
    @. zout = zin - op.z3

    return nothing
end


"""
Multiply a vector by the block diagonal matrices Mᵀ[ω,ω]
"""
function mul!(zout::AbstractVector{T},op::RightDiagonalPreconditioner,zin::AbstractVector{T}) where {T<:Complex}

    N = op.holstein.nsites
    L = op.holstein.Lτ
    
    # multiply by exp{-Δτ⋅V}ᵀ_bar⋅exp{iϕ(ω)}ᵀ
    @fastmath @inbounds for i in 1:N
        for ω in 1:L
            indx = get_index(ω,i,L)
            op.z3[indx] = conj(op.expnΔτV_bar[i]) * conj(op.ω_phases[ω]) * zin[indx]
        end
    end

    # multiply by exp{-Δτ⋅K}
    checkerboard_transpose_mul!(op.z3, op.holstein.neighbor_table_tij, op.holstein.coshtij, op.holstein.sinhtij)

    # get final result
    @. zout = zin - op.z3

    return nothing
end

end