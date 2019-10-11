module Preconditioners


using FFTW
using LinearAlgebra
using IterativeSolvers

import LinearAlgebra: mul!, ldiv!
import Base: eltype, size

using ..HolsteinModels: HolsteinModel, mulM!
using ..Checkerboard: checkerboard_mul!


######################################################################################
# Multiplication by full M matrix.
# This struct can be replaced by mul!(r_out, HolsteinModel, r) in the future.

struct MatrixMOp
    holstein :: HolsteinModel
end

function Base.size(op::MatrixMOp, d)
    size(op.holstein)[d]
end

function mul!(r_out, op::MatrixMOp, r)
   mulM!(r_out, op.holstein, r)
end


######################################################################################
# Mtilde(ω) is the NxN block matrix appearing when M_{τ,τ'} is expressed in the ω basis.

mutable struct MtildeBlockOp
    "Diagonal index for this block matrix, ω = 1..L"
    ω :: Int
    
    "Holstein model"
    holstein :: HolsteinModel
    
    "τ-averaged matrix exp(-Δτ⋅V[ϕ])"
    expnΔτV_bar :: Vector{Float64}
    
    "Array of complex phases for each ω"
    phases :: Vector{Complex{Float64}}
    
    "Temporary storage of length `L`"
    z1 :: Vector{Complex{Float64}}
    z2 :: Vector{Complex{Float64}}
    
    function MtildeBlockOp(ω, holstein)
        N = holstein.nsites
        L = holstein.Lτ
        
        expnΔτV_bar = dropdims(sum(reshape(holstein.expnΔτV, (L, N)); dims=1); dims=1) / L
        
        phases = [exp(-2π*im*((ω-1)+1/2)/L) for ω = 1:L]
        
        z1 = zeros(Complex{Float64}, N)
        z2 = zeros(Complex{Float64}, N)
        
        new(ω, holstein, expnΔτV_bar, phases, z1, z2)
    end
end

function Base.eltype(::Type{MtildeBlockOp})
    Complex{Float64}
end

function Base.size(op::MtildeBlockOp, d)
    op.holstein.nsites
end

function mul!(z_out, op::MtildeBlockOp, z)
    N = op.holstein.nsites
    
    op.z1 .= z
    checkerboard_mul!(
        op.z1, op.holstein.neighbor_table_tij, op.holstein.coshtij, op.holstein.sinhtij, 1)
    
    @. op.z1 = op.phases[op.ω] * op.expnΔτV_bar * op.z1
    
    @. z_out = z - op.z1
end


######################################################################################
# Block diagonal (Jacobi) preconditioner applied in Fourier basis, ω

mutable struct BlockPreconditioner
    "Holstein model"
    holstein :: HolsteinModel
    
    "Block matrix multiplication"
    mtilde :: MtildeBlockOp
    
    "Tolerance of GMRES sub-solver"
    subtol :: Float64

    "Array of complex phases for each ω"
    phases :: Vector{Complex{Float64}}

    "Temporary storage of size (L, N)"
    z1 :: Array{Complex{Float64}, 1}
    z2 :: Array{Complex{Float64}, 1}
    
    plan :: FFTW.cFFTWPlan
    
    
    function BlockPreconditioner(holstein; subtol=1e-1)
        L = holstein.Lτ
        N = holstein.nsites
        
        mtilde = MtildeBlockOp(1, holstein)
    
        phases = [exp(π * im * (τ-1) / L) for τ = 1:L]
        
        z1 = zeros(Complex{Float64}, L*N)
        z2 = zeros(Complex{Float64}, L*N)

        plan = plan_fft(reshape(z1, (L, N)), (1,), flags=FFTW.PATIENT)
        
        new(holstein, mtilde, subtol, phases, z1, z2, plan)
    end
end


function Base.eltype(::Type{BlockPreconditioner})
    Float64
end


function Base.size(op::BlockPreconditioner, d)
    op.holstein.Lτ * op.holstein.nsites
end


function ldiv!(r_out, op::BlockPreconditioner, r)
    mvps_total = 0
    
    L = op.holstein.Lτ
    N = op.holstein.nsites

    @. op.z1 = complex(r)

    z1 = reshape(op.z1, (L, N))
    z1t = reshape(op.z1, (N, L))

    z2 = reshape(op.z2, (L, N))
    z2t = reshape(op.z2, (N, L))
    
    # apply Θ phase
    for i = 1:N
        @. z1[:, i] = z1[:, i] * op.phases[:]
    end
    
    # transform basis τ → ω by applying F
    # (whoops, this is an inverse FFT because I got sign backwards in my notes...)
    ldiv!(z2, op.plan, z1)
    
    transpose!(z1t, z2)
        
    # apply Mtilde_{ω, ω}
    for ω = 1:L
        z1tv = @view z1t[:, ω]
        z2tv = @view z2t[:, ω]
    
        fill!(z2tv, 0)
        
        op.mtilde.ω = ω
        
        # mul!(z2tv, op.mtilde, z1tv)
        
        history = IterativeSolvers.gmres!(z2tv, op.mtilde, z1tv, tol=op.subtol, maxiter=100, restart=5, log=true)
#        history = IterativeSolvers.gmres!(temp2, op.mtilde, temp1, tol=1e-2, log=true, maxiter=100, restart=5)
        @assert history[2].isconverged
        mvps_total += history[2].mvps
        
        #IterativeSolvers.bicgstabl(op.mtilde, r, 2, tol=1e-2, log=true, max_mv_products=1000)
    end
    
    transpose!(z1, z2t)
    
    # transform basis ω → τ by applying F†
    mul!(z2, op.plan, z1)

    # apply Θ† phase
    for i = 1:N
        @. z2[:, i] = z2[:, i] * conj(op.phases[:])
    end

    @assert norm(imag(z2)) < 1e-10
    
    println("Effective mat-vec products: ", mvps_total/L)
    
    @. r_out = real(z2[:])
end

function ldiv!(op::BlockPreconditioner, r)
    ldiv!(r, op, r)
end

end
