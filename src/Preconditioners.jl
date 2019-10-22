module Preconditioners


using FFTW
using LinearAlgebra
using IterativeSolvers

using ..HolsteinModels: HolsteinModel, mulM!
using ..Checkerboard: checkerboard_mul!



######################################################################################
# GMRES that can restart without reallocation

using Printf
using IterativeSolvers: GMRESIterable, reserve!, gmres_iterable!, init!, init_residual!, converged

function reset_gmres_iterable!(g::GMRESIterable, new_x, new_A, new_b;
        tol = nothing,
        initially_zero::Bool = nothing)
    @assert size(g.x, 1) == size(new_x, 1) == size(new_A, 2) == size(new_b, 1)
    @assert size(g.arnoldi.A, 1) == size(new_A, 1)

    # Reset solution vector and RHS
    g.x = new_x
    g.b = new_b

    # I want to do this:
    #     g.arnoldi.A = new_A
    #
    # However, 'arnoldi' field is immutable. For now, simply assert that
    # the pointer to A is unchanged (although internally it may change)
    @assert g.arnoldi.A == new_A

    fill!(g.arnoldi.V, zero(eltype(new_x)))
    fill!(g.arnoldi.H, zero(eltype(new_x)))

    # Reset size of Krylov space
    g.k = 1

    # One matrix-vector product so far
    g.mv_products = initially_zero ? 1 : 0

    # Set the first basis vector
    g.residual.current = init!(g.arnoldi, g.x, g.b, g.Pl, g.Ax, initially_zero=initially_zero)
    init_residual!(g.residual, g.residual.current)

    # Set the tolerance for the relative residual
    g.reltol = tol * g.residual.current

    # Figuring out this line took me 6 hours...
    g.β = g.residual.current

    nothing
end

function run_gmres_iterable!(g::GMRESIterable; verbose=false):: Bool
    for (iteration, residual) = enumerate(g)
        verbose && @printf("%3d\t%3d\t%1.2e\n", 1 + div(iteration - 1, g.restart), 1 + mod(iteration - 1, g.restart), residual)
    end

    converged(g)
end


######################################################################################
# Multiplication by full M matrix.
# This struct can be replaced by mul!(r_out, HolsteinModel, r) in the future.

struct MatrixMOp
    holstein :: HolsteinModel{Float64, Float64}
end

function Base.size(op::MatrixMOp, d)
    size(op.holstein)[d]
end

function LinearAlgebra.mul!(r_out, op::MatrixMOp, r)
   mulM!(r_out, op.holstein, r)
end


######################################################################################
# Mtilde(ω) is the NxN block matrix appearing when M_{τ,τ'} is expressed in the ω basis.

mutable struct MtildeBlockOp
    "Diagonal index for this block matrix, ω = 1..L"
    ω :: Int
    
    "Holstein model"
    holstein :: HolsteinModel{Float64, Float64}
    
    "τ-averaged matrix exp(-Δτ⋅V[ϕ])"
    expnΔτV_bar :: Vector{Float64}
    
    "Array of complex phases for each ω"
    phases :: Vector{ComplexF64}

    "Temporary storage of length `L`"
    z1 :: Vector{ComplexF64}
    z2 :: Vector{ComplexF64}
    
    function MtildeBlockOp(ω, holstein)
        N = holstein.nsites
        L = holstein.Lτ
        
        expnΔτV_bar = dropdims(sum(reshape(holstein.expnΔτV, (L, N)); dims=1); dims=1) / L
        
        phases = [exp(2π*im*((ω-1)+1/2)/L) for ω = 1:L]

        z1 = zeros(ComplexF64, N)
        z2 = zeros(ComplexF64, N)
        
        new(ω, holstein, expnΔτV_bar, phases, z1, z2)
    end
end

function Base.eltype(::Type{MtildeBlockOp})
    ComplexF64
end

function Base.size(op::MtildeBlockOp, d)
    op.holstein.nsites
end

function LinearAlgebra.mul!(z_out, op::MtildeBlockOp, z)
    N = op.holstein.nsites
    
    op.z1 .= z
    checkerboard_mul!(
        op.z1, op.holstein.neighbor_table_tij, op.holstein.coshtij, op.holstein.sinhtij, 1)
    
    @. op.z1 = op.phases[op.ω] * op.expnΔτV_bar * op.z1
    
    @. z_out = z - op.z1
end

function Base.:*(op::MtildeBlockOp, z)
    z_out = complex(similar(z))
    mul!(z_out, op, z)
end

function construct_matrix(op::MtildeBlockOp)
    N = op.holstein.nsites
    out = Array{ComplexF64}(I, N, N)
    for j = 1:N
        col = view(out, :, j)
        mul!(col, op, col)
    end
    out
end


######################################################################################
# Block diagonal (Jacobi) preconditioner applied in Fourier basis, ω

const BlockGMRESIterable = IterativeSolvers.GMRESIterable{
    Identity,Identity,
    SubArray{ComplexF64,
            1,Array{ComplexF64,2},
            Tuple{Base.Slice{Base.OneTo{Int64}},Int64},true},
    SubArray{ComplexF64,1,
            Array{ComplexF64,2},
            Tuple{Base.Slice{Base.OneTo{Int64}},Int64},true},
    Array{ComplexF64,1},
    IterativeSolvers.ArnoldiDecomp{ComplexF64,MtildeBlockOp},
    IterativeSolvers.Residual{ComplexF64,Float64},Float64
}

mutable struct BlockPreconditioner
    "Holstein model"
    holstein :: HolsteinModel{Float64, Float64}
    
    "Block matrix multiplication"
    mtilde :: MtildeBlockOp
    
    "Tolerance of GMRES sub-solver"
    subtol :: Float64

    "Array of complex phases for each ω"
    phases :: Vector{ComplexF64}

    "Temporary storage of size (L, N)"
    z1 :: Array{ComplexF64, 1}
    z2 :: Array{ComplexF64, 1}
    
    "Reuse of GMRES storage"
    block_gmres :: Union{Nothing, BlockGMRESIterable}

    "Plan for Fourier transform"
    plan :: FFTW.cFFTWPlan{ComplexF64,-1,false,2}
    
    
    function BlockPreconditioner(holstein; subtol=1e-1)
        L = holstein.Lτ
        N = holstein.nsites
        
        mtilde = MtildeBlockOp(1, holstein)
    
        phases = [exp(-π*im*(τ-1)/L) for τ = 1:L]
        
        z1 = zeros(ComplexF64, L*N)
        z2 = zeros(ComplexF64, L*N)

        plan = plan_fft(reshape(z1, (L, N)), (1,), flags=FFTW.PATIENT)
        
        new(holstein, mtilde, subtol, phases, z1, z2, nothing, plan)
    end
end


function Base.eltype(::Type{BlockPreconditioner})
    Float64
end


function Base.size(op::BlockPreconditioner, d)
    op.holstein.Lτ * op.holstein.nsites
end


function LinearAlgebra.ldiv!(r_out, op::BlockPreconditioner, r)
    mvps_total = 0
    
    L = op.holstein.Lτ
    N = op.holstein.nsites

    z1 = reshape(op.z1, (L, N))
    z1t = reshape(op.z1, (N, L))

    z2 = reshape(op.z2, (L, N))
    z2t = reshape(op.z2, (N, L))

    @. op.z1 = complex(r)
    
    # apply Θ phase
    for i = 1:N
        for τ = 1:L
            z1[τ, i] = z1[τ, i] * op.phases[τ]
        end
    end
    
    # transform basis τ → ω by applying F
    mul!(z2, op.plan, z1)
    
    transpose!(z1t, z2)
    
    # apply Mtilde_{ω, ω}
    for ω = 1 : cld(L, 2)
        z1tv = view(z1t, :, ω)
        z2tv = view(z2t, :, ω)
    
        fill!(z2tv, 0) # TODO: find better initial guess?
        op.mtilde.ω = ω
        
        mvps = 0
        if false
            history = IterativeSolvers.gmres!(z2tv, op.mtilde, z1tv, tol=op.subtol, maxiter=1000, restart=5, log=true)

            # history = IterativeSolvers.bicgstabl!(z2tv, op.mtilde, z1tv, 4, tol=op.subtol, max_mv_products=1000, log=true)

            # Mtilde = construct_matrix(op.mtilde)
            # history = IterativeSolvers.cg!(z2tv, Mtilde' * Mtilde, Mtilde' * z1tv, tol=op.subtol, maxiter=100, log=true)

            mvps = history[2].mvps + history[2].mtvps
            @assert history[2].isconverged "Krylov inversion for ω=$ω did not converge to tolerance $(op.subtol) after $mvps mat-vec products"
        else
            if isnothing(op.block_gmres)
                op.block_gmres = gmres_iterable!(z2tv, op.mtilde, z1tv; tol=op.subtol, maxiter=1000, restart=5, initially_zero=true)
            else
                reset_gmres_iterable!(op.block_gmres, z2tv, op.mtilde, z1tv; tol=op.subtol, initially_zero=true)
            end
            converged = run_gmres_iterable!(op.block_gmres)
            mvps = op.block_gmres.mv_products
            @assert converged "Krylov inversion for ω=$ω did not converge to tolerance $(op.subtol) after $mvps mat-vec products"
        end

        # println(mvps)
        mvps_total += mvps

        for i = 1:N
            z2t[i, L-ω+1] = conj(z2t[i, ω])
        end
    end
    
    transpose!(z1, z2t)
    
    # transform basis ω → τ by applying F†
     ldiv!(z2, op.plan, z1)

    # apply Θ† phase
    for i = 1:N
        for τ = 1:L
            z2[τ, i] *= conj(op.phases[τ])
        end
    end

    @assert norm(imag(z2)) < 1e-12 "Imaginary component imag(z2)=$(norm(imag(z2))) too large."
    
    # println("Effective mat-vec products: ", mvps_total/L)
    
    @. r_out = real(op.z2)
end

function LinearAlgebra.ldiv!(op::BlockPreconditioner, r)
    ldiv!(r, op, r)
end

end
