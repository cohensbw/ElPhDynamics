using Statistics
using Printf

using ..UnitCells: UnitCell
using ..Lattices: Lattice, sorted_neighbor_table_perm, loc_to_site, calc_neighbor_table
using ..Checkerboard: checkerboard_mul!, checkerboard_transpose_mul!, checkerboard_order, checkerboard_groups
using ..IterativeSolvers: IterativeSolver, GMRES, ConjugateGradient, BiCGStab, solve!
using ..Utilities: get_index

export HolsteinModel
export assign_μ!, assign_ω!, assign_λ!, assign_ω4!
export assign_t!, assign_ωij!
export get_index, get_site, get_τ
export setup_checkerboard!, update_model!
export write_phonons, read_phonons

import Base: eltype, size, length, *
import LinearAlgebra: mul!, ldiv!, transpose!

using LinearAlgebra

mutable struct HolsteinModel{T1,T2,T3} <: AbstractModel{T1,T2,T3}

    ################################
    ## COMPLETE MODEL HAMILTONIAN ##
    ################################

    # H =  ∑ Pᵢ²/2 + ∑ (ωᵢ²/2) xᵢ² [Einstein Phonons]
    #   +  ∑ λᵢ xᵢ nᵢ              [El-Ph Coupling]
    #   +  ∑ ωᵢⱼ(xᵢ ± xⱼ)²         [Phonon Dispersion]
    #   -  ∑ μᵢ nᵢ                 [Chemical Potential]
    #   -  ∑ tᵢⱼ(c⁺ᵢcⱼ + h.c.)     [Electron Kinetic Energy]

    ####################################################################
    ## CHARACTERIZING TEMPERATURE AND SIZE OF D+1 DIMENSIONAL LATTICE ##
    ####################################################################

    "inverse temperature"
    β::T1

    "discretization in imaginary time direction"
    Δτ::T1

    "length of imaginary time axis"
    Lτ::Int

    "number of sites in physical lattice"
    Nsites::Int

    "Number of bonds/hoppings in lattice"
    Nbonds::Int

    "number of phonons in lattice"
    Nph::Int

    "dimension of M matrix"
    Ndim::Int

    "number of degrees of freedom"
    Ndof::Int

    "number of type of bonds in lattice."
    nbonds::Int

    "number of type of phonons in lattice."
    nph::Int

    #######################################
    ## FOR REPRESENTING LATTICE GEOMETRY ##
    #######################################

    "represents lattice"
    lattice::Lattice{T1}

    ############################
    ## HOLSTEIN PHONON FIELDS ##
    ############################

    "phonon fields stored in the order `[x₁(1),...,x₁(Lτ),...,xₙ(1),...,xₙ(Lτ)]`
    where `n` is the number of sites in the lattice and `Lτ` is the length of the imaginary time axis."
    x::Vector{T1}

    ##############################################################
    ## VECTORS REPRESENTING MATRICE NEEDED TO LANGEVIN DYNAMICS ##
    ##############################################################

    "a vector representing the diagonal matrix exp(-Δτ⋅V[x])"
    expnΔτV::Vector{T2}

    #############################################################
    ## SPECIFIES ELECTRON KINETIC ENERGY (TIGHT BINDING MODEL) ##
    #############################################################

    "definitions of bonds in lattice."
    bond_definitions::Vector{Bond{T2}}

    "electron hopping energies in tight binding model"
    t::Vector{T2}

    "checkerboard permutation"
    checkerboard_perm::Vector{Int}

    "cosh of electron hopping parameters in t. Sorted according to checkerboard decomposition."
    cosht::Vector{T2}

    "sinh of electron hopping parameters in t. Sorted according to checkerboard decomposition."
    sinht::Vector{T2}

    "neighboring sites in tight binding model. Sorted according to checkerboard decomposition."
    neighbor_table::Matrix{Int}

    #####################################################
    ## SPECIFIES ON-SITE ENERGIES (CHEMICAL POTENTIAL) ##
    #####################################################

    "chemical potential for each site in lattice"
    μ::Vector{T1}

    ##############################
    ## SPECIFIES HOLSTEIN MODEL ##
    ##############################

    "frequency of each phonon"
    ω::Vector{T1}

    "local electron-phonon coupling"
    λ::Vector{T1}

    "coefficient for anharmonic term X^4"
    ω4::Vector{T1}

    ###################################
    ## SPECIFIES HOLSTEIN DISPERSION ##
    ###################################

    "extended holstein frequency of the form ωij(xᵢ±xⱼ)²"
    ωij::Vector{T1}

    "specifies which two sites i,j that are coupled in ωij(xᵢ±xⱼ)²"
    neighbor_table_ωij::Matrix{Int}

    "specifies the sign: ωij(xᵢ+xⱼ)² or ωij(xᵢ-xⱼ)²"
    sign_ωij::Vector{Int}

    #################################################
    ## VARIBALES FOR SOLVING M⋅x=g VIA ITERATIVELY ##
    #################################################

    "A vector of length `ninidces` to temporarily store data."
    ytemp::Vector{T2}

    "A vector for storing the temporary product Mᵀ⋅g needed for Conjugate Gradient method."
    Mᵀg::Vector{T2}

    "If true the default matrix multiplication uses just the M matrix.
    If false the default matrix multiplication use the symmetric matrix MᵀM instead."
    mul_by_M::Bool

    "If true multiply by Mᵀ instead of M."
    transposed::Bool

    """
    Iterative Solver
    """
    solver::T3

    #######################
    ## INNER CONSTRUCTOR ##
    #######################

    """
    Constructor for Holstein type.
    """
    function HolsteinModel(lattice::Lattice{T1}, β::T1, Δτ::T1;
                           is_complex::Bool=false, iterativesolver::String="cg",
                           tol::T1=1e-4, maxiter::Int=10000, restart::Int=-1) where {T1<:AbstractFloat}

        # define data type of matrix elements of M matrix
        if is_complex
            T2 = Complex{T1}
        else
            T2 = T1
        end

        # calculating length of imaginary time axis
        Lτ = round(Int,β/Δτ)

        # number of sites in lattice
        Nsites = lattice.nsites

        # number of phonons on lattice
        Nph = Nsites

        # number of degrees of freedom
        Ndof = Nph*Lτ

        # dimension of M matrix
        Ndim = Ndof

        # number of bonds in lattice
        Nbonds = 0

        # zero types of bonds initially
        nbonds = 0

        # number of types of phonons in lattice
        nph = lattice.unit_cell.norbits

        # initialize ineraction matrix, which is diagonal so stored as vector
        expnΔτV = zeros(T1,Ndim)

        # intializing phonon fields to zero
        x = zeros(T1,Ndof)

        # initializing all on-site energies to zero
        μ = zeros(T1,Nsites)

        # initializing vector to contain bond definitions
        bond_definitions = Vector{Bond{T2}}(undef,0)

        # initialize hopping parameters to empty vector
        t = Vector{T2}(undef,0)

        # initialize hopping parameters to empty vector
        cosht = Vector{T2}(undef,0)

        # initialize hopping parameters to empty vector
        sinht = Vector{T2}(undef,0)

        # initializing empty matrix to contain tight binding model neighbor_table
        neighbor_table = Matrix{Int}(undef,2,0)

        # checkerboard perumtation
        checkerboard_perm = Vector{Int}(undef,0)

        # intializing phonon frequencies to one
        ω = zeros(T1,Nph)

        # initialize electron-phonon coupling to zero
        λ = zeros(T1,Nph)

        # initialize anharmonic X^4 term coefficient
        ω4 = zeros(T1,Nph)

        # initizlize empty vector for inter-site phonon frequencies
        ωij = Vector{T1}(undef,0)

        # intialize empty matrix for storing inter-site phonon frequency neighbor_table
        neighbor_table_ωij = Matrix{Int}(undef,2,0)

        # intialize empty vector for sign_ωij
        sign_ωij = Vector{Int}(undef,0)

        # temporary vectors
        ytemp = zeros(T2,Ndim)

        # if true multiply by Mᵀ instead of M
        transposed = false

        # temporary vector
        Mᵀg = zeros(T2,Ndim)

        # construct solver
        if lowercase(iterativesolver)=="cg"
            mul_by_M = false
            solver = ConjugateGradient(Mᵀg,tol=tol,maxiter=maxiter)
        elseif lowercase(iterativesolver)=="gmres"
            mul_by_M = true
            solver = GMRES(Mᵀg,tol=tol,restart=restart,maxiter=maxiter)
        elseif lowercase(iterativesolver)=="bicgstab"
            mul_by_M = true
            solver = BiCGStab(Mᵀg,tol=tol,maxiter=maxiter)
        end

        # get data type of iterative solver used
        T3 = typeof(solver)

        # constructing holstein model
        return new{T1,T2,T3}(β,Δτ,Lτ,Nsites,Nbonds,Nph,Ndim,Ndof,nbonds,nph,
                             lattice, x, expnΔτV,
                             bond_definitions, t, checkerboard_perm, cosht, sinht, neighbor_table,
                             μ, ω, λ, ω4, ωij, neighbor_table_ωij, sign_ωij,
                             ytemp, Mᵀg, mul_by_M, transposed, solver)
    end

end

#############################################################################
## DEFINING METHODS TO INCREMENTALLY SPECIFY THE HOLSTEIN MODEL PARAMETERS ##
#############################################################################

"""
Assign chemical potential in model for given type of orbital.
"""
function assign_μ!(holstein::HolsteinModel,μ0::T,σ0::T,orbit::Int=0) where {T<:AbstractFloat}

    if orbit==0
        R = randn(holstein.Nsites)
        @. holstein.μ = μ0 + σ0 * R
    else
        for i in 1:holstein.Nsites
            if holstein.lattice.site_to_orbit[i]==orbit
                holstein.μ[i] = μ0 + σ0 * randn()
            end
        end
    end
        
    return nothing
end

"""
Assign electron-phonon coupling in model for given type of orbital.
"""
function assign_λ!(holstein::HolsteinModel,μ0::T,σ0::T,orbit::Int=0) where {T<:AbstractFloat}

    if orbit==0
        R = randn(holstein.Nph)
        @. holstein.μ = μ0 + σ0 * R
    else
        for i in 1:holstein.Nph
            if holstein.lattice.site_to_orbit[i]==orbit
                holstein.λ[i] = μ0 + σ0 * randn()
            end
        end
    end
        
    return nothing
end

"""
Assign phonon frequency in model.
"""
function assign_ω!(holstein::HolsteinModel,μ0::T,σ0::T,orbit::Int=0) where {T<:AbstractFloat}

    if orbit==0
        R = randn(holstein.Nph)
        @. holstein.ω = μ0 + σ0 * R
    else
        for i in 1:holstein.Nph
            if holstein.lattice.site_to_orbit[i]==orbit
                holstein.ω[i] = μ0 + σ0 * randn()
            end
        end
    end
        
    return nothing
end

"""
Assign coefficient for quartic phonon term.
"""
function assign_ω4!(holstein::HolsteinModel,μ0::T,σ0::T,orbit::Int=0) where {T<:AbstractFloat}

    if orbit==0
        R = randn(holstein.Nsites)
        @. holstein.ω4 = μ0 + σ0 * R
    else
        for i in 1:holstein.Nsites
            if holstein.lattice.site_to_orbit[i]==orbit
                holstein.ω4[i] = μ0 + σ0 * randn()
            end
        end
    end
        
    return nothing
end

"""
Define a type of hopping/bond in the lattice.
"""
function assign_t!(holstein::HolsteinModel{T1,T2,T3}, t::T2, σt::T2, o₁::Int, o₂::Int, v::AbstractVector{Int}) where {T1,T2,T3}

    # increment number of bond definitions
    holstein.nbonds += 1

    # bond definition
    bond = Bond(t,σt,o₁,o₂,v)
    push!(holstein.bond_definitions,bond)

    # getting new neighbors accounting for periodic boundary conditions
    newneighbors = calc_neighbor_table(holstein.lattice,o₁,o₂,v)

    # getting number of new neighbors
    Nnewneighbors = size(newneighbors,2)

    # adding new neighbors to neighbor tables
    holstein.neighbor_table = hcat(holstein.neighbor_table,newneighbors)

    # phase of hopping
    phase = t/abs(t)

    # adding new hopping values
    t_new = @. phase * ( fill(abs(t),Nnewneighbors) + σt*randn(Nnewneighbors) )
    append!(holstein.t, t_new)

    return nothing
end

"""
Define phonon dispersion relationship between phonons.
"""
function assign_ωij!(holstein::HolsteinModel{T1,T2,T3}, μω::T1, σω::T1, sgn::Int, o1::Int, o2::Int, v::AbstractVector{Int}) where {T1,T2,T3}

    # getting new neighbors accounting for periodic boundary conditions
    newneighbors = calc_neighbor_table(holstein.lattice,o1,o2,v)

    # getting number of new neighbors
    Nnewneighbors = size(newneighbors,2)

    # adding new neighbors to neighbor tables
    holstein.neighbor_table = hcat(holstein.neighbor_table,newneighbors)

    # adding new hopping values
    append!(holstein.ωij, fill(μω,Nnewneighbors) + σω*randn(Nnewneighbors) )

    # updating neighbor table and parameter values
    assign_ωij!(holstein,μω,σω,o1,o2,v)

    # modifying holsteinmodel.sign_ωij array
    @assert abs(sgn)==1
    append!( holstein.sign_ωij , fill(Int,sgn,Nnewneighbors) )

    return nothing
end

##########################################################
## FUNCTIONS FOR SPECIFYING AND UPDATING HOLSTEIN MODEL ##
##########################################################

"""
    function setup_checkerboard!(holstein::HolsteinModel{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

Function for sorting the hopping parameters in HolsteinModel and calculating the
cosh and sinh of the hopping parameters so that the checkerboard decomposition
is ready to go.
"""
function initialize_model!(holstein::HolsteinModel)

    if length(holstein.t)>0

        # set number of bonds in lattice
        holstein.Nbonds = length(holstein.t)

        # calculate cosh and sinh of hopping parameters
        holstein.cosht = @. cosh(holstein.Δτ*holstein.t)
        holstein.sinht = @. sinh(holstein.Δτ*holstein.t)

        # sort neighbor_table
        perm = sorted_neighbor_table_perm(holstein.neighbor_table)
        holstein.neighbor_table .= holstein.neighbor_table[:,perm]
        holstein.cosht          .= holstein.cosht[perm]
        holstein.sinht          .= holstein.sinht[perm]

        # get checkerboard groups
        groups = checkerboard_groups(holstein.neighbor_table)

        # get checkerboard permutation
        new_perm = checkerboard_order(groups)

        # applying checkerboard permutation
        holstein.neighbor_table .= holstein.neighbor_table[:,new_perm]
        holstein.cosht          .= holstein.cosht[new_perm]
        holstein.sinht          .= holstein.sinht[new_perm]

        # record checkerboard permutation
        holstein.checkerboard_perm = perm[new_perm]
    end

    return nothing
end

"""
    function update_model!(holstein::HolsteinModel)

Constructs the exponentiated interaction matrix for the Holstein Model
exp(-Δτ⋅V[x]) based on the current phonon fields x. Note that the matrix
exp(-Δτ⋅V[x]) is stored as a vector as it is a diagonal matrix.
"""
function update_model!(holstein::HolsteinModel)

    expnΔτV  = holstein.expnΔτV
    λ        = holstein.λ
    μ        = holstein.μ
    x        = holstein.x
    Δτ       = holstein.Δτ
    Lτ       = holstein.Lτ
    Nsites   = holstein.Nsites

    # iterating over time slices
    @inbounds @fastmath for i in 1:Nsites
        # iterating over sites in lattice
        for τ in 1:Lτ
            # getting index in vector
            index = get_index(τ,i,Lτ)
            # updating matrix element exp{-Δτ⋅Vᵢᵢ(τ)} = exp{-Δτ⋅(λᵢ⋅xᵢ(τ)-μᵢ)}
            expnΔτV[index] = exp( -Δτ * ( λ[i] * x[index] - μ[i] ) )
        end
    end

    return nothing
end

####################################
## MATRIX MULTIPLICATION ROUTINES ##
####################################

"""
Multiply vector by M.
"""
function mulM!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T3},v::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Number,T3<:Number}

    ####################################
    ## PERFORM MULTIPLICATION y = M⋅v ##
    ####################################

    #           EXAMPLE OF M MATRIX CONVENTION
    #     |   1     0      0      0      0   +B(1) |
    #     | -B(2)   1      0      0      0     0   |
    # M = |   0   -B(3)    1      0      0     0   |
    #     |   0     0    -B(4)    1      0     0   |
    #     |   0     0      0    -B(5)    1     0   |
    #     |   0     0      0      0    -B(6)   1   | 

    # Notes:
    # • y(1) = [M⋅v](1) = v(1) + B(1)⋅v(Lτ)  for τ = 1
    # • y(τ) = [M⋅v](τ) = v(τ) - B(τ)⋅v(τ-1) for τ > 1
    # • B(τ) = exp{-Δτ⋅V[x(τ)]}⋅exp{-Δτ⋅K}
    # • exp{-Δτ⋅V[x(τ)]} is the exponentiated interaction matrix and is diagonal,
    #   and as such is stored as a vector
    # • exp{-Δτ⋅K} is given by the checkerboard approximation matrix.

    # y = v
    copyto!(y, v)

    # y(τ) = exp{-Δτ⋅K}⋅v(τ)
    checkerboard_mul!(y, holstein.neighbor_table, holstein.cosht, holstein.sinht, holstein.Lτ)

    # iterating over sites in lattice
    @fastmath @inbounds for i in 1:holstein.Nsites

        # y(1) = v(1) + B(1)⋅v(Lτ)
        idx_L   = get_index(holstein.Lτ, i,  holstein.Lτ)
        idx_1   = get_index(1,           i,  holstein.Lτ)
        y1_temp = v[idx_1] + holstein.expnΔτV[idx_1] * y[idx_L]

        # iterating over time slices
        for τ in holstein.Lτ:-1:2

            # y(τ) = v(τ) - B(τ)⋅v(τ-1) for τ>1
            idx_τ    = get_index(τ,   i, holstein.Lτ)
            idx_τm   = get_index(τ-1, i, holstein.Lτ)
            y[idx_τ] = v[idx_τ] - holstein.expnΔτV[idx_τ] * y[idx_τm]
        end

        # y(1) = v(1) + B(1)⋅v(Lτ)
        y[idx_1] = y1_temp
    end

    # println("Mv")
    # println(y)

    return nothing
end

"""
Multiply vector by Mᵀ.
"""
function mulMᵀ!(y::AbstractVector{T2},holstein::HolsteinModel{T1,T3},v::AbstractVector{T2}) where {T1<:AbstractFloat,T2<:Number,T3<:Number}

    #####################################
    ## PERFORM MULTIPLICATION y = Mᵀ⋅v ##
    #####################################

    # Notes:
    # • y(τ)  = [Mᵀ⋅v](τ)  = v(τ)  - Bᵀ(τ+1)⋅v(τ+1) for τ < Lτ
    # • y(Lτ) = [Mᵀ⋅v](Lτ) = v(Lτ) + Bᵀ(1)⋅v(1)     for τ = Lτ
    # • Bᵀ(τ) = exp{-Δτ⋅K}ᵀ⋅exp{-Δτ⋅V[x(τ)]}ᵀ 
    # • exp{-Δτ⋅V[x(τ)]}ᵀ =exp{-Δτ⋅V[x(τ)]} is the exponentiated diagona interaction
    #   matrix and is diagonal, and as such is stored as a vector
    # • [exp{-Δτ⋅K}]ᵀ is given by adjoint of the checkerboard approximation matrix.

    # iterating over sites in lattice
    @fastmath @inbounds for i in 1:holstein.Nsites

        # iterating over imaginary time slices
        for τ in 1:holstein.Lτ-1

            # y(τ) = -exp{-Δτ⋅V[x(τ+1)]}ᵀ⋅v(τ+1) for τ < Lτ
            idx_τp   = get_index(τ+1, i, holstein.Lτ)
            idx_τ    = get_index(τ,   i, holstein.Lτ)
            y[idx_τ] = -conj(holstein.expnΔτV[idx_τp]) * v[idx_τp]
        end

        # y(Lτ) = +exp{-Δτ⋅V[x(1)]}ᵀ⋅v(1) for τ=Lτ
        idx_L    = get_index(holstein.Lτ, i, holstein.Lτ)
        idx_1    = get_index(1,           i, holstein.Lτ)
        y[idx_L] = conj(holstein.expnΔτV[idx_1]) * v[idx_1]
    end

    # y(τ) = -Bᵀ(τ+1)⋅v(τ+1) for τ < Lτ
    # y(τ) = +Bᵀ(τ+1)⋅v(τ+1) for τ = Lτ 
    checkerboard_transpose_mul!(y, holstein.neighbor_table, holstein.cosht, holstein.sinht, holstein.Lτ)

    # y(τ) = v(τ) - Bᵀ(τ+1)⋅v(τ+1) for τ < Lτ
    # y(τ) = v(τ) + Bᵀ(τ+1)⋅v(τ+1) for τ = Lτ
    @. y = v + y

    # println("Mᵀv")
    # println(y)

    return nothing
end


"""
Calculates ⟨∂M/∂xᵢⱼ(τ)⟩ = uᵀ⋅[∂M/∂xᵢⱼ(τ)]⋅v for each xᵢⱼ(τ) degree of freedom,
writing each of the ⟨∂M/∂xᵢⱼ(τ)⟩ values to a vector dMdx.
"""
function muldMdx!(dMdx::AbstractVector{T2},u::AbstractVector{T2},holstein::HolsteinModel{T1,T2,T3},v::AbstractVector{T2}) where {T1,T2,T3}

    ########################################
    ## PERFORM MULTIPLICATION y = ∂M/∂x⋅v ##
    ########################################

    # Notes:
    # • Consider y = ∂M/∂xᵢ(τ)⋅v ==>
    #
    # • yᵢ(τ) = -∂B/∂xᵢ(τ)⋅vᵢ(τ-1) for τ > 1
    # • yᵢ(1) = +∂B/∂xᵢ(1)⋅vᵢ(Lτ)  for τ = 1
    #
    # • B(τ) = exp{-Δτ⋅V[x(τ)]}⋅exp{-Δτ⋅K}
    # • ∂B/∂xᵢ(τ) = -Δτ⋅dV/dxᵢ(τ)⋅exp{-Δτ⋅V[x(τ)]}⋅exp{-Δτ⋅K}
    # • ∂B/∂xᵢ(τ) = -Δτ⋅   λᵢ    ⋅exp{-Δτ⋅V[x(τ)]}⋅exp{-Δτ⋅K}
    #
    # • Therefore the final expression is:
    # • yᵢ(τ-1) = +Δτ⋅λᵢ⋅exp{-Δτ⋅V[x(τ)]}⋅exp{-Δτ⋅K}⋅vᵢ(τ) for τ > 1
    # • yᵢ(Lτ)  = -Δτ⋅λᵢ⋅exp{-Δτ⋅V[x(1)]}⋅exp{-Δτ⋅K}⋅vᵢ(1) for τ = 1
    #
    # • Simplifying a little bit:
    # • yᵢ(τ) = +Δτ⋅λᵢ⋅B(τ)⋅vᵢ(τ-1) for τ > 1
    # • yᵢ(1) = -Δτ⋅λᵢ⋅B(1)⋅vᵢ(Lτ)  for τ = 1

    # ⟨∂M/∂xᵢⱼ(τ)⟩ = v(τ)
    copyto!(dMdx, v)

    # ⟨∂M/∂xᵢⱼ(τ)⟩ = exp{-Δτ⋅K}⋅v(τ)
    checkerboard_mul!(dMdx, holstein.neighbor_table, holstein.cosht, holstein.sinht, holstein.Lτ)
    
    # iterating over sites in lattice
    @fastmath @inbounds for i in 1:holstein.Nsites

        # ⟨∂M/∂xᵢⱼ(τ)⟩ = -Δτ⋅λᵢ⋅B(1)⋅v(1) for τ=1
        idx_1 = get_index(1,           i,  holstein.Lτ)
        idx_L = get_index(holstein.Lτ, i,  holstein.Lτ)
        dMdx1_temp = -holstein.Δτ * holstein.λ[i] * holstein.expnΔτV[idx_1] * dMdx[idx_L]

        # iterating over time slices
        for τ in holstein.Lτ:-1:2

            # ⟨∂M/∂xᵢⱼ(τ)⟩ = +Δτ⋅λ⋅B(τ)⋅v(τ) for τ>1
            idx_τ   = get_index(τ,   i, holstein.Lτ)
            idx_τm1 = get_index(τ-1, i, holstein.Lτ)
            dMdx[idx_τ] = holstein.Δτ * holstein.λ[i] * holstein.expnΔτV[idx_τ] * dMdx[idx_τm1]
        end

        # ⟨∂M/∂xᵢⱼ(τ)⟩ = -Δτ⋅λ⋅B(1)⋅v(Lτ) for τ=1
        dMdx[idx_1] = dMdx1_temp
    end

    # finalize calculation of ⟨∂M/∂xᵢⱼ(τ)⟩ = uᵀ⋅[∂M/∂xᵢⱼ(τ)]⋅v
    @. dMdx *= u

    # println("dMdx")
    # println(dMdx)

    return nothing
end

##############################
## PHONON FIELD IO ROUTINES ##
##############################

"""
Writes the current phonon field configuration for a HolsteinModel to file.
"""
function write_phonons(holstein::HolsteinModel,filename::String)

    # get lattice associated with holstein model
    lattice = holstein.lattice

    # get phonon fields
    x = holstein.x

    # open file
    open(filename,"w") do file

        # write header to file
        write(file, "tau orbit L1 L2 L3 x\n")

        # iterate over unit cells
        for l3 in 0:lattice.L3-1
            for l2 in 0:lattice.L2-1
                for l1 in 0:lattice.L1-1

                    # iterate over orbitals/sites in each unit cell
                    for orbit in 1:lattice.unit_cell.norbits

                        # get site in lattice
                        site = loc_to_site(lattice,orbit,l1,l2,l3)

                        # iterate of time slice
                        for τ in 1:holstein.Lτ

                            # get index
                            i = get_index(τ, site, holstein.Lτ)
                            
                            # get phonon field for site
                            xi = x[i]

                            # write to file
                            write(file, @sprintf("%d %d %d %d %d %.6f\n",τ-1,orbit,l1,l2,l3,xi))
                        end
                    end
                end
            end
        end
    end

    return nothing
end

"""
Read phonon config from file.
"""
function read_phonons(holstein::HolsteinModel{T1,T2,T3},filename::String) where {T1<:AbstractFloat,T2<:Number,T3<:IterativeSolver}

    # open file
    open(filename,"r") do file

        # read in header
        header = readline(file)

        # iterate over lines in file
        for line in eachline(file)

            # split line at white space
            atoms = split(line," ")

            # extract info about location
            τ     = parse(Int,atoms[1]) + 1
            orbit = parse(Int,atoms[2])
            l1    = parse(Int,atoms[3])
            l2    = parse(Int,atoms[4])
            l3    = parse(Int,atoms[5])

            # get phonon field value
            xi = parse(T1,atoms[6])

            # map location on site
            site = loc_to_site(holstein.lattice, orbit, l1, l2, l3)

            # map site and τ onto index
            i = get_index(τ, site, holstein.Lτ)

            # assign phonon value
            holstein.x[i] = xi
        end

    end

    # construct exponentiated interaction matrix
    update_model!(holstein)

    return nothing
end