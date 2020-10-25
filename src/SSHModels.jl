using Parameters
using LinearAlgebra

import LinearAlgebra: mul!, ldiv!, transpose!

using ..UnitCells: UnitCell
using ..Lattices: Lattice, sorted_neighbor_table_perm!, loc_to_site, calc_neighbor_table
using ..Checkerboard: checkerboard_order, checkerboard_groups, checkerboard_mul!, checkerboard_transpose_mul!
using ..IterativeSolvers: GMRES, ConjugateGradient, BiCGStab
using ..Utilities: get_index, get_τ, δ, reshaped

struct SSHBond{T1<:AbstractFloat,T2<:Continuous} <: AbstractBond

    "Average hopping energy."
    t::T2

    "Standard deviation hopping energy."
    σt::T2

    "Average phonon frequency"
    ω::T1

    "Standard deviation phonon frequency."
    σω::T1

    "Anharmonic term coefficient for ω₄⋅X⁴ term."
    ω₄::T1

    "Standard deviation in ω₄"
    σω₄::T1

    "Average linear electron-phonon coupling"
    α::T1

    "Standard deviation linear electron-phonon coupling"
    σα::T1

    "Non-linear electron phonon coupling of the form α₂⋅X²."
    α₂::T1

    "Standard deviation in α₂."
    σα₂::T1

    "orbital"
    o₁::Int

    "orbital"
    o₂::Int

    "displacement in unit cells"
    v::Vector{Int}

    "whether or not a phonon lives on the bond."
    has_phonon::Bool

    function SSHBond(t::T2,σt::T2,ω::T1,σω::T1,ω₄::T1,σω₄::T1,α::T1,σα::T1,α₂::T1,σα₂::T1,
                     o₁::Int,o₂::Int,v::AbstractVector{Int}) where {T1<:AbstractFloat,T2<:Continuous}

        @assert length(v)==3
        v′ = zeros(Int,3)
        copyto!(v′,v)
        has_phonon = !iszero(ω)||!iszero(σω)
        return new{T1,T2}(t,σt,ω,σω,ω₄,σω₄,α,σα,α₂,σα₂,o₁,o₂,v′,has_phonon)
    end
end

mutable struct SSHModel{T1,T2,T3} <: AbstractModel{T1,T2,T3}

    "Inverse temperature"
    β::T1

    "Discretization in imaginary time direction"
    Δτ::T1

    "Length of imaginary time axis"
    Lτ::Int

    "Number of sites in physical lattice"
    Nsites::Int

    "Number of bonds/hoppings in lattice"
    Nbonds::Int

    "Number of phonons in lattice"
    Nph::Int

    "Dimension of M matrix"
    Ndim::Int

    "Number of degrees of freedom"
    Ndof::Int

    "Number of types of bonds in lattice."
    nbonds::Int

    "Number of types of phonons in lattice."
    nph::Int

    "Represents lattice"
    lattice::Lattice{T1}

    "Bond defintions, including description of phonon living on the bond if there is one."
    bond_definitions::Vector{SSHBond{T1,T2}}

    "Ndof=Nph*Lτ phonon fields. Ordered in vector by phonon defintion."
    x::Vector{T1}

    "Nbond bare electron hoppings."
    t::Vector{T2}

    "Nph phonon frequencies. Ordered in vector by phonon defintion."
    ω::Vector{T1}

    "Anharmonic phonon term ω₄⋅X⁴."
    ω₄::Vector{T1}

    "Nph electron-phonon couplings. Ordered in vector by phonon defintion."
    α::Vector{T2}

    "Non-linear electron-phonon coupling."
    α₂::Vector{T2}

    "Chemical potential"
    μ::Vector{T1}

    "Diagonal matrix exp{Δτ⋅μ}"
    expΔτμ::Vector{T1}

    """
    Modulated hoppings t′ = t-α⋅X.
    """
    t′::Matrix{T2}

    "Map field to phonon"
    field_to_phonon::Vector{Int}

    "Map Ndof phonon fields onto a time slice τ"
    field_to_τ::Vector{Int}

    "Map phonon to bond"
    phonon_to_bond::Vector{Int}

    "Map bond to phonon. If 0 then no phonon to map to."
    bond_to_phonon::Vector{Int}

    "Maps a given phonon onto an SSHPhonon defintion."
    bond_to_definition::Vector{Int}

    """
    Checkerboard bond order. Maps bond onto neighbor table entry.
    """
    checkerboard_perm::Vector{Int}

    "Neighbor table telling which sites in lattice are connect by bonds.
    Ordered according to checkerboard decomposition."
    neighbor_table::Matrix{Int}

    "Matrix of dimensions (Lτ,Nbonds) containing cosh(t′) where t′=Δτ(t-α⋅x).
    Ordered in vector according to checkerboard decomposition."
    cosht::Matrix{T2}

    "Matrix of dimensions (Lτ,Nbonds) containing sinh(t′) where t′=Δτ(t-α⋅x).
    Ordered in vector according to checkerboard decomposition."
    sinht::Matrix{T2}

    "If true the default mul! routine multiplies by the M (or Mᵀ) matrix.
    If false the default mul! routine multiplies by the symmetric matrix MᵀM (or MMᵀ) instead."
    mul_by_M::Bool

    "If true multiply by Mᵀ (or MMᵀ) instead of M (or MᵀM)."
    transposed::Bool

    "A vector of length `Ndim` to temporarily store data.
    Used in multiplication by MᵀM or MMᵀ."
    v′::Vector{T2}

    "A vector of length `Ndim` to temporarily store data.
    Used to store Mᵀ⋅b result when solving M⋅x=b via MᵀM⋅x=Mᵀ⋅b."
    v″::Vector{T2}

    "A vector of length `Ndim` to temporarily store data.
    Used for calculating true residual error after linear solve."
    v‴::Vector{T2}

    """
    Iterative Solver
    """
    solver::T3

    function SSHModel(lattice::Lattice{T1}, β::T1, Δτ::T1;
                     is_complex::Bool=false, iterativesolver::String="cg",
                     tol::T1=1e-4, maxiter::Int=10000, restart::Int=-1) where {T1<:AbstractFloat}

        # determines data type of matrix elements of M
        if is_complex
            T2 = Complex{T1}
        else
            T2 = T1
        end

        # length of imaginary time axis
        Lτ = round(Int,β/Δτ)

        # number of sites in lattice
        Nsites = lattice.nsites

        # dimension of M matrix
        Ndim = Nsites*Lτ

        # zero phonons defined initially
        Nph = 0

        # zero degrees of freedom initially
        Ndof = 0

        # zero bonds in lattice initially
        Nbonds = 0

        # zero types of bonds initially
        nbonds = 0

        # zero types of phonons initially
        nph = 0

        # intialize vector to contain phonon defintions
        bond_definitions = Vector{SSHBond{T1,T2}}(undef,0)

        # initialize arrays
        x                  = Vector{T1}(undef,0)
        t                  = Vector{T2}(undef,0)
        ω                  = Vector{T1}(undef,0)
        ω₄                 = Vector{T1}(undef,0)
        α                  = Vector{T2}(undef,0)
        α₂                  = Vector{T2}(undef,0)
        μ                  = zeros(T1,Nsites)
        expΔτμ             = ones(T1,Nsites)
        t′                 = Matrix{T2}(undef,Lτ,0)
        checkerboard_perm  = Vector{Int}(undef,0)
        neighbor_table     = Matrix{Int}(undef,2,0)
        cosht              = Matrix{T2}(undef,Lτ,0)
        sinht              = Matrix{T2}(undef,Lτ,0)
        field_to_phonon    = Vector{Int}(undef,0)
        field_to_τ         = Vector{Int}(undef,0)
        phonon_to_bond     = Vector{Int}(undef,0)
        bond_to_phonon     = Vector{Int}(undef,0)
        bond_to_definition = Vector{Int}(undef,0)

        # declaring temporary storage vectors
        v′ = zeros(T2,Ndim)
        v″ = zeros(T2,Ndim)
        v‴ = zeros(T2,Ndim)

        # construct solver
        if lowercase(iterativesolver)=="cg"
            mul_by_M   = false
            transposed = false
            solver = ConjugateGradient(v″,tol=tol,maxiter=maxiter)
        elseif lowercase(iterativesolver)=="gmres"
            mul_by_M   = true
            transposed = false
            solver = GMRES(v″,tol=tol,restart=restart,maxiter=maxiter)
        elseif lowercase(iterativesolver)=="bicgstab"
            mul_by_M   = true
            transposed = false
            solver = BiCGStab(v″,tol=tol,maxiter=maxiter)
        end
        T3 = typeof(solver)

        return new{T1,T2,T3}(β,Δτ,Lτ,Nsites,Nbonds,Nph,Ndim,Ndof,nbonds,nph,
                             lattice, bond_definitions,
                             x,t,ω,ω₄,α,α₂,μ,expΔτμ,t′,
                             field_to_phonon,field_to_τ,phonon_to_bond,bond_to_phonon,bond_to_definition,
                             checkerboard_perm,neighbor_table,cosht,sinht,
                             mul_by_M, transposed, v′, v″, v‴, solver)
    end
end

"""
Assign hopping (and phonon) in SSH model.
"""
function assign_hopping!(ssh::SSHModel{T1,T2},t::T2,σt::T1,ω::T1,σω::T1,ω₄::T1,σω₄::T1,α::T1,σα::T1,α₂::T1,σα₂::T1,
                         o₁::Int,o₂::Int,v::Vector{Int}) where {T1<:AbstractFloat,T2<:Continuous}

    # define ssh bond
    sshbond = SSHBond(t,σt,ω,σω,ω₄,σω₄,α,σα,α₂,σα₂,o₁,o₂,v)
    push!(ssh.bond_definitions,sshbond)

    return nothing
end

"""
Assign chemical potential in SSH model.
"""
function assign_μ!(ssh::SSHModel,μ::T1,σμ::T1,orbit::Int) where {T1<:AbstractFloat}

    site_to_orbit = ssh.lattice.site_to_orbit::Vector{Int}
    for i in 1:ssh.Nsites
        if orbit==0 || orbit==site_to_orbit[i]
            ssh.μ[i]      = μ + σμ * randn()
            ssh.expΔτμ[i] = exp( ssh.Δτ * ssh.μ[i] )
        end
    end

    return nothing
end

"""
Initialize SSH Model.
"""
function initialize_model!(ssh::SSHModel{T1,T2}) where {T1,T2}

    # number of types of bonds in lattice
    ssh.nbonds = length(ssh.bond_definitions)

    # initially zero types of phonons
    ssh.nph = 0

    # iterate over types of bonds in lattices
    for i in 1:ssh.nbonds

        # getting parameters that define bond
        @unpack t, σt, α, σα, α₂, σα₂, ω, σω, ω₄, σω₄, o₁, o₂, v, has_phonon = ssh.bond_definitions[i]

        # calculate new neighbors
        new_neighbors = calc_neighbor_table(ssh.lattice,o₁,o₂,v)

        # number of new bonds
        Nnewbonds = size(new_neighbors,2)

        # adding new neighbors to neighbor tables for current bond defintion
        ssh.neighbor_table = hcat(ssh.neighbor_table, new_neighbors)

        # phase of hopping
        phase = t/abs(t)

        # adding new hopppings for current bond
        t_new = @. phase * ( fill(abs(t),Nnewbonds) + σt*randn(Nnewbonds) )
        append!(ssh.t, t_new)

        # adding bond to definition mapping
        append!(ssh.bond_to_definition, fill(i,Nnewbonds))

        # if bond has phonon on it
        if has_phonon

            # incrementing count of number of types of phonons
            ssh.nph += 1

            # adding new phonon frequencies
            ω_new = @. fill(ω,Nnewbonds) + σω*randn(Nnewbonds)
            append!(ssh.ω, ω_new)

            # adding anharmonic phonon coefficient
            ω₄_new = @. fill(ω₄,Nnewbonds) + σω₄*randn(Nnewbonds)
            append!(ssh.ω₄, ω₄_new)

            # adding new linear electron-phonon coupling
            α_new = @. phase * ( fill(α,Nnewbonds) + σα*randn(Nnewbonds) )
            append!(ssh.α, α_new)

            # adding new non-linear electron-phonon coupling
            α₂_new = @. phase * ( fill(α₂,Nnewbonds) + σα₂*randn(Nnewbonds) )
            append!(ssh.α₂, α₂_new)

            # adding phonon to bond mapping
            append!(ssh.phonon_to_bond, collect((i-1)*Nnewbonds+1:i*Nnewbonds))

            # adding bond to phonon mapping
            append!(ssh.bond_to_phonon, collect((ssh.nph-1)*Nnewbonds+1:ssh.nph*Nnewbonds))
        else

            # add null mapping from bond to no phonon
            append!(ssh.bond_to_phonon, zeros(Int,Nnewbonds))
        end
    end

    # sort neighbor_table
    perm                = sorted_neighbor_table_perm!(ssh.neighbor_table)
    ssh.neighbor_table .= ssh.neighbor_table[:,perm]

    # get checkerboard groups
    groups = checkerboard_groups(ssh.neighbor_table)

    # get checkerboard permutation and apply to neighbor table
    new_perm = checkerboard_order(groups)
    ssh.checkerboard_perm = perm[new_perm]
    ssh.neighbor_table   .= ssh.neighbor_table[:,new_perm]

    # number of bonds in lattice
    ssh.Nbonds = size(ssh.neighbor_table,2)

    # number of phonons in lattice
    ssh.Nph = length(ssh.ω)

    # number of phonon fields/degrees of freedom
    ssh.Ndof = ssh.Lτ * ssh.Nph

    # initialize phonon fields
    ssh.x = zeros(T1,ssh.Ndof)

    # mapping from field to phonon
    ssh.field_to_phonon = repeat(collect(1:ssh.Nph), inner=ssh.Lτ)
    ssh.field_to_τ      = repeat(collect(1:ssh.Lτ),  outer=ssh.Nph)

    # intialize checkerboard matrix elements
    ssh.t′    = zeros(T2,ssh.Lτ,ssh.Nbonds)
    ssh.cosht = zeros(T2,ssh.Lτ,ssh.Nbonds)
    ssh.sinht = zeros(T2,ssh.Lτ,ssh.Nbonds)
    for bond in 1:ssh.Nbonds
        t′    = ssh.t[bond]
        cosht = cosh(ssh.Δτ*t′)
        sinht = sinh(ssh.Δτ*t′)
        index = ssh.checkerboard_perm[bond]
        for τ in 1:ssh.Lτ
            ssh.t′[τ,bond]     = t′
            ssh.cosht[τ,index] = cosht
            ssh.sinht[τ,index] = sinht
        end
    end

    return nothing
end

"""
Update SSH model.
"""
function update_model!(ssh::SSHModel{T1,T2}) where {T1,T2}

    # iterate of phonon fields
    for field in 1:ssh.Ndof
        # get phonon
        phonon = ssh.field_to_phonon[field]
        # get time slice
        τ = ssh.field_to_τ[field]
        # get bond
        bond = ssh.phonon_to_bond[phonon]
        # getting indexing for checkerboard order
        index = ssh.checkerboard_perm[bond]
        # update matrix elements of exp{-Δτ⋅K} = exp{Δτ⋅(t-α⋅x)}
        t′                 = ssh.t[bond] - ssh.α[phonon] * ssh.x[field] - ssh.α₂[phonon] * ssh.x[field]^2
        ssh.t′[τ,bond]     = t′
        ssh.cosht[τ,index] = cosh(ssh.Δτ*t′)
        ssh.sinht[τ,index] = sinh(ssh.Δτ*t′)
    end

    return nothing
end

"""
Multiply by M matrix.
"""
function mulM!(Mv::AbstractVector{T2},ssh::SSHModel{T1,T2},v::AbstractVector{T2}) where {T1,T2}

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
    # • [M⋅v](1) = v(1) + B(1)⋅v(Lτ)  for τ = 1
    # • [M⋅v](τ) = v(τ) - B(τ)⋅v(τ-1) for τ > 1
    # • B(τ) = exp{Δτ⋅μ}⋅exp{-Δτ⋅K[x(τ)]}
    # • exp{Δτ⋅μ} is represent an NxN diagonal matrix that is the same for all
    #   time slice τ and is stored as a vector.
    # • exp{-Δτ⋅K[x(τ)]} is applied to a vector/represented using the checkerboard approximation.

    @unpack cosht, sinht, neighbor_table, Nbonds, Nsites, Lτ, expΔτμ = ssh

    # Calculate [M⋅v](τ) = v(τ-1)

    @fastmath @inbounds for i in 1:Nsites
        for τ in 1:Lτ
            τm1    = mod1(τ-1,Lτ)
            iτ     = get_index(τ,  i,Lτ)
            iτm1   = get_index(τm1,i,Lτ)
            Mv[iτ] = v[iτm1]
        end
    end

    # Calculate [M⋅v](τ) = exp{-Δτ⋅K[x(τ)]}⋅v(τ-1)
    
    checkerboard_mul!(Mv,neighbor_table,cosht,sinht)

    # Calcualte [M⋅v](τ) = B(τ)⋅v(τ-1) = exp{Δτ⋅μ}⋅[exp{-Δτ⋅K[x(τ)]}⋅v(τ-1)]

    # iterate of sites in lattice
    @fastmath @inbounds for i in 1:Nsites

        # get matrix elements 
        expΔτμi = expΔτμ[i]

        # iterate over time slices
        for τ in 1:Lτ

            # multiply vector element by matrix element
            iτ     = get_index(τ,i,Lτ)
            Mv[iτ] = expΔτμi * Mv[iτ]
        end
    end

    # Calculate [M⋅v](τ) = v(τ) - B(τ)⋅v(τ-1)

    # iterate over sites in lattice
    @fastmath @inbounds for i in 1:Nsites

        # [M⋅v](1) = v(1) + B(1)⋅v(Lτ)
        i1     = get_index(1,i,Lτ)
        Mv[i1] = v[i1] + Mv[i1]

        # iterate of time slices
        for τ in 2:Lτ

            # [M⋅v](τ) = v(τ) - B(τ)⋅v(τ-1)
            iτ     = get_index(τ,i,Lτ)
            Mv[iτ] = v[iτ] - Mv[iτ]
        end
    end

    return nothing
end


"""
Multiply by Mᵀ matrix.
"""
function mulMᵀ!(Mᵀv::AbstractVector{T2},ssh::SSHModel{T1,T2},v::AbstractVector{T2}) where {T1,T2}

    #####################################
    ## PERFORM MULTIPLICATION y = Mᵀ⋅v ##
    #####################################

    #           EXAMPLE OF Mᵀ MATRIX CONVENTION
    #      |    1   -Bᵀ(2)   0      0      0      0    |
    #      |    0     1    -Bᵀ(3)   0      0      0    |
    # Mᵀ = |    0     0      1    -Bᵀ(4)   0      0    |
    #      |    0     0      0      1    -Bᵀ(5)   0    |
    #      |    0     0      0      0      1    -Bᵀ(6) |
    #      | +Bᵀ(1)   0      0      0      0      1    | 

    # Notes:
    # • y(τ)  = [Mᵀ⋅v](τ)  = v(τ)  - Bᵀ(τ+1)⋅v(τ+1) for τ < Lτ
    # • y(Lτ) = [Mᵀ⋅v](Lτ) = v(Lτ) + Bᵀ(1)⋅v(1)     for τ = Lτ
    # • Bᵀ(τ) = exp{-Δτ⋅K[x(τ)]}ᵀ⋅exp{Δτ⋅μ}
    # • exp{Δτ⋅μ} is represent an NxN diagonal matrix that is the same for all
    #   time slice τ and is stored as a vector.
    # • exp{-Δτ⋅K[x(τ)]} is applied to a vector/represented using the checkerboard approximation.

    @unpack cosht, sinht, neighbor_table, Nbonds, Nsites, Lτ, expΔτμ = ssh

    # Calculate exp{Δτ⋅μ}⋅v(τ+1)

    # iterate of sites in lattice
    @fastmath @inbounds for i in 1:Nsites

        # get matrix elements 
        expΔτμi = expΔτμ[i]

        # iterate over time slices
        for τ in 1:Lτ

            # get index into output vector
            iτ = get_index(τ,i,Lτ)

            # multiply vector element by matrix element
            Mᵀv[iτ] = expΔτμi * v[iτ]
        end
    end

    # Caluclate Bᵀ(τ+1)⋅v(τ+1) = exp{-Δτ⋅K[x(τ+1)]}ᵀ⋅[exp{Δτ⋅μ}⋅v(τ+1)]

    checkerboard_transpose_mul!(Mᵀv,neighbor_table,cosht,sinht)

    # Calculate [Mᵀ⋅v](τ) = v(τ) - Bᵀ(τ+1)⋅v(τ+1)

    # iterate over sites in lattice
    for i in 1:Nsites

        # record [Mᵀ⋅v](L) = v(L) + Bᵀ(1)⋅v(1)
        i1     = get_index(1 ,i,Lτ)
        iL     = get_index(Lτ,i,Lτ)
        Mᵀv_iL = v[iL] + Mᵀv[i1]

        # iterate over time slices
        for τ in 1:Lτ-1

            # [Mᵀ⋅v](τ) = v(τ) - Bᵀ(τ+1)⋅v(τ+1)
            iτ      = get_index(τ,i,Lτ)
            iτp1    = get_index(τ+1,i,Lτ)
            Mᵀv[iτ] = v[iτ] - Mᵀv[iτp1]
        end

        # [Mᵀ⋅v](L) = v(L) + Bᵀ(1)⋅v(1)
        Mᵀv[iL] = Mᵀv_iL
    end

    return nothing
end

"""
Calculates ⟨∂M/∂xᵢⱼ(τ)⟩ = uᵀ⋅[∂M/∂xᵢⱼ(τ)]⋅v for each xᵢⱼ(τ) degree of freedom,
writing each of the ⟨∂M/∂xᵢⱼ(τ)⟩ values to a vector dMdx.
"""
function muldMdx!(dMdx::AbstractVector{T2},u::AbstractVector{T2},ssh::SSHModel{T1,T2},v::AbstractVector{T2}) where {T1,T2}

    #           EXAMPLE OF dM/dx MATRIX CONVENTION
    #         |     0        0        0        0        0   +dB₁/dx₂ |
    #         | -dB₂/dx₂     0        0        0        0       0    |
    # dM/dx = |     0    -dB₃/dx₃     0        0        0       0    |
    #         |     0        0    -dB₄/dx₄     0        0       0    |
    #         |     0        0        0    -dB₅/dx₅     0       0    |
    #         |     0        0        0        0    -dB₆/dx₆    0    | 

    # Notes:
    # • B(τ)          = exp{Δτ⋅μ}⋅exp{-Δτ⋅K(τ)}
    # • ∂B(τ)/∂xᵢⱼ(τ) = -Δτ⋅∂K(τ)/∂xᵢⱼ(τ)⋅exp{Δτ⋅μ}⋅exp{-Δτ⋅K(τ)} = -Δτ⋅∂K(τ)/∂xᵢⱼ(τ)⋅B(τ)
    # • Kᵢⱼ(τ)        = -(tᵢⱼ-αᵢⱼ⋅xᵢⱼ(τ)) = matrix elements of a symmetric NxN matrix K(τ)
    # • ∂K(τ)/∂xᵢⱼ(τ) = a symmetric NxN matrix with two non-zero matrix elements given by ∂Kᵢⱼ(τ)/∂xᵢⱼ(τ) = αᵢⱼ
    #
    # • ⟨∂M/∂xᵢⱼ(τ)⟩ = uᵀ⋅[∂M/∂xᵢⱼ(τ)]⋅v                       for τ > 1
    #                = uᵀ(τ)⋅[-∂B(τ)/∂xᵢⱼ(τ)]⋅v(τ-1)           for τ > 1
    #                = uᵀ(τ)⋅[Δτ⋅∂K(τ)/∂xᵢⱼ(τ)⋅B(τ)]⋅v(τ-1)    for τ > 1
    #                = +uᵀ(τ)⋅[Δτ⋅∂K(τ)/∂xᵢⱼ(τ)]⋅[B(τ)⋅v(τ-1)] for τ > 1
    #
    # • ⟨∂M/∂xᵢⱼ(1)⟩ = uᵀ⋅[∂M/∂xᵢⱼ(1)]⋅v                     for τ = 1
    #                = uᵀ(1)⋅[+∂B(1)/∂xᵢⱼ(1)]⋅v(L)           for τ = 1
    #                = uᵀ(1)⋅[-Δτ⋅∂K(1)/∂xᵢⱼ(τ)⋅B(1)]⋅v(L)   for τ = 1
    #                = -uᵀ(1)⋅[Δτ⋅∂K(1)/∂xᵢⱼ(τ)]⋅[B(1)⋅v(L)] for τ = 1

    @unpack cosht, sinht, neighbor_table, checkerboard_perm, Nbonds, Ndof, Nsites, Lτ, Δτ, expΔτμ, α, α₂, x = ssh
    @unpack field_to_phonon, field_to_τ, phonon_to_bond = ssh

    ################################
    ## FIRST CALUCATE B(τ)⋅v(τ-1) ##
    ################################

    # vector to represent B(τ)⋅v(τ-1)
    Bv = ssh.v′

    # Calculate [B⋅v](τ) = v(τ-1)
    
    @fastmath @inbounds for i in 1:Nsites
        for τ in 1:Lτ
            τm1    = mod1(τ-1,Lτ)
            iτm1   = get_index(τm1,i,Lτ)
            iτ     = get_index(τ,i,Lτ)
            Bv[iτ] = v[iτm1]
        end
    end

    # Calculate [B⋅v](τ) = exp{-Δτ⋅K[x(τ)]}⋅v(τ-1)
    
    checkerboard_mul!(Bv,neighbor_table,cosht,sinht)

    # Calcualte [B⋅v](τ) = B(τ)⋅v(τ-1) = exp{Δτ⋅μ}⋅[exp{-Δτ⋅K[x(τ)]}⋅v(τ-1)]

    # iterate of sites in lattice
    @fastmath @inbounds for i in 1:Nsites

        # get matrix elements 
        expΔτμi = expΔτμ[i]

        # iterate over time slices
        for τ in 1:Lτ

            # multiply vector element by matrix element
            Bv[get_index(τ,i,Lτ)] *= expΔτμi
        end
    end

    #######################################
    ## CALCULATE ALL ⟨∂M/∂xᵢⱼ(τ)⟩ VALUES ##
    #######################################

    # iterate over degrees of freedom
    @fastmath @inbounds for field in 1:Ndof

        # field to phonon
        phonon = field_to_phonon[field]

        # field to τ
        τ = field_to_τ[field]

        # phonon to bond
        bond  = phonon_to_bond[phonon]

        # get index into neighbor table assoicated with bond
        index = checkerboard_perm[bond]

        # the pair of sites associated with the bond/phonon
        i = neighbor_table[1,index]
        j = neighbor_table[2,index]

        # electron-phonon coupling associated with phonon
        αᵢⱼ = α[phonon]

        # non-linear electron-phonon coupling
        α₂ᵢⱼ = α₂[phonon]

        # get index into vectors associated (i,τ) and (j,τ) space-time coordinates
        iτ = get_index(τ,i,Lτ)
        jτ = get_index(τ,j,Lτ)

        # get the phonon fields
        x₀ = x[field]

        # ⟨∂M/∂xᵢⱼ(τ)⟩ = +uᵀ(τ)⋅[Δτ⋅∂K(τ)/∂xᵢⱼ(τ)]⋅[B(τ)⋅v(τ-1)]
        dMdx[field] = Δτ * ((αᵢⱼ + 2*α₂ᵢⱼ*x₀) * u[iτ] * Bv[jτ] + conj(αᵢⱼ + 2*α₂ᵢⱼ*x₀) * u[jτ] * Bv[iτ])

        # ⟨∂M/∂xᵢⱼ(1)⟩ = -uᵀ(1)⋅[Δτ⋅∂K(1)/∂xᵢⱼ(1)]⋅[B(1)⋅v(L)]
        if τ==1
            dMdx[field] = -dMdx[field]
        end
    end

    return nothing
end

##############################
## PHONON FIELD IO ROUTINES ##
##############################

"""
Writes the current phonon field configuration to file.
"""
function write_phonons!(ssh::SSHModel{T1,T2,T3},filename::String) where {T1,T2,T3}

    if ssh.Nph>0

        # get info about size of lattice and number of phonons
        L  = ssh.Lτ
        n  = ssh.nph
        N  = div(ssh.Nph,n)

        # get phonon fields
        x = reshaped(ssh.x,(L,N,n))

        # open file
        open(filename,"w") do file

            # write header to file
            write(file, "type loc tau x\n")

            # iterate over phonon fileds
            for phonon in 1:n
                for i in 1:N
                    for τ in 1:L 
                        # get phonon field
                        x₀ = x[τ,i,phonon]
                        # write to file
                        write(file, @sprintf("%d %d %d %.6f\n",phonon,i,τ,x₀))
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
function read_phonons!(ssh::SSHModel{T1,T2,T3},filename::String) where {T1,T2,T3}

    # get info about size of lattice and number of phonons
    L  = ssh.Lτ
    n  = ssh.nph
    N  = div(ssh.Nph,n)

    # get phonon fields
    x = reshaped(ssh.x,(L,N,n))

    # open file
    open(filename,"r") do file

        # read in header
        header = readline(file)

        # iterate over lines in file
        for line in eachline(file)

            # split line at white space
            atoms = split(line," ")

            # extract info about phonon field and location
            phonon = parse(Int,atoms[1])
            cell   = parse(Int,atoms[2])
            τ      = parse(Int,atoms[3])
            x₀     = parse(T1, atoms[4])

            # record phonon field
            x[τ,cell,phonon] = x₀
        end
    end

    # construct exponentiated interaction matrix
    update_model!(ssh)

    return nothing
end