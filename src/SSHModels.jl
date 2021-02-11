using Parameters
using LinearAlgebra
using Random
using Logging

import LinearAlgebra: mul!, ldiv!, transpose!
import Random: randn!

using ..UnitCells: UnitCell
using ..Lattices: Lattice, sorted_neighbor_table_perm!, loc_to_site, calc_neighbor_table
using ..Checkerboard: checkerboard_order, checkerboard_groups, checkerboard_mul!, checkerboard_transpose_mul!
using ..IterativeSolvers: GMRES, ConjugateGradient, BiCGStab
using ..Utilities: get_index, get_τ, δ, reshaped

export write_K_matrix!

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

    "name of the phonon."
    name::String

    function SSHBond(t::T2,σt::T2,ω::T1,σω::T1,ω₄::T1,σω₄::T1,α::T1,σα::T1,α₂::T1,σα₂::T1,
                     o₁::Int,o₂::Int,v::AbstractVector{Int},name::String="") where {T1<:AbstractFloat,T2<:Continuous}

        @assert length(v)==3

        if name==""
            name = randstring(5)
        end

        v′ = zeros(Int,3)
        copyto!(v′,v)
        has_phonon = (!iszero(ω))||(!iszero(σω))
        return new{T1,T2}(t,σt,ω,σω,ω₄,σω₄,α,σα,α₂,σα₂,o₁,o₂,v′,has_phonon,name)
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

    "Maps every field to its equivalent primary field.
    If a field maps onto itself then it is primary."
    primary_field::Vector{Int}

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

    """
    Inverse checkerboard order. Map neighbor table entry onto bond.
    """
    inv_checkerboard_perm::Vector{Int}

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

    """
    data folder
    """
    datafolder::String


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
        x                     = Vector{T1}(undef,0)
        t                     = Vector{T2}(undef,0)
        ω                     = Vector{T1}(undef,0)
        ω₄                    = Vector{T1}(undef,0)
        α                     = Vector{T2}(undef,0)
        α₂                    = Vector{T2}(undef,0)
        μ                     = zeros(T1,Nsites)
        expΔτμ                = ones(T1,Nsites)
        t′                    = Matrix{T2}(undef,Lτ,0)
        checkerboard_perm     = Vector{Int}(undef,0)
        inv_checkerboard_perm = Vector{Int}(undef,0)
        neighbor_table        = Matrix{Int}(undef,2,0)
        cosht                 = Matrix{T2}(undef,Lτ,0)
        sinht                 = Matrix{T2}(undef,Lτ,0)
        field_to_phonon       = Vector{Int}(undef,0)
        field_to_τ            = Vector{Int}(undef,0)
        primary_field         = Vector{Int}(undef,0)
        phonon_to_bond        = Vector{Int}(undef,0)
        bond_to_phonon        = Vector{Int}(undef,0)
        bond_to_definition    = Vector{Int}(undef,0)

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
                             field_to_phonon,field_to_τ,primary_field,
                             phonon_to_bond, bond_to_phonon,bond_to_definition,
                             checkerboard_perm,inv_checkerboard_perm,neighbor_table,cosht,sinht,
                             mul_by_M, transposed, v′, v″, v‴, solver,"")
    end
end

"""
Assign hopping (and phonon) in SSH model.
"""
function assign_hopping!(ssh::SSHModel{T1,T2},t::T2,σt::T1,ω::T1,σω::T1,ω₄::T1,σω₄::T1,α::T1,σα::T1,α₂::T1,σα₂::T1,
                         o₁::Int,o₂::Int,v::Vector{Int},name::String="") where {T1<:AbstractFloat,T2<:Continuous}

    # define ssh bond
    sshbond = SSHBond(t,σt,ω,σω,ω₄,σω₄,α,σα,α₂,σα₂,o₁,o₂,v,name)
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

    # names of phonons
    names = Vector{String}(undef,0)

    # iterate over types of bonds in lattices
    for i in 1:ssh.nbonds

        # getting parameters that define bond
        @unpack t, σt, α, σα, α₂, σα₂, ω, σω, ω₄, σω₄, o₁, o₂, v, has_phonon, name = ssh.bond_definitions[i]

        # calculate new neighbors
        new_neighbors = calc_neighbor_table(ssh.lattice,o₁,o₂,v)

        # number of new bonds
        Nnewbonds = size(new_neighbors,2)

        # adding new neighbors to neighbor tables for current bond defintion
        ssh.neighbor_table = hcat(ssh.neighbor_table, new_neighbors)

        # phase of hopping
        if iszero(t)
            phase = 1.0
        else
            phase = t/abs(t)
        end

        # adding new hopppings for current bond
        t_new = @. phase * ( fill(abs(t),Nnewbonds) + σt*randn(Nnewbonds) )
        append!(ssh.t, t_new)

        # adding bond to definition mapping
        append!(ssh.bond_to_definition, fill(i,Nnewbonds))

        # if bond has phonon on it
        if has_phonon

            # incrementing count of number of types of phonons
            ssh.nph += 1

            # record phonon names
            push!(names,name)

            # adding new phonon frequencies
            ω_new = @. fill(ω,Nnewbonds) + σω*randn(Nnewbonds)
            append!(ssh.ω, ω_new)

            # adding anharmonic phonon coefficient
            ω₄_new = @. fill(ω₄,Nnewbonds) + σω₄*randn(Nnewbonds)
            append!(ssh.ω₄, ω₄_new)

            # adding new linear electron-phonon coupling
            if iszero(α)
                phase = 1.0
            else
                phase = α/abs(α)
            end
            α_new = @. phase * ( fill(abs(α),Nnewbonds) + σα*randn(Nnewbonds) )
            append!(ssh.α, α_new)

            # adding new non-linear electron-phonon coupling
            if iszero(α₂)
                phase = 1.0
            else
                phase = α₂/abs(α₂)
            end
            α₂_new = @. phase * ( fill(abs(α₂),Nnewbonds) + σα₂*randn(Nnewbonds) )
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
    new_perm                  = checkerboard_order(groups)
    ssh.neighbor_table       .= ssh.neighbor_table[:,new_perm]
    ssh.inv_checkerboard_perm = perm[new_perm]
    ssh.checkerboard_perm     = sortperm(ssh.inv_checkerboard_perm)

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

    # initiailize primary fields mapping
    primary_field  = collect(1:ssh.Ndof)
    primary_field′ = reshape(primary_field,(div(ssh.Ndof,ssh.nph),ssh.nph))

    # construct a tabluation of all the fields
    fields = reshape(collect(1:ssh.Ndof),(div(ssh.Ndof,ssh.nph),ssh.nph))

    # iterate over types of phonons
    for ph_type in 1:ssh.nph
        # iterate over remaining types of phonons
        for ph_type′ in (ph_type+1):ssh.nph
            # if phonons share the same name
            if names[ph_type]==names[ph_type′]
                # record primary field
                if primary_field′[1,ph_type′] > fields[1,ph_type]
                    @views @. primary_field′[:,ph_type′] = fields[:,ph_type]
                end
            end
        end
    end

    # record primary fields
    ssh.primary_field = primary_field
    
    return nothing
end

"""
Update SSH model.
"""
function update_model!(ssh::SSHModel{T1,T2}) where {T1,T2}

    # account of updated chemical potential
    @. ssh.expΔτμ = exp( ssh.Δτ * ssh.μ )

    # warning flag for unphysically large phonon displacement
    flag_large_displacement = false

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
        v                  = ssh.α[phonon]*ssh.x[field] + ssh.α₂[phonon]*ssh.x[field]^2
        t′                 = ssh.t[bond] - v
        ssh.t′[τ,bond]     = t′
        ssh.cosht[τ,index] = cosh(ssh.Δτ*t′)
        ssh.sinht[τ,index] = sinh(ssh.Δτ*t′)
        # detect unphysically large phonon displacement
        if abs(ssh.t[bond]) < abs(v)
            flag_large_displacement = true
        end
    end

    # report unphysically large phonon displacement
    if flag_large_displacement
        @info("Unphysically Large Phonon Displacement\n")
        logger = global_logger()
        flush(logger.stream)
    end

    # make sure equivalent fields are equal
    for field in 1:ssh.Ndof
        # get primary field
        field′ = ssh.primary_field[field]
        # if current field is not primary
        if field != field′
            if !(ssh.x[field]≈ssh.x[field′])
                error("(x[$field]=$(ssh.x[field])) != (x[$field′]=$(ssh.x[field′]))\n")
            end
        end
    end

    return nothing
end

"""
Fill vector that will ultimate be used to update fields in model.
"""
function randn!(v::AbstractVector{T},ssh::SSHModel{T}) where {T}

    # fill with random numbers
    randn!(v)

    # account for some fields being equivalent
    @views @. v = v[ssh.primary_field]

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
    # • B(τ) = exp{-Δτ⋅K[x(τ)]}⋅exp{Δτ⋅μ}
    # • exp{Δτ⋅μ} is represent an NxN diagonal matrix that is the same for all
    #   time slice τ and is stored as a vector.
    # • exp{-Δτ⋅K[x(τ)]} is applied to a vector/represented using the checkerboard approximation.

    @unpack cosht, sinht, neighbor_table, Nbonds, Nsites, Lτ, expΔτμ = ssh

    # Calculate [M⋅v](τ) = exp{Δτ⋅μ}⋅v(τ-1)

    @fastmath @inbounds for i in 1:Nsites
        expΔτμi = expΔτμ[i]
        for τ in 1:Lτ
            τm1    = mod1(τ-1,Lτ)
            iτ     = get_index(τ,  i,Lτ)
            iτm1   = get_index(τm1,i,Lτ)
            Mv[iτ] = expΔτμi * v[iτm1]
        end
    end

    # Calculate [M⋅v](τ) = B(τ)⋅v(τ-1) = exp{-Δτ⋅K[x(τ)]}⋅[exp{Δτ⋅μ}⋅v(τ-1)]
    
    checkerboard_mul!(Mv,neighbor_table,cosht,sinht)

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
    # • Bᵀ(τ) = exp{Δτ⋅μ}⋅exp{-Δτ⋅K[x(τ)]}ᵀ
    # • exp{Δτ⋅μ} is represent an NxN diagonal matrix that is the same for all
    #   time slice τ and is stored as a vector.
    # • exp{-Δτ⋅K[x(τ)]} is applied to a vector/represented using the checkerboard approximation.

    @unpack cosht, sinht, neighbor_table, Nbonds, Nsites, Lτ, expΔτμ = ssh

    # Calculate exp{-Δτ⋅K[x(τ+1)]}ᵀ⋅v(τ+1)

    copyto!(Mᵀv,v)
    checkerboard_transpose_mul!(Mᵀv,neighbor_table,cosht,sinht)

    # Calculate [Mᵀ⋅v](τ) = v(τ) - Bᵀ(τ+1)⋅v(τ+1)

    # iterate over sites in lattice
    for i in 1:Nsites

        expΔτμi = expΔτμ[i]

        # record [Mᵀ⋅v](L) = v(L) + Bᵀ(1)⋅v(1) = v(L) + exp{Δτ⋅μ}⋅exp{-Δτ⋅K[x(1)]}ᵀ⋅v(1)
        i1     = get_index(1 ,i,Lτ)
        iL     = get_index(Lτ,i,Lτ)
        Mᵀv_iL = v[iL] + expΔτμi * Mᵀv[i1]

        # iterate over time slices
        for τ in 1:Lτ-1

            # [Mᵀ⋅v](τ) = v(τ) - Bᵀ(τ+1)⋅v(τ+1) = v(τ) - exp{Δτ⋅μ}⋅exp{-Δτ⋅K[x(τ+1)]}ᵀ⋅v(τ+1)
            iτ      = get_index(τ,i,Lτ)
            iτp1    = get_index(τ+1,i,Lτ)
            Mᵀv[iτ] = v[iτ] - expΔτμi * Mᵀv[iτp1]
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
    #          |     0        0        0        0        0       0    |
    #          |     0        0        0        0        0       0    |
    # dM/dx₃ = |     0    -dB₃/dx₃     0        0        0       0    |
    #          |     0        0        0        0        0       0    |
    #          |     0        0        0        0        0       0    |
    #          |     0        0        0        0        0       0    | 

    # Notes:
    # • B(τ)          = exp{-Δτ⋅K(τ)}⋅exp{Δτ⋅μ}
    # • ∂B(τ)/∂xᵢⱼ(τ) = -Δτ⋅∂K(τ)/∂xᵢⱼ(τ)⋅exp{-Δτ⋅K(τ)}⋅exp{Δτ⋅μ} = -Δτ⋅∂K(τ)/∂xᵢⱼ(τ)⋅B(τ)
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

    # Checkerboard Decomposition:           exp{-Δτ⋅K(τ)}  ≈ ∏(n=1...N)[exp{-Δτ⋅Kₙ(τ)}]
    # Transpose Checkerboard Decomposition: exp{-Δτ⋅K(τ)}ᵀ ≈ ∏(n=N...1)[exp{-Δτ⋅Kₙ(τ)}]
    # Note: exp{-Δτ⋅Kₙ(τ)}ᵀ = exp{-Δτ⋅Kₙ(τ)} [Hermitian]

    @unpack cosht, sinht, neighbor_table, checkerboard_perm, inv_checkerboard_perm = ssh
    @unpack Nbonds, Nph, Ndof, Nsites, Lτ, Δτ, expΔτμ, α, α₂, x = ssh
    @unpack field_to_phonon, field_to_τ, phonon_to_bond, bond_to_phonon = ssh
    @unpack primary_field = ssh

    b = ssh.v′
    c = ssh.v″

    # |b₀(τ)⟩ = exp{Δτ⋅μ}|v(τ-1)⟩
    @fastmath @inbounds for i in 1:Nsites
        expΔτμi = expΔτμ[i]
        for τ in 1:Lτ
            τm1   = mod1(τ-1,Lτ)
            iτ    = get_index(τ,  i,Lτ)
            iτm1  = get_index(τm1,i,Lτ)
            b[iτ] = expΔτμi * v[iτm1]
        end
    end

    # ⟨c₀(τ)| = ⟨u(τ)|exp{-Δτ⋅K(τ)} ⟹ |c₀(τ)⟩ = exp{-Δτ⋅K(τ)}ᵀ|u(τ)⟩
    copyto!(c,u)
    checkerboard_transpose_mul!(c,neighbor_table,cosht,sinht)

    # initialize dMdx to zero
    fill!(dMdx,0.0)

    # iterate over bonds in checkerboard order
    for n in 1:Nbonds

        # get bond
        bond = inv_checkerboard_perm[n]

        # get phonon
        phonon = bond_to_phonon[bond]

        # get pair of sites
        i = neighbor_table[1,n]
        j = neighbor_table[2,n]

        # iterate over imaginary time
        for τ in 1:Lτ

            # get matrix elements of exp{-Δτ⋅Kₙ(τ)}
            cosht′ = cosht[τ,n]
            sinht′ = sinht[τ,n]

            # get indices into vectors
            iτ = get_index(τ,i,Lτ)
            jτ = get_index(τ,j,Lτ)

            # |bₙ(τ)⟩ = exp{-Δτ⋅Kₙ(τ)}|bₙ₋₁(τ)⟩
            biτ   = b[iτ]
            bjτ   = b[jτ]
            b[iτ] = cosht′ * biτ +      sinht′  * bjτ
            b[jτ] = cosht′ * bjτ + conj(sinht′) * biτ

            # |cₙ(τ)⟩ = exp{-Δτ⋅Kₙ(τ)}⁻¹|cₙ₋₁(τ)⟩
            ciτ   = c[iτ]
            cjτ   = c[jτ]
            c[iτ] = cosht′ * ciτ -      sinht′  * cjτ
            c[jτ] = cosht′ * cjτ - conj(sinht′) * ciτ

            # if there is a phonon on the bond
            if phonon != 0

                # get index to current field
                field = get_index(τ,phonon,Lτ)

                # get xₙ(τ)
                xₙ = x[field]

                # calculate ∂Kₙ(τ)/∂xₙ(τ)
                dKdx = α[phonon] + 2*α₂[phonon]*xₙ

                # ⟨∂M/∂xᵢⱼ(τ)⟩ = ⟨cₙ(τ)|Δτ⋅∂Kₙ(τ)/∂xₙ(τ)|bₙ(τ)⟩
                dmdx = conj(c[jτ])*Δτ*dKdx*b[iτ] + conj(c[iτ]*Δτ*dKdx)*b[jτ]

                # flip sign if τ=1
                if τ==1
                    dmdx = -dmdx
                end

                # record result
                dMdx[primary_field[field]] += dmdx
            end
        end
    end

    # account for equivalent fields
    @views @. dMdx = dMdx[primary_field]

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

"Write Hamiltonian K[τ] matrix for given time slice τ to file. Includes on-site energies."
function write_K_matrix!(filename::String, model::SSHModel{T1,T2,T3}, τ::Int=1) where {T1,T2,T3}

    open(filename,"w") do file

        # write header
        write(file,"col row val\n")

        # iterate over on-site energies
        for i in 1:model.Nsites
            ϵ = -model.μ[i]
            write(file,"$i $i $ϵ\n")
        end

        # iterate over bonds
        for bond in 1:model.Nbonds
            # matrix element based on hopping amplitude
            val = -model.t′[τ,bond]
            # neighboring sites associated with bond
            indx = model.checkerboard_perm[bond]
            i = model.neighbor_table[1,indx]
            j = model.neighbor_table[2,indx]
            # write to file
            write(file,"$i $j $val\n")
            write(file,"$j $i $(conj(val))\n")
        end
    end

    return nothing
end