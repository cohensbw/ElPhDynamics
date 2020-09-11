using Parameters
using LinearAlgebra

import LinearAlgebra: mul!, ldiv!, transpose!

using ..UnitCells: UnitCell
using ..Lattices: Lattice, sorted_neighbor_table_perm, loc_to_site, calc_neighbor_table
using ..Checkerboard: checkerboard_order, checkerboard_groups
using ..IterativeSolvers: GMRES, ConjugateGradient, BiCGStab
using ..Utilities: get_index, get_τ, δ

struct SSHPhonon{T1<AbstractFloat,T2<:Continuous}

    "bare phonon hopping"
    t::T2

    "average phonon frequency"
    ω::T1

    "electron-phonon coupling"
    α::T2

    "orbital"
    o₁::Int

    "orbital"
    o₂::Int

    "displacement in unit cells"
    v::Vector{Int}
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

    "Represents lattice"
    lattice::Lattice{T1}

    "Nbond bare electron hoppings. Ordered in vector according to checkerboard decomposition."
    t::Vector{T2}

    "Nph phonon frequencies. Ordered in vector by phonon defintion."
    ω::Vector{T1}

    "Nph electron-phonon couplings. Ordered in vector by phonon defintion."
    α::Vector{T2}

    "Chemical potential"
    μ::Vector{T1}

    "Diagonal matrix exp{Δτ⋅μ}"
    expΔτμ::Vector{T1}

    "Ndof=Nph*Lτ phonon fields. Ordered in vector by phonon defintion."
    x::Vector{T1}

    "Matrix of dimensions (Lτ,Nbonds) containing cosh(t′) where t′=Δτ(t-α⋅x).
    Ordered in vector according to checkerboard decomposition."
    cosht::Matrix{T2}

    "Matrix of dimensions (Lτ,Nbonds) containing sinh(t′) where t′=Δτ(t-α⋅x).
    Ordered in vector according to checkerboard decomposition."
    sinht::Matrix{T2}

    "SSHPhonon defintions"
    phonon_defintions::Vector{SSHPhonon{T1,T2}}

    "Neighbor table telling which sites in lattice are connect by bonds.
    Ordered according to checkerboard decomposition."
    neighbor_table::Matrix{Int}

    "Map Ndof phonon fields onto one of the Nph phonons"
    field_to_phonon::Vector{Int}

    "Map Ndof phonon fields onto a time slice τ"
    field_to_τ::Vector{Int}

    "Maps a given phonon onto an SSHPhonon defintion."
    phonon_to_definition::Vector{Int}

    "Maps phonon to bond"
    phonon_to_bond::Vector{Int}

    "A vector of length `ninidces` to temporarily store data."
    ytemp::Vector{T2}

    "A vector for storing the temporary product Mᵀ⋅g needed for Conjugate Gradient method."
    Mᵀg::Vector{T2}

    "If true the default mul! routine multiplies by the M matrix.
    If false the default mul! routine multiplies by the symmetric matrix MᵀM instead."
    mul_by_M::Bool

    "If true multiply by Mᵀ instead of M."
    transposed::Bool

    """
    Iterative Solver
    """
    solver::T3

    function SSHModel(lattice::Lattice{T}, β::T1, Δτ::T1;
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

        # intialize vector to contain phonon defintions
        phonon_defintions = Vector{SSHPhonon{T1,T2}}(undef,0)

        # initialize vectors for hamiltonian parameters
        t      = Vector{T2}(undef,0)
        ω      = Vector{T1}(under,0)
        α      = Vector{T2}(undef,0)
        μ      = zeros(T1,Nsites)
        expΔτμ = Vector{T1}(undef,0)
        cosht  = Matrix{T2}(undef,Lτ,0)
        sinht  = Matrix{T2}(undef,Lτ,0)

        # intialize empty vector for phonon fields
        x = Vector{T1}(undef,0)

        # declare other empty array
        neighbor_table       = Matrix{Int}(undef,2,0)
        field_to_phonon      = Vector{Int}(undef,0)
        field_to_τ           = Vector{Int}(undef,0)
        phonon_to_bond       = Vector{Int}(undef,0)
        phonon_to_definition = Vector{Int}(undef,0)

        # declaring temporary storage vectors
        ytemp = zeros(T2,Ndim)
        Mᵀg   = zeros(T2,Ndim)

        # construct solver
        if lowercase(iterativesolver)=="cg"
            mul_by_M   = false
            transposed = false
            solver = ConjugateGradient(Mᵀg,tol=tol,maxiter=maxiter)
        elseif lowercase(iterativesolver)=="gmres"
            mul_by_M   = true
            transposed = false
            solver = GMRES(Mᵀg,tol=tol,restart=restart,maxiter=maxiter)
        elseif lowercase(iterativesolver)=="bicgstab"
            mul_by_M   = true
            transposed = false
            solver = BiCGStab(Mᵀg,tol=tol,maxiter=maxiter)
        end

        return SSHModel{T1,T2,T3}(β,Δτ,Lτ,Nsites,Nbonds,Nph,Ndim,Nbonds,Nph,Ndof,lattice,
                                  t,ω,α,μ,expΔτμ,x,cosht,sinht,
                                  phonon_defintions,neighbor_table,field_to_phonon,field_to_τ,
                                  phonon_to_definition,phonon_to_bond,
                                  ytemp, Mᵀg, mul_by_M, transposed)
    end
end

"""
Assign hopping (and phonon) in SSH model.
"""
function assign_hopping!(ssh::SSHModel{T1,T2},t::T2,σt::T1,ω::T1,σω::T1,α::T1,σα::T1,
                         o₁::Int,o₂::Int,v::Vector{Int}) where {T1<:AbstractFloat,T2<:Continuous}

    # getting new neighbors accounting for periodic boundary conditions
    newneighbors = calc_neighbor_table(holstein.lattice,o₁,o₂,v)

    # getting number of new neighbors
    Nnewneighbors = size(newneighbors,2)

    # adding new neighbors to neighbor tables
    ssh.neighbor_table = hcat(ssh.neighbor_table,newneighbors)

    # define ssh phonon
    if α>0.0
        append!( ssh.phonon_defintions    , SSHPhonon(t,ω,α,o₁,o₂,v) )
        append!( ssh.phonon_to_definition , fill(length(ssh.phonon_defintions),Nnewneighbors) )
    else
        append!( ssh.phonon_to_definition , fill(0,Nnewneighbors) )
    end

    # phase of hopping
    if T2<:Complex
        eiϕ = exp(im*angle(t))
    else
        eiϕ = sign(t)
    end

    # adding new hopping values
    append!(ssh.t, eiϕ.*(fill(abs(t),Nnewneighbors) + σt*randn(newneighbors)) )

    # adding new phonon frequencies
    append!(ssh.ω, fill(ω,Nnewneighbors) + σω*randn(newneighbors) )

    # adding new phonon coupling
    append!(ssh.α, eiϕ.*(fill(abs(α),Nnewneighbors) + σα*randn(newneighbors)) )

    return nothing
end

"""
Assign chemical potential in SSH model.
"""
function assign_μ!(ssh::SSHModel,μ::T1,σμ::T1,orbit::Int) where {T1<:AbstractFloat}

    for i in 1:ssh.Nsites
        if orbit==0 || orbit==lattice.site_to_orbit[i]
            ssh.μ[i] = μ + σμ * randn()
            ssh.expΔτμ[i] = exp( ssh.Δτ * ssh.μ[i] )
        end
    end

    return nothing
end

"""
Initialize SSH Model.
"""
function initialize_model!(ssh::SSHModel{T1,T2}) where {T1,T2}

    @unpack neighbor_table, t, ω, α, phonon_to_definition, phonon_to_bond, Lτ, Δτ = ssh

    # sort neighbor_table
    perm = sorted_neighbor_table_perm!(neighbor_table)
    neighbor_table .= neighbor_table[:,perm]

    # get checkerboard groups
    groups = checkerboard_groups(neighbor_table)

    # get checkerboard permutation
    new_perm = checkerboard_order(groups)

    # apply checkerboard ordering
    @. neighbor_table = neighbor_table[new_perm]
    @. perm = perm[new_perm]
    @. t = t[perm]
    @. ω = ω[perm]
    @. α = α[perm]
    @. phonon_to_definition = phonon_to_definition[perm]

    # mapping from phonons to bonds
    phonon_to_bond = collect(1:length(t))

    # remove "null" phonons where electron-phonon coupling α=0
    index = 1
    while index <= length(ω)
        # if null phonon remove
        if phonon_to_definition[index]==0
            popat!(ω,index)
            popat!(α,index)
            popat!(phonon_to_definition,index)
            popat!(phonon_to_bond,index)
        # else increment index to next phonon
        else
            index += 1
        end
    end

    # re-order phonons according to phonon definition
    perm                  = sortperm(phonon_to_definition)
    phonon_to_definition .= phonon_to_definition[perm]
    phonon_to_bond       .= phonon_to_bond[perm]
    ω                    .= ω[perm]
    α                    .= α[perm]

    # number of phonons in lattice
    ssh.Nph = length(ω)

    # number of bonds in lattice
    ssh.Nbonds = length(t)

    # number of phonon fields/degrees of freedom
    ssh.Ndof = Lτ * ssh.Nph

    # initialize phonon fields
    ssh.x = zeros(T1,ssh.Ndof)

    # mapping from field to phonon
    ssh.field_to_phonon = zeros(Int,ssh.Ndof)
    ssh.field_to_τ      = zeros(Int,ssh.Ndof)
    field = 0
    for phonon in 1:ssh.Ndof
        for τ in 1:Lτ
            field += 1
            ssh.field_to_phonon[field] = phonon
            ssh.field_to_τ[field]      = τ
        end
    end

    # intialize checkerboard matrix elements
    ssh.cosht = zeros(T2,Lτ,ssh.Nbonds)
    ssh.sinht = zeros(T2,Lτ,ssh.Nbonds)
    for bond in 1:ssh.Nbonds
        coshΔτt = cosh(Δτ*t[bond])
        sinhΔτt = sinh(Δτ*t[bond])
        for τ in 1:ssh.Lτ
            ssh.cosht[τ,bond] = coshΔτt
            ssh.sinht[τ,bond] = sinhΔτt
        end
    end

    return nothing
end

"""
Update SSH model.
"""
function update_model!(ssh::SSHModel{T1,T2}) where {T1,T2}

    @unpack x, t, ω, α, cosht, sinht, field_to_phonon, field_to_τ, phonon_to_bond, Ndof, Nph, Lτ, Δτ = ssh

    # iterate of phonon fields
    for field in 1:Ndof
        # get phonon
        phonon = field_to_phonon[field]
        # get time slice
        τ = field_to_τ[field]
        # get bond
        bond = phonon_to_bond[field]
        # update matrix elements of exp{-Δτ⋅K} = exp{Δτ⋅(t-α⋅x)}
        t′ = Δτ*(t[bond]-α[phonon]*x[field])
        cosht[τ,bond] = cosh(t′)
        sinht[τ,bond] = sinh(t′)
    end

    return nothing
end

"""
Multiply by M matrix.
"""
function mulM!(Mv::AbstractVector{T2},ssh::SSHModel{T1,T2},v::AbstractVector{T1}) where {T1,T2}

    ####################################
    ## PERFORM MULTIPLICATION y = M⋅v ##
    ####################################

    #           EXAMPLE OF M MATRIX CONVENTION
    #     |   1     0      0      0      0   +B(1) |
    #     | -B(2)   1      0      0      0     0   |
    # M = |   0   -B(2)    1      0      0     0   |
    #     |   0     0    -B(3)    1      0     0   |
    #     |   0     0      0    -B(4)    1     0   |
    #     |   0     0      0      0    -B(5)   1   | 

    # Notes:
    # • y(τ) = [M⋅v](τ) = v(τ) - B(τ)⋅v(τ-1) for τ > 1
    # • y(1) = [M⋅v](1) = v(1) + B(1)⋅v(Lτ)  for τ = 1
    # • B(τ) = exp{Δτ⋅μ}⋅exp{-Δτ⋅K[x(τ)]}
    # • exp{Δτ⋅μ} is represent an NxN diagonal matrix that is the same for all
    #   time slice τ and is stored as a vector.
    # • exp{-Δτ⋅K[x(τ)]} is applied to a vector/represented using the checkerboard approximation.

    @unpack cosht, sinht, neighbor_table, Nbonds, Nsites, Lτ, expΔτμ = ssh

    # Calculate exp{-Δτ⋅K[x(τ)]}⋅v(τ-1)
    
    # iterate over bonds
    @fastmath @inbounds for bond in 1:Nbonds

        # get pair of sites associated with bond
        i = neighbor_table[1,bond]
        j = neighbor_table[2,bond]

        # iterate over time slices
        @simd for τ in 1:Lτ

            # τ-1
            τm1 = mod1(τ-1,Lτ)

            # get index into input vector
            iτm1 = get_index(τm1,i,Lτ)
            jτm1 = get_index(τm1,j,Lτ)

            # get index into output vector
            iτ = get_index(τ,i,Lτ)
            jτ = get_index(τ,j,Lτ)

            # get matrix elements of 2x2 checkerboard matrix associated with bond
            c = cosht[τ,bond]
            s = sinht[τ,bond]

            # perform multiplication
            Mv[iτ] = c*v[iτm1] + s*v[jτm1]
            Mv[jτ] = c*v[jτm1] + conj(s)*s[iτm1]
        end
    end

    # Calcualte B(τ)⋅v(τ-1) = exp{Δτ⋅μ}⋅[exp{-Δτ⋅K[x(τ)]}⋅v(τ-1)]

    # iterate of sites in lattice
    @fastmath @inbounds for i in 1:Nsites

        # get matrix elements 
        expΔτμᵢ = expΔτμ[i]

        # iterate over time slices
        for τ in 1:Lτ

            # multiply vector element by matrix element
            Mv[get_index(τ,i,Lτ)] *= expΔτμᵢ
        end
    end

    # iterate over sites in lattice
    for i in 1:Nsites

        # Mv(1) = v(1) + B(1)⋅v(Lτ)
        index = get_index(1,i,Lτ)
        Mv[index] = v[index] + Mv[index]

        # iterate of time slices
        for τ in 2:Lτ

            # get index into vector
            index = get_index(τ,i,Lτ)

            # Mv(τ) = v(τ) - B(τ)⋅v(τ-1)
            Mv[index] = v[index] - Mv[index]
        end
    end

    return nothing
end


"""
Multiply by Mᵀ matrix.
"""
function mulMᵀ!(Mᵀv::AbstractVector{T2},ssh::SSHModel{T1,T2},v::AbstractVector{T1}) where {T1,T2}

    #####################################
    ## PERFORM MULTIPLICATION y = Mᵀ⋅v ##
    #####################################

    # Notes:
    # • y(τ)  = [Mᵀ⋅v](τ)  = v(τ)  - Bᵀ(τ+1)⋅v(τ+1) for τ < Lτ
    # • y(Lτ) = [Mᵀ⋅v](Lτ) = v(Lτ) + Bᵀ(1)⋅v(1)     for τ = Lτ
    # • Bᵀ(τ) = exp{-Δτ⋅K[x(τ)]}ᵀ⋅exp{Δτ⋅μ}
    # • exp{Δτ⋅μ} is represent an NxN diagonal matrix that is the same for all
    #   time slice τ and is stored as a vector.
    # • exp{-Δτ⋅K[x(τ)]} is applied to a vector/represented using the checkerboard approximation.

    # Calculate exp{Δτ⋅μ}⋅v(τ)

    # iterate of sites in lattice
    @fastmath @inbounds for i in 1:Nsites

        # get matrix elements 
        expΔτμᵢ = expΔτμ[i]

        # iterate over time slices
        for τ in 1:Lτ

            # get index into output vector
            iτ = get_index(τ,i,Lτ)

            # multiply vector element by matrix element
            Mv[iτ] *= expΔτμᵢ * v[iτ]
        end
    end

    # caluclate Bᵀ(τ+1)⋅v(τ+1) = exp{-Δτ⋅K[x(τ+1)]}ᵀ⋅[exp{Δτ⋅μ}⋅v(τ+1)]

    # iterate over bonds
    for bond in Nbonds:-1:1

        # get pair of sites associated with bond
        i = neighbor_table[1,bond]
        j = neighbor_table[2,bond]

        # iterate over time slices
        for τ in 1:Lτ

            # get index into output vector
            iτ = get_index(τ,i,Lτ)
            jτ = get_index(τ,j,Lτ)

            # get matrix elements of 2x2 checkerboard matrix associated with bond
            c = cosht[τ,bond]
            s = sinht[τ,bond]

            # apply checkarboard matrix
            val_iτ = Mv[iτ]
            val_jτ = Mv[jτ]
            Mv[iτ] = c*val_iτ + s*val_jτ
            Mv[jτ] = c*val_jτ + conj(s)*val_iτ
        end
    end

    # Calculate Mv(τ) = v(τ) - Bᵀ(τ+1)⋅v(τ+1)

    # iterate over sites in lattice
    for i in 1:Nsites

        # record Bᵀ(1)⋅v(1) value
        i1    = get_index(1,i,Lτ)
        Bᵀv_1 = Mv[i1]

        # iterate over time slices
        for τ in 1:Lτ-1

            # get index
            iτ   = get_index(τ,i,Lτ)
            iτp1 = get_index(τp1,i,Lτ)

            # Mv(τ) = v(τ) - Bᵀ(τ+1)⋅v(τ+1)
            Mv[iτ] = v[iτ] - Mv[iτp1]
        end

        # v(Lτ) + Bᵀ(1)⋅v(1)
        iLτ = get_index(Lτ,i,Lτ)
        Mv[iLτ] = v[iLτ] + Bᵀv_1
    end

    return nothing
end

"""
Calculates ⟨∂M/∂xᵢⱼ(τ)⟩ = uᵀ⋅[∂M/∂xᵢⱼ(τ)]⋅v for each xᵢⱼ(τ) degree of freedom,
writing each of the ⟨∂M/∂xᵢⱼ(τ)⟩ values to a vector dMdx.
"""
function muldMdx!(dMdx::AbstractVector{T2},u::AbstractVector{T2},ssh::SSHModel{T1,T2},v::AbstractVector{T2}) where {T1,T2}

    # Notes:
    # • B(τ)          = exp{Δτ⋅μ}⋅exp{-Δτ⋅K(τ)}
    # • ∂B(τ)/∂xᵢⱼ(τ) = -Δτ⋅∂K(τ)/∂xᵢⱼ(τ)⋅exp{Δτ⋅μ}⋅exp{-Δτ⋅K(τ)} = -Δτ⋅∂K(τ)/∂xᵢⱼ(τ)⋅B(τ)
    # • Kᵢⱼ(τ)        = -(tᵢⱼ-αᵢⱼ⋅xᵢⱼ(τ)) = matrix elements of NxN matrix K(τ)
    # • ∂K(τ)/∂xᵢⱼ(τ) = an NxN matrix with two non-zero matrix elements given by ∂Kᵢⱼ(τ)/∂xᵢⱼ(τ) = αᵢⱼ
    #
    # • ⟨∂M/∂xᵢⱼ(τ)⟩ = uᵀ⋅[∂M/∂xᵢⱼ(τ)]⋅v                       for τ > 1
    #                = uᵀ(τ)⋅[-∂B/∂xᵢⱼ(τ)]⋅v(τ-1)              for τ > 1
    #                = uᵀ(τ)⋅[Δτ⋅∂K(τ)/∂xᵢⱼ(τ)⋅B(τ)]⋅v(τ-1)    for τ > 1
    #                = +Δτ⋅uᵀ(τ)⋅[∂K(τ)/∂xᵢⱼ(τ)]⋅[B(τ)⋅v(τ-1)] for τ > 1
    #
    # • ⟨∂M/∂xᵢⱼ(1)⟩ = uᵀ⋅[∂M/∂xᵢⱼ(1)]⋅v                      for τ = 1
    #                = uᵀ(1)⋅[+∂B/∂xᵢⱼ(1)]⋅v(Lτ)              for τ = 1
    #                = uᵀ(1)⋅[-Δτ⋅∂K(1)/∂xᵢⱼ(τ)⋅B(1)]⋅v(Lτ)   for τ = 1
    #                = -Δτ⋅uᵀ(1)⋅[∂K(1)/∂xᵢⱼ(τ)]⋅[B(1)⋅v(Lτ)] for τ = 1

    @unpack cosht, sinht, neighbor_table, Nbonds, Ndof, Nsites, Lτ, Δτ, expΔτμ, α = ssh
    @unpack field_to_phonon, field_to_τ, phonon_to_bond = ssh

    ################################
    ## FIRST CALUCATE B(τ)⋅v(τ-1) ##
    ################################

    # vector to represent B(τ)⋅v(τ-1)
    Bv = ssh.ytemp

    # Calculate exp{-Δτ⋅K[x(τ)]}⋅v(τ-1)
    
    # iterate over bonds
    @fastmath @inbounds for bond in 1:Nbonds

        # get pair of sites associated with bond
        i = neighbor_table[1,bond]
        j = neighbor_table[2,bond]

        # iterate over time slices
        @simd for τ in 1:Lτ

            # τ-1
            τm1 = mod1(τ-1,Lτ)

            # get index into input vector
            iτm1 = get_index(τm1,i,Lτ)
            jτm1 = get_index(τm1,j,Lτ)

            # get index into output vector
            iτ = get_index(τ,i,Lτ)
            jτ = get_index(τ,j,Lτ)

            # get matrix elements of 2x2 checkerboard matrix associated with bond
            c = cosht[τ,bond]
            s = sinht[τ,bond]

            # perform multiplication
            Bv[iτ] = c*v[iτm1] + s*v[jτm1]
            Bv[jτ] = c*v[jτm1] + conj(s)*s[iτm1]
        end
    end

    # Finish calculation of B(τ)⋅v(τ-1) = exp{Δτ⋅μ}⋅[exp{-Δτ⋅K[x(τ)]}⋅v(τ-1)]

    # iterate of sites in lattice
    @fastmath @inbounds for i in 1:Nsites

        # get matrix element associated with i'th site in lattice
        expΔτμᵢ = expΔτμ[i]

        # iterate over time slices
        for τ in 1:Lτ

            # multiply vector element by matrix element
            Bv[get_index(τ,i,Lτ)] *= expΔτμᵢ
        end
    end

    #######################################
    ## CALCULATE ALL ⟨∂M/∂xᵢⱼ(τ)⟩ VALUES ##
    #######################################

    # iterate over degrees of freedom
    for field in 1:Ndof

        # field to phonon
        phonon = field_to_phonon[field]

        # field to τ
        τ = field_to_τ[field]

        # phonon to bond
        bond = phonon_to_bond[phonon]

        # electron-phonon coupling associated with phonon
        αᵢⱼ = α[phonon]

        # the pair of sites associated with the bond/phonon
        i = neighbor_table[1,bond]
        j = neighbor_table[2,bond]

        # get index into vectors associated (i,τ) and (j,τ) space-time coordinates
        iτ = get_index(τ,i,Lτ)
        jτ = get_index(τ,j,Lτ)

        # ⟨∂M/∂xᵢⱼ(τ)⟩ = +Δτ⋅uᵀ(τ)⋅[∂K(τ)/∂xᵢⱼ(τ)]⋅[B(τ)⋅v(τ-1)]
        dMdx[field] = Δτ * (α*u[iτ]*Bv[jτ] + conj(α)*u[jτ]*Bv[iτ])

        # ⟨∂M/∂xᵢⱼ(1)⟩ = -Δτ⋅uᵀ(1)⋅[∂K(1)/∂xᵢⱼ(1)]⋅[B(1)⋅v(Lτ)]
        if τ==1
            dMdx[field] = -dMdx[field]
        end
    end

    return nothing
end