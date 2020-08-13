using Statistics
using Printf

using ..UnitCells: UnitCell
using ..Lattices: Lattice, sort_neighbor_table!, loc_to_site, calc_neighbor_table
using ..Checkerboard: checkerboard_order, checkerboard_groups
using ..IterativeSolvers: GMRES, ConjugateGradient, BiCGStab
using ..Utilities: get_index

export HolsteinModel
export assign_μ!, assign_ω!, assign_λ!, assign_ω4!
export assign_tij!, assign_ωij!
export get_index, get_site, get_τ
export setup_checkerboard!, construct_expnΔτV!
export write_phonons, read_phonons

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
    nsites::Int

    "size of D+1 dimensional lattice"
    nindices::Int

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

    #####################################################
    ## SPECIFIES ON-SITE ENERGIES (CHEMICAL POTENTIAL) ##
    #####################################################

    "chemical potential for each site in lattice"
    μ::Vector{T1}

    #############################################################
    ## SPECIFIES ELECTRON KINETIC ENERGY (TIGHT BINDING MODEL) ##
    #############################################################

    "electron hopping energies in tight binding model"
    tij::Vector{T2}

    "cosh of electron hopping parameters in tij"
    coshtij::Vector{T2}

    "sinh of electron hopping parameters in tij"
    sinhtij::Vector{T2}

    "neighboring sites in tight binding model"
    neighbor_table_tij::Matrix{Int}

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
    function HolsteinModel(lattice::Lattice{T}, β::T, Δτ::T;
                           is_complex::Bool=false, iterativesolver::String="cg", tol::T=1e-4, maxiter::Int=10000, restart::Int=-1) where {T<:AbstractFloat}

        # calculating length of imaginary time axis
        Lτ = round(Int,β/Δτ)

        # number of sites in lattice
        nsites = lattice.nsites

        # size of D+1 dimensional lattice
        nindices = nsites*Lτ

        # initialize ineraction matrix, which is diagonal so stored as vector
        expnΔτV = zeros(T,nindices)

        # intializing phonon fields to zero
        x = zeros(T,nindices)

        # initializing all on-site energies to zero
        μ = zeros(T,nsites)

        # initialize hopping parameters to empty vector
        tij = Vector{T}(undef,0)

        # initialize hopping parameters to empty vector
        coshtij = Vector{T}(undef,0)

        # initialize hopping parameters to empty vector
        sinhtij = Vector{T}(undef,0)

        # initializing empty matrix to contain tight binding model neighbor_table
        neighbor_table_tij = Matrix{Int}(undef,2,0)

        # intializing phonon frequencies to zero
        ω = zeros(T,nsites)

        # initialize electron-phonon coupling to zero
        λ = zeros(T,nsites)

        # initialize anharmonic X^4 term coefficient
        ω4 = zeros(T,nsites)

        # initizlize empty vector for inter-site phonon frequencies
        ωij = Vector{T}(undef,0)

        # intialize empty matrix for storing inter-site phonon frequency neighbor_table
        neighbor_table_ωij = Matrix{Int}(undef,2,0)

        # intialize empty vector for sign_ωij
        sign_ωij = Vector{Int}(undef,0)

        # temporary vectors
        ytemp = zeros(T,nindices)

        # if true multiply by Mᵀ instead of M
        transposed = false

        # temporary vector
        if is_complex
            Mᵀg = zeros(Complex{T},nindices)
        else
            Mᵀg = zeros(T,nindices)
        end

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

        # constructing holstein model
        if is_complex
            new{T,Complex{T},typeof(solver)}(β, Δτ, Lτ, nsites, nindices, lattice, x, expnΔτV,
                                             μ, tij, coshtij, sinhtij, neighbor_table_tij,
                                             ω, λ, ω4, ωij, neighbor_table_ωij, sign_ωij,
                                             ytemp, Mᵀg, mul_by_M, transposed, solver)
        else
            new{T,T,typeof(solver)}(β, Δτ, Lτ, nsites, nindices, lattice, x, expnΔτV,
                                    μ, tij, coshtij, sinhtij, neighbor_table_tij,
                                    ω, λ, ω4, ωij, neighbor_table_ωij, sign_ωij,
                                    ytemp, Mᵀg, mul_by_M, transposed, solver)
        end
    end

end

#############################################################################
## DEFINING METHODS TO INCREMENTALLY SPECIFY THE HOLSTEIN MODEL PARAMETERS ##
#############################################################################

# GENERATE THE FOLLOWING FUNCTIONS: assign_μ!, assign_ω!, assign_λ!, assign_ω4!
for param in [ :μ , :ω , :λ, :ω4 ]

    # constructing symbol for function name
    op = Symbol(:assign_,param,:!)

    # defining functions
    @eval begin
        function $op(holstein::HolsteinModel,μ0::T,σ0::T,orbit::Int=0) where {T<:AbstractFloat}

            if orbit==0 # assigning parameter values for all sites
                R = randn(length(holstein.$param))
                @. holstein.$param = μ0 + σ0 * R
            else # assigning paramerter values for only sites of certain kind of orbital
                for i in 1:length(holstein.$param)
                    if holstein.lattice.site_to_orbit[i]==orbit
                        holstein.$param[i] = μ0 + σ0 * randn()
                    end
                end
            end

            return nothing
        end
    end

end


# GENERATE THE FOLLOWING FUNCTIONS: assign_tij!, assign_ωij!
for param in [ :tij , :ωij ]

    # defining symbol for function name
    op = Symbol(:assign_,param,:!)

    # symbol for name of neighbor table
    neighbor_table = Symbol(:neighbor_table_,param)

    # defining functions when parameter value is complex
    @eval begin
        function $op(holstein::HolsteinModel, μ0::T2, σ0::T1, orbit1::Int, orbit2::Int, displacement::Vector{Int}) where {T1<:AbstractFloat,T2<:Complex}

            # phase of μ0
            phase = angle(μ0)

            # amplitude of μ0
            mag = abs(μ0)

            # getting total number of neighbors before new neighbors added
            nneighbors = length(holstein.$param)

            # getting parameters values ignoring complex phase
            $op(holstein,mag,σ0,orbit1,orbit2,displacement)

            # reapplying phase of complex number
            holstein.$param[nneighbors+1:end] .*= exp(im*phase)

            return nothing
        end
    end

    # defining functions when parameter value is real
    @eval begin
        function $op(holstein::HolsteinModel, μ0::T, σ0::T, orbit1::Int, orbit2::Int, displacement::Vector{Int}) where {T<:AbstractFloat}

            # getting new neighbors accounting for periodic boundary conditions
            newneighbors = calc_neighbor_table(holstein.lattice,orbit1,orbit2,displacement)

            # getting number of new neighbors
            nnewneighbors = size(newneighbors,2)

            # adding new neighbors to neighbor table
            holstein.$neighbor_table = hcat(holstein.$neighbor_table,newneighbors)

            # getting parameter value associated with each new neighbor
            for i in 1:nnewneighbors
                push!( holstein.$param , μ0 + σ0 * randn() )
            end

            return nothing
        end
    end

end


# adding functionality to assign_ωij! function so that the array holsteinmodel.sign_ωij is also modified
function assign_ωij!(holstein::HolsteinModel, μ0::Number, σ0::Number, sgn::Int, orbit1::Int, orbit2::Int, displacement::Vector{Int})

    @assert abs(sgn)==1

    # updating neighbor table and parameter values
    assign_ωij!(holstein,μ0,σ0,orbit1,orbit2,displacement)

    # number of new neighbors constructed
    nnewneighbors = div(holstein.lattice.nsites,holstein.lattice.unit_cell.norbits)

    # modifying holsteinmodel.sign_ωij array
    append!( holstein.sign_ωij , fill(Int,sgn,nnewneighbors) )

    return nothing
end

#######################################################
## MORE FUNCTIONS ASSOCIATED WITH HOLSTEINMODEL TYPE ##
#######################################################

"""
    function setup_checkerboard!(holstein::HolsteinModel{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

Function for sorting the hopping parameters in HolsteinModel and calculating the
cosh and sinh of the hopping parameters so that the checkerboard decomposition
is ready to go.
"""
function setup_checkerboard!(holstein::HolsteinModel)

    if length(holstein.tij)>0

        # sort neighbor_table_tij
        perm = sort_neighbor_table!(holstein.neighbor_table_tij)
        holstein.tij .= holstein.tij[perm]

        # get checkerboard groups
        groups = checkerboard_groups(holstein.neighbor_table_tij)

        # get checkerboard permutation
        perm .= checkerboard_order(groups)

        # applying permutation
        holstein.neighbor_table_tij .= holstein.neighbor_table_tij[:,perm]
        holstein.tij .= holstein.tij[perm]

        # calculate cosh and sinh of hopping parameters
        holstein.coshtij = @. cosh(holstein.Δτ*holstein.tij)
        holstein.sinhtij = @. sinh(holstein.Δτ*holstein.tij)
    end

    return nothing
end

"""
    function construct_expnΔτV!(holstein::HolsteinModel)

Constructs the exponentiated interaction matrix for the Holstein Model
exp(-Δτ⋅V[x]) based on the current phonon fields x. Note that the matrix
exp(-Δτ⋅V[x]) is stored as a vector as it is a diagonal matrix.
"""
function construct_expnΔτV!(holstein::HolsteinModel)

    expnΔτV  = holstein.expnΔτV
    λ        = holstein.λ
    μ        = holstein.μ
    x        = holstein.x
    Δτ       = holstein.Δτ
    Lτ       = holstein.Lτ
    nsites   = holstein.nsites

    # iterating over time slices
    @inbounds @fastmath for i in 1:nsites
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

##################################################
## Functionality for Phonon Field Read/Write IO ##
##################################################

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
    construct_expnΔτV!(holstein)

    return nothing
end