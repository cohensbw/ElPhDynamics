module HolsteinModels

using Statistics
using IterativeSolvers
using Printf

using ..Geometries: Geometry
using ..Lattices: Lattice, translationally_equivalent_sets, sort_neighbor_table!, loc_to_site
using ..Checkerboard: checkerboard_order, checkerboard_groups
using ..RestartedGMRES: GMRES, solve!
using ..Utilities: get_index

export HolsteinModel
export assign_μ!, assign_ω!, assign_λ!
export assign_tij!, assign_ωij!
export get_index, get_site, get_τ
export setup_checkerboard!, construct_expnΔτV!
export write_phonons, read_phonons

mutable struct HolsteinModel{ T1<:AbstractFloat , T2<:Union{Float32,Float64,Complex{Float32},Complex{Float64}} }

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

    "represents lattice geometry"
    geom::Geometry{T1}

    "represents lattice"
    lattice::Lattice{T1}

    "stores sets of translationally equivalent pairs of sites in lattice."
    trans_equiv_sets::Array{Int,7}

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

    ###################################
    ## SPECIFIES HOLSTEIN DISPERSION ##
    ###################################

    "extended holstein frequency of the form ωij(xᵢ±xⱼ)²"
    ωij::Vector{T2}

    "specifies which two sites i,j that are coupled in ωij(xᵢ±xⱼ)²"
    neighbor_table_ωij::Matrix{Int}

    "specifies the sign: ωij(xᵢ+xⱼ)² or ωij(xᵢ-xⱼ)²"
    sign_ωij::Vector{Int}

    #################################################
    ## VARIBALES FOR SOLVING M⋅x=g VIA ITERATIVELY ##
    #################################################

    "Tolerace when solve M⋅x=g iteratively."
    tol::T1

    "A vector of length `ninidces` to temporarily store data."
    ytemp::Vector{T2}

    "A vector for storing the temporary product Mᵀ⋅g needed for Conjugate Gradient method."
    Mᵀg::Vector{T2}

    "Stores state vectors for Conjugate Gradient algorithm so as to avoid
    extra memory allocations."
    cg_state_vars::CGStateVariables{T2,Vector{T2}}

    "Boolean to signify if GMRES should be used instead of Conjugate Gradient."
    use_gmres::Bool

    "GMRES type that preallocates memory for algorithm."
    gmres::GMRES{T1,T2}

    #######################
    ## INNER CONSTRUCTOR ##
    #######################

    """
    Constructor for Holstein type.
    """
    function HolsteinModel(geom::Geometry{T}, lattice::Lattice{T}, β::T, Δτ::T;
                           is_complex::Bool=false, tol::T=1e-4, use_gmres::Bool=false, restart::Int=-1) where {T<:AbstractFloat}

        # calculating length of imaginary time axis
        Lτ = round(Int,β/Δτ)

        # number of sites in lattice
        nsites = lattice.nsites

        # size of D+1 dimensional lattice
        nindices = nsites*Lτ

        # constructing translationally equivalent sets of sites
        trans_equiv_sets = translationally_equivalent_sets(lattice)

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

        # initizlize empty vector for inter-site phonon frequencies
        ωij = Vector{T}(undef,0)

        # intialize empty matrix for storing inter-site phonon frequency neighbor_table
        neighbor_table_ωij = Matrix{Int}(undef,2,0)

        # intialize empty vector for sign_ωij
        sign_ωij = Vector{Int}(undef,0)

        # temporary vectors
        ytemp = zeros(T,nindices)

        # constructing holstein model
        if is_complex

            # temporary vectors
            Mᵀg = zeros(Complex{T},nindices)

            # conjugate gradient state variables
            cg_state_vars = CGStateVariables(zeros(Complex{T},nindices),zeros(Complex{T},nindices),zeros(Complex{T},nindices))

            # GMRES type
            gmres = GMRES(Mᵀg,tol=tol,restart=restart)

            new{T,Complex{T}}(β, Δτ, Lτ, nsites, nindices, geom, lattice, trans_equiv_sets, x, expnΔτV,
                              μ, tij, coshtij, sinhtij, neighbor_table_tij,
                              ω, λ, ωij, neighbor_table_ωij, sign_ωij,
                              tol, ytemp, Mᵀg, cg_state_vars, use_gmres, gmres)
        else

            # temporary vectors
            Mᵀg = zeros(T,nindices)

            # conjugate gradient state variables
            cg_state_vars = CGStateVariables(zeros(T,nindices),zeros(T,nindices),zeros(T,nindices))

            # GMRES type
            gmres = GMRES(Mᵀg,tol=tol,restart=restart)

            new{T,T}(β, Δτ, Lτ, nsites, nindices, geom, lattice, trans_equiv_sets, x, expnΔτV,
                     μ, tij, coshtij, sinhtij, neighbor_table_tij,
                     ω, λ, ωij, neighbor_table_ωij, sign_ωij,
                     tol, ytemp, Mᵀg, cg_state_vars, use_gmres, gmres)
        end
    end

end

#####################
## PRETTY PRINTING ##
#####################

function Base.show(io::IO, holstein::HolsteinModel)

    type1 = typeof(holstein.ω).parameters[1]
    type2 = typeof(holstein.tij).parameters[1]
    printstyled( "HolsteinModel{" , type1 , "," , type2 , "}\n" ; bold=true , color=:cyan )
    print('\n')
    println("β = ",holstein.β)
    println("Δτ = ",holstein.Δτ)
    println("Lτ = ",holstein.Lτ)
    println("nsites = ",holstein.nsites)
    println("nindices = ",holstein.nindices)
    print('\n')
    print(holstein.geom)
    print('\n')
    print('\n')
    print(holstein.lattice)
    print('\n')
    printstyled("Parameters\n";bold=true)
    print('\n')
    println("trans_equiv_sets: ", typeof(holstein.trans_equiv_sets),size(holstein.trans_equiv_sets))
    print('\n')
    _print_local_param(holstein.lattice.site_to_orbit,holstein.μ,"μ")
    _print_local_param(holstein.lattice.site_to_orbit,holstein.ω,"ω")
    _print_local_param(holstein.lattice.site_to_orbit,holstein.λ,"λ")
    if length(holstein.tij)>0
        print('\n')
        _print_nonlocal_param(holstein.tij,holstein.neighbor_table_tij,"tij")
    end
    if length(holstein.ωij)>0
        print('\n')
        _print_nonlocal_param(holstein.ωij,holstein.neighbor_table_ωij,"ωij")
    end
end

function _print_local_param(orbits::Vector,vals::Vector,param::String)

    for orbit in 1:maximum(orbits)
        avg = mean(vals[orbits.==orbit])
        sd = std(vals[orbits.==orbit])
        println(param, ", orbit = ", orbit, ", mean = ",avg,", std = ",sd)
    end
    return nothing
end

function _print_nonlocal_param(vals::Vector,neighbor_table::Matrix{Int},param::String)

    println(param,": ", typeof(vals),size(vals), ", mean = ", mean(vals), ", std = ", std(vals))
    println("neighbor_table_", param, ": ", typeof(neighbor_table), size(neighbor_table))
    # show(IOContext(stdout, :limit => true), "text/plain", neighbor_table)
    print('\n')
    return nothing
end

#############################################################################
## DEFINING METHODS TO INCREMENTALLY SPECIFY THE HOLSTEIN MODEL PARAMETERS ##
#############################################################################

# GENERATE THE FOLLOWING FUNCTIONS: assign_μ!, assign_ω!, assign_λ!
for param in [ :μ , :ω , :λ ]

    # constructing symbol for function name
    op = Symbol(:assign_,param,:!)

    # defining functions
    @eval begin
        function $op(holstein::HolsteinModel{T},μ0::T,σ0::T,orbit::Int=0) where {T<:AbstractFloat}

            if orbit==0 # assigning parameter values for all sites
                holstein.$param .= μ0 .+ σ0 .* randn(length(holstein.$param))
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
        function $op(holstein::HolsteinModel{T1,T2}, μ0::T2, σ0::T1, orbit1::Int, orbit2::Int, displacement::Vector{Int}) where {T1<:AbstractFloat,T2<:Complex}

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
        function $op(holstein::HolsteinModel{T1,T2}, μ0::T1, σ0::T1, orbit1::Int, orbit2::Int, displacement::Vector{Int}) where {T1<:AbstractFloat,T2<:Number}

            # getting new neighbors
            newneighbors = holstein.trans_equiv_sets[:,:,displacement[1]+1,displacement[2]+1,displacement[3]+1,orbit2,orbit1]

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


# adding functionality to assign_ωij! function so that the
# array holsteinmodel.sign_ωij is also modified
function assign_ωij!(holstein::HolsteinModel, μ0::Number, σ0::Number, sgn::Int, orbit1::Int, orbit2::Int, displacement::Vector{Int})

    @assert abs(sgn)==1

    # updating neighbor table and parameter values
    assign_ωij!(holstein,μ0,σ0,orbit1,orbit2,displacement)

    # number of new neighbors constructed
    nnewneighbors = div(holstein.lattice.nsites,holstein.lattice.norbits)

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
function setup_checkerboard!(holstein::HolsteinModel{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

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
    function construct_expnΔτV!(holstein::HolsteinModel{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

Constructs the exponentiated interaction matrix for the Holstein Model
exp(-Δτ⋅V[x]) based on the current phonon fields x. Note that the matrix
exp(-Δτ⋅V[x]) is stored as a vector as it is a diagonal matrix.
"""
function construct_expnΔτV!(holstein::HolsteinModel{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

    expnΔτV  = holstein.expnΔτV::Vector{T2}
    λ        = holstein.λ::Vector{T1}
    μ        = holstein.μ::Vector{T1}
    x        = holstein.x::Vector{T1}
    Δτ       = holstein.Δτ::T1
    Lτ       = holstein.Lτ::Int
    nsites   = holstein.nsites::Int

    # iterating over time slices
    @inbounds @fastmath for site in 1:nsites
        # iterating over sites in lattice
        for τ in 1:Lτ
            # getting index in vector
            i = get_index(τ,site,Lτ)
            # updating matrix element exp{-Δτ⋅Vᵢᵢ(τ)} = exp{-Δτ⋅(λᵢ⋅xᵢ(τ)-μᵢ)}
            expnΔτV[i] = exp( -Δτ * ( λ[site] * x[i] - μ[site] ) )
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
                    for orbit in 1:lattice.norbits

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
function read_phonons(holstein::HolsteinModel{T1,T2},filename::String) where {T1<:AbstractFloat,T2<:Number}

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

    return nothing
end

##############################################
## Include Functionality to Handle M Matrix ##
##############################################

include("HolsteinModelMatrix.jl")

end
