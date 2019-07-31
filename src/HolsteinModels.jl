module HolsteinModels

using Statistics

using Langevin.Geometries: Geometry
using Langevin.Lattices: Lattice, translationally_equivalent_sets, sort_neighbor_table!
using Langevin.QuantumLattices: QuantumLattice, view_by_site, view_by_τ
using Langevin.Checkerboard: checkerboard_order, checkerboard_groups

export HolsteinModel
export assign_μ!, assign_ω!, assign_λ!
export assign_tij!, assign_ωij!, assign_λij!
export setup_checkerboard!, construct_expnΔτV!

mutable struct HolsteinModel{T<:AbstractFloat}

    ################################
    ## COMPLETE MODEL HAMILTONIAN ##
    ################################

    # H =  ∑ Pᵢ²/2 + ∑ (ωᵢ²/2) ϕᵢ² [Einstein Phonons]
    #   +  ∑ λᵢ ϕᵢ nᵢ              [El-Ph Coupling]
    #   +  ∑ ωᵢⱼ(ϕᵢ ± ϕⱼ)²         [Phonon Dispersion]
    #   +  ∑ λᵢⱼ ϕᵢ nⱼ             [Extended El-Ph Coupling]
    #   -  ∑ μᵢ nᵢ                 [Chemical Potential]
    #   -  ∑ tᵢⱼ(c⁺ᵢcⱼ + h.c.)     [Electron Kinetic Energy]

    #######################################
    ## FOR REPRESENTING LATTICE GEOMETRY ##
    #######################################

    "represents lattice geometry"
    geom::Geometry{T}

    "represents lattice"
    lattice::Lattice{T}

    "represents D+1 dimensional lattice resulting from Suzuki-Trotter approximation"
    qlattice::QuantumLattice{T}

    "stores sets of translationally equivalent pairs of sites in lattice."
    trans_equiv_sets::Array{Int32,7}

    #####################################################################
    ## VECTOR REPRESENTING EXPONENTIATED INTERACTION MATRIX exp(-Δτ⋅V) ##
    #####################################################################

    "stores as a vector the diagonal matrix exp(-Δτ⋅V[ϕ])"
    expnΔτV::Vector{Complex{T}}

    ############################
    ## HOLSTEIN PHONON FIELDS ##
    ############################

    "phonon fields stored in the order [ϕ₁(τ=1),ϕ₁(τ=2),...,ϕ₁(τ=Lτ),...,ϕₙ(τ=1),...,ϕₙ(τ=Lτ)]
    where n is the number of sites in the lattice."
    ϕ::Vector{Complex{T}}

    ################################
    ## SPECIFIES ON-SITE ENERGIES ##
    ################################

    "chemical potential for each site in lattice"
    μ::Vector{Complex{T}}

    #############################################################
    ## SPECIFIES ELECTRON KINETIC ENERGY (TIGHT BINDING MODEL) ##
    #############################################################

    "electron hopping energies in tight binding model"
    tij::Vector{Complex{T}}

    "cosh of electron hopping parameters in tij"
    coshtij::Vector{Complex{T}}

    "sinh of electron hopping parameters in tij"
    sinhtij::Vector{Complex{T}}

    "neighboring sites in tight binding model"
    neighbor_table_tij::Matrix{Int}

    ##############################
    ## SPECIFIES HOLSTEIN MODEL ##
    ##############################

    "frequency of each phonon"
    ω::Vector{Complex{T}}

    "local electron-phonon coupling"
    λ::Vector{Complex{T}}

    ###################################
    ## SPECIFIES HOLSTEIN DISPERSION ##
    ###################################

    "extended holstein frequency of the form ωij(ϕᵢ±ϕⱼ)²"
    ωij::Vector{Complex{T}}

    "specifies which two sites i,j that are coupled in ωij(ϕᵢ±ϕⱼ)²"
    neighbor_table_ωij::Matrix{Int}

    "specifies the sign: ωij(ϕᵢ+ϕⱼ)² or ωij(ϕᵢ-ϕⱼ)²"
    sign_ωij::Vector{Int}

    ###############################################
    ## SPECIFIES HOLSTEIN LONG RANGE INTERACTION ##
    ###############################################

    "extended holstein el-pho coupling of the form λij(ϕᵢ⋅nⱼ)"
    λij::Vector{Complex{T}}

    "sepcifies which two sites neighbor_table_λij[i,j] in term λij(ϕᵢ⋅nⱼ)"
    neighbor_table_λij::Matrix{Int}

    ##############################
    ## TEMPORARY STORAGE VECTOR ##
    ##############################

    "When doing multiplication y = Mᵀ⋅y' = MᵀM⋅v, you need a temperorary
    vector y' to store an intermediate result. This is that vector."
    temporary_vector::Vector{Complex{T}}

    #######################
    ## INNER CONSTRUCTOR ##
    #######################

    """
    Constructor for Holstein type.
    """
    function HolsteinModel(geom::Geometry{T},lattice::Lattice{T},β::T,Δτ::T) where {T<:AbstractFloat}

        # constructing translationally equivalent sets of sites
        trans_equiv_sets = Array{Int32,7}(translationally_equivalent_sets(lattice))

        # cosntruct quantum lattice object
        qlattice = QuantumLattice(lattice,β,Δτ)

        # number of sites in lattice
        nsites = lattice.nsites

        # initialize ineraction matrix, which is diagonal so stored as vector
        expnΔτV = zeros(Complex{T},qlattice.nindices)

        # intializing phonon fields to zero
        ϕ = zeros(Complex{T},qlattice.nindices)

        # initializing all on-site energies to zero
        μ = zeros(Complex{T},nsites)

        # initialize hopping parameters to empty vector
        tij = Vector{Complex{T}}(undef,0)

        # initialize hopping parameters to empty vector
        coshtij = Vector{Complex{T}}(undef,0)

        # initialize hopping parameters to empty vector
        sinhtij = Vector{Complex{T}}(undef,0)

        # initializing empty matrix to contain tight binding model neighbor_table
        neighbor_table_tij = Matrix{Int}(undef,2,0)

        # intializing phonon frequencies to zero
        ω = zeros(Complex{T},nsites)

        # initialize electron-phonon coupling to zero
        λ = zeros(Complex{T},nsites)

        # initizlize empty vector for inter-site phonon frequencies
        ωij = Vector{Complex{T}}(undef,0)

        # intialize empty matrix for storing inter-site phonon frequency neighbor_table
        neighbor_table_ωij = Matrix{Int}(undef,2,0)

        # intialize empty vector for sign_ωij
        sign_ωij = Vector{Int}(undef,0)

        # initialize empty vector for λij
        λij = Vector{Complex{T}}(undef,0)

        # initialize empty matrix vector for neighbor_table_λij
        neighbor_table_λij = Matrix{Int}(undef,2,0)

        # temporary vector
        temporary_vector = zeros(Complex{T},qlattice.nindices)

        # constructing holstein model
        new{T}(geom, lattice, qlattice, trans_equiv_sets, expnΔτV, ϕ, μ, tij, coshtij, sinhtij, neighbor_table_tij,ω, λ,
               ωij, neighbor_table_ωij, sign_ωij, λij, neighbor_table_λij, temporary_vector)

    end

end

#####################
## PRETTY PRINTING ##
#####################

function Base.show(io::IO, holstein::HolsteinModel)

    printstyled("HolsteinModel{",typeof(holstein.ω[1]),"}\n";bold=true,color=:cyan)
    print('\n')
    print(holstein.geom)
    print('\n')
    print('\n')
    print(holstein.lattice)
    print('\n')
    print('\n')
    print(holstein.qlattice)
    print('\n')
    printstyled("Parameters\n";bold=true)
    print('\n')
    println("•trans_equiv_sets: ", typeof(holstein.trans_equiv_sets),size(holstein.trans_equiv_sets))
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
    if length(holstein.λij)>0
        print('\n')
        _print_nonlocal_param(holstein.λij,holstein.neighbor_table_λij,"λij")
    end
end

function _print_local_param(orbits::Vector,vals::Vector,param::String)

    for orbit in 1:maximum(orbits)
        avg = mean(vals[orbits.==orbit])
        sd = std(vals[orbits.==orbit])
        println("•", param, ", orbit = ", orbit, ", mean = ",avg,", std = ",sd)
    end
    return nothing
end

function _print_nonlocal_param(vals::Vector,neighbor_table::Matrix{Int},param::String)

    println("•",param,": ", typeof(vals),size(vals), ", mean = ", mean(vals), ", std = ", std(vals))
    print('\n')
    println("•neighbor_table_", param, " =")
    show(IOContext(stdout, :limit => true), "text/plain", neighbor_table)
    print('\n')
    return nothing
end

#############################################################################
## DEFINING METHODS TO INCREMENTALLY SPECIFY THE HOLSTEIN MODEL PARAMETERS ##
#############################################################################

# GENERATE THE FOLLOWING FUNCTIONS: assign_μ!, assign_ω!, assign_λ!
for param in [:μ,:ω,:λ]

    # constructing symbol for function name
    op = Symbol(:assign_,param,:!)

    # defining functions
    @eval begin
        function $op(holstein::HolsteinModel{T},μ0::Number,σ0::Number,orbit::Int=0) where {T<:AbstractFloat}
            
            # getting phase of complex number
            phase = angle(μ0)

            # getting maginiute of complex number
            mag = abs(μ0)

            if orbit==0 # assigning parameter values for all sites
                holstein.$param .= mag .+ σ0 .* randn(length(holstein.$param))
            else # assigning paramerter values for only sites of certain kind of orbital
                for i in 1:length(holstein.$param)
                    if holstein.lattice.site_to_orbit[i]==orbit
                        holstein.$param[i] = mag + σ0 * randn()
                    end
                end
            end

            # reapplying phase of values
            holstein.$param .*= exp(phase*im)

            return nothing
        end 
    end

end


# GENERATE THE FOLLOWING FUNCTIONS: assign_tij!, assign_ωij!, assign_λij!
for param in [ :tij , :ωij , :λij ]
    
    # defining symbol for function name
    op = Symbol(:assign_,param,:!)
    
    # symbol for name of neighbor table
    neighbor_table = Symbol(:neighbor_table_,param)
    
    # defining functions
    @eval begin
        function $op(holstein::HolsteinModel{T}, μ0::Number, σ0::Number, orbit1::Int, orbit2::Int, displacement::Vector{Int}) where {T<:AbstractFloat}
            
            # phase of μ0
            phase = angle(μ0)
            
            # amplitude of μ0
            mag = abs(μ0)
            
            ## recasting 0 displacement in unit cell to displacement by L (width of lattice)
            # so that indexing in next step works
            for i in 1:3
                if displacement[i] == 0
                    displacement[i] = holstein.lattice.dims[i]
                end
            end 
            
            # getting new neighbors
            newneighbors = holstein.trans_equiv_sets[:,:,orbit2,orbit1,displacement[1],displacement[2],displacement[3]]
            
            # getting number of new neighbors
            nnewneighbors = size(newneighbors,2)
            
            # adding new neighbors to neighbor table
            holstein.$neighbor_table = hcat(holstein.$neighbor_table,newneighbors)
            
            # getting parameter value associated with each new neighbor
            for i in 1:nnewneighbors
                push!( holstein.$param , mag + σ0 * randn() )
            end
            
            # total number of neighbors
            nneighbors = size(holstein.$neighbor_table,2)
            
            # reapplying phase of complex number
            holstein.$param[nneighbors-nnewneighbors+1:end] .*= exp(im*phase)
            
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
    append!( holsteinmodel.sign_ωij , fill(Int,sgn,nnewneighbors) )
    
    return nothing
end

#######################################################
## MORE FUNCTIONS ASSOCIATED WITH HOLSTEINMODEL TYPE ##
#######################################################

"""
    function setup_checkerboard!(holstein::HolsteinModel{T}) where {T<:AbstractFloat}

Function for sorting the hopping parameters in HolsteinModel and calculating the
cosh and sinh of the hopping parameters so that the checkerboard decomposition
is ready to go.
"""
function setup_checkerboard!(holstein::HolsteinModel{T}) where {T<:AbstractFloat}

    # sort neighbor_table_tij
    perm = sort_neighbor_table!(holstein.neighbor_table_tij)
    @. holstein.tij = holstein.tij[perm]

    # get checkerboard groups
    groups = checkerboard_groups(holstein.neighbor_table_tij)

    # get checkerboard permutation
    perm = checkerboard_order(groups)

    # applying permutation
    @. holstein.neighbor_table_tij = holstein.neighbor_table_tij[:,perm]
    @. holstein.tij = holstein.tij[perm]

    # calculate cosh and sinh of hopping parameters
    holstein.coshtij = @. cosh(holstein.qlattice.Δτ*holstein.tij)
    holstein.sinhtij = @. sinh(holstein.qlattice.Δτ*holstein.tij)

    return nothing
end

"""
    function construct_expnΔτV!(holstein::HolsteinModel)

Constructs the exponentiated interaction matrix for the Holstein Model
exp(-Δτ⋅V[ϕ]) based on the current phonon fields ϕ. Note that the matrix
exp(-Δτ⋅V[ϕ]) is stored as a vector as it is a diagonal matrix.
"""
function construct_expnΔτV!(holstein::HolsteinModel{T}) where {T<:AbstractFloat}

    expnΔτV = holstein.expnΔτV::Vector{Complex{T}}
    λ       = holstein.λ::Vector{Complex{T}}
    μ       = holstein.μ::Vector{Complex{T}}
    ϕ       = holstein.ϕ::Vector{Complex{T}}
    Δτ      = holstein.qlattice.Δτ::T
    Lτ      = holstein.qlattice.Lτ::Int
    nsites  = holstein.lattice.nsites::Int
    λi::Complex{T} = 0.0
    μi::Complex{T} = 0.0

    ###############################################################
    ## UPDATING exp(-Δτ⋅V[ϕ]) BASED ON ON-SITE EL-PH INTERACTION ##
    ###############################################################

    # iterate over sites in lattice
    for i in 1:holstein.lattice.nsites
        # get a view into the phonon fields associated with current site
        ϕi = view_by_site(ϕ,i,nsites)
        # getting a view into matrix elements associated with current site
        expnΔτVi = view_by_site(expnΔτV,i,nsites)
        # get parameter values for current site
        λi = λ[i]
        μi = μ[i]
        # iterate over time slices
        for τ in 1:Lτ
            # setting matrix element
            expnΔτVi[τ] = exp(-Δτ*(λi*ϕi[τ]-μi))
        end
    end

    ##################################################################
    ## UPDATING exp(-Δτ⋅V[ϕ]) BASED ON LONG-RANGE EL-PH INTERACTION ##
    ##################################################################

    # checking if any long-range couplings are defined
    if length(holstein.λij)>0
        λij = holstein.λij
        neighbor_table_λij = holstein.neighbor_table_λij
        i = 0
        j = 0
        # iterating over defined long range interaction
        for l in 1:length(λij)
            # get pairs of interaction sites i,j
            i = neighbor_table_λij[1,l]
            j = neighbor_table_λij[2,l]
            # getting a view into the phonon fields form site i
            ϕi = view_by_site(ϕ,i,nsites)
            # getting view into matrix elements associated with site j
            expnΔτVj = view_by_site(expnΔτV,j,nsites)
            # updating all matrix elements associated with site j
            @. expnΔτVj += exp(-Δτ*λij[l]*ϕi)
        end
    end

    return nothing
end

##############################################
## Include Functionality to Handle M Matrix ##
##############################################

include("HolsteinModelMatrix.jl")

end