module HolsteinModels

using Statistics
using IterativeSolvers

using Langevin.Geometries: Geometry
using Langevin.Lattices: Lattice, translationally_equivalent_sets, sort_neighbor_table!
using Langevin.Checkerboard: checkerboard_order, checkerboard_groups

export HolsteinModel
export assign_μ!, assign_ω!, assign_λ!
export assign_tij!, assign_ωij!
export setup_checkerboard!, construct_expnΔτV!

mutable struct HolsteinModel{ T1<:AbstractFloat , T2<:Union{Float32,Float64,Complex{Float32},Complex{Float64}} }

    ################################
    ## COMPLETE MODEL HAMILTONIAN ##
    ################################

    # H =  ∑ Pᵢ²/2 + ∑ (ωᵢ²/2) ϕᵢ² [Einstein Phonons]
    #   +  ∑ λᵢ ϕᵢ nᵢ              [El-Ph Coupling]
    #   +  ∑ ωᵢⱼ(ϕᵢ ± ϕⱼ)²         [Phonon Dispersion]
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

    "phonon fields stored in the order `[ϕ₁(τ=1),ϕ₂(τ=1),...,ϕₙ(τ=1),...,ϕ₁(τ=Lτ),...,ϕₙ(τ=Lτ)]`
    where `n` is the number of sites in the lattice and `Lτ` is the length of the imaginary time axis."
    ϕ::Vector{T1}

    ##############################################################
    ## VECTORS REPRESENTING MATRICE NEEDED TO LANGEVIN DYNAMICS ##
    ##############################################################

    "a vector representing the diagonal matrix exp(-Δτ⋅V[ϕ])"
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

    "extended holstein frequency of the form ωij(ϕᵢ±ϕⱼ)²"
    ωij::Vector{T2}

    "specifies which two sites i,j that are coupled in ωij(ϕᵢ±ϕⱼ)²"
    neighbor_table_ωij::Matrix{Int}

    "specifies the sign: ωij(ϕᵢ+ϕⱼ)² or ωij(ϕᵢ-ϕⱼ)²"
    sign_ωij::Vector{Int}

    ##############################################################
    ## TEMPORARY STORAGE VECTORS FOR MEMORY EFFICIENCY PURPOSES ##
    ##############################################################

    "A vector of length `ninidces` to temporarily store data."
    y′::Vector{T2}

    "A vector of length `nsites` to temporarily store the data for a single time slice."
    yτ′::Vector{T2}

    "A vector of length `Lτ` to temporarily store the data for a single site."
    yi′::Vector{T2}

    "Stores state vectors for Conjugate Gradient algorithm so as to avoid
    extra memory allocations."
    cg_state_vars::CGStateVariables{T2,Vector{T2}}

    #######################
    ## INNER CONSTRUCTOR ##
    #######################

    """
    Constructor for Holstein type.
    """
    function HolsteinModel(geom::Geometry{T},lattice::Lattice{T},β::T,Δτ::T,is_complex::Bool=false) where {T<:AbstractFloat}

        # calculating length of imaginary time axis
        Lτ = round(Int,β/Δτ)

        # number of sites in lattice
        nsites = lattice.nsites

        # size of D+1 dimensional lattice
        nindices = nsites*Lτ

        # constructing translationally equivalent sets of sites
        trans_equiv_sets = Array{Int32,7}(translationally_equivalent_sets(lattice))

        # initialize ineraction matrix, which is diagonal so stored as vector
        expnΔτV = zeros(T,nindices)

        # intializing phonon fields to zero
        ϕ = zeros(T,nindices)

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
        y′   = zeros(T,nindices)
        yτ′  = zeros(T,nsites)
        yi′  = zeros(T,Lτ)

        # constructing holstein model
        if is_complex

            # conjugate gradient state variables
            cg_state_vars = CGStateVariables(zeros(Complex{T},nindices),zeros(Complex{T},nindices),zeros(Complex{T},nindices))

            new{T,Complex{T}}(β, Δτ, Lτ, nsites, nindices, geom, lattice, trans_equiv_sets, ϕ, expnΔτV,
                              μ, tij, coshtij, sinhtij, neighbor_table_tij,
                              ω, λ, ωij, neighbor_table_ωij, sign_ωij, y′, yτ′, yi′, cg_state_vars)
        else

            # conjugate gradient state variables
            cg_state_vars = CGStateVariables(zeros(T,nindices),zeros(T,nindices),zeros(T,nindices))

            new{T,T}(β, Δτ, Lτ, nsites, nindices, geom, lattice, trans_equiv_sets, ϕ, expnΔτV,
                     μ, tij, coshtij, sinhtij, neighbor_table_tij,
                     ω, λ, ωij, neighbor_table_ωij, sign_ωij, y′, yτ′, yi′, cg_state_vars)
        end
    end

end

#####################
## PRETTY PRINTING ##
#####################

function Base.show(io::IO, holstein::HolsteinModel)

    printstyled( "HolsteinModel{" , typeof(holstein.ω[1]) , "," , typeof(holstein.tij[1]) , "}\n" ; bold=true , color=:cyan )
    print('\n')
    println("•β = ",holstein.β)
    println("•Δτ = ",holstein.Δτ)
    println("•Lτ = ",holstein.Lτ)
    println("•nsites = ",holstein.nsites)
    println("•nindices = ",holstein.nindices)
    print('\n')
    print(holstein.geom)
    print('\n')
    print('\n')
    print(holstein.lattice)
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


# GENERATE THE FOLLOWING FUNCTIONS: assign_tij!, assign_ωij!, assign_λij!
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
        function $op(holstein::HolsteinModel{T1,T2}, μ0::T1, σ0::T1, orbit1::Int, orbit2::Int, displacement::Vector{Int}) where {T1<:AbstractFloat,T2<:AbstractFloat}

            # recasting 0 displacement in unit cell to displacement by L (width of lattice)
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
                push!( holstein.$param , μ0 + σ0 * randn() )
            end

            # total number of neighbors
            nneighbors = size(holstein.$neighbor_table,2)

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
exp(-Δτ⋅V[ϕ]) based on the current phonon fields ϕ. Note that the matrix
exp(-Δτ⋅V[ϕ]) is stored as a vector as it is a diagonal matrix.
"""
function construct_expnΔτV!(holstein::HolsteinModel{T1,T2}) where {T1<:AbstractFloat,T2<:Number}

    expnΔτV  = holstein.expnΔτV::Vector{T2}
    λ        = holstein.λ::Vector{T1}
    μ        = holstein.μ::Vector{T1}
    ϕ        = holstein.ϕ::Vector{T1}
    Δτ       = holstein.Δτ::T1
    Lτ       = holstein.Lτ::Int
    nsites   = holstein.nsites::Int
    offset_τ = 0

    # iterating over time slices
    for τ in 1:Lτ
        # calculating the indexing offset associated with τ time slice
        offset_τ = (τ-1)*nsites
        # iterating over sites in lattice
        for i in 1:nsites
            # updating matrix element exp{-Δτ⋅Vᵢᵢ(τ)} = exp{-Δτ⋅(λᵢ⋅ϕᵢ(τ)-μᵢ)}
            expnΔτV[offset_τ+i] = exp( -Δτ * ( λ[i] * ϕ[offset_τ+i] - μ[i] ) )
        end
    end

    return nothing
end

##############################################
## Include Functionality to Handle M Matrix ##
##############################################

include("HolsteinModelMatrix.jl")

end
