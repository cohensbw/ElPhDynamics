module HolsteinModels

using Statistics

using Langevin.Geometries: Geometry
using Langevin.Lattices: Lattice, translationally_equivalent_sets
using Langevin.QuantumLattices: QuantumLattice

export HolsteinModel
export assign_μ!, assign_ω!, assign_λ!
export assign_tij!, assign_ωij!, assign_λij!

# allow for both Float64 and Complex{Float64} hopping parameters
mutable struct HolsteinModel{T<:Number}

    ################################
    ## COMPLETE MODEL HAMILTONIAN ##
    ################################

    # H =  ∑ Pᵢ²/2 + ∑ (ωᵢ²/2) ϕᵢ²  [Einstein Phonons]
    #   +  ∑ λᵢ ϕᵢ nᵢ              [El-Ph Coupling]
    #   +  ∑ ωᵢⱼ(ϕᵢ ± ϕⱼ)²         [Phonon Dispersion]
    #   +  ∑ λᵢⱼ ϕᵢ nⱼ             [Extended El-Ph Coupling]
    #   -  ∑ μᵢ nᵢ                 [Chemical Potential]

    #######################################
    ## FOR REPRESENTING LATTICE GEOMETRY ##
    #######################################

    "represents lattice geometry"
    geom::Geometry

    "represents lattice"
    lattice::Lattice

    "represents D+1 dimensional lattice resulting from Suzuki-Trotter approximation"
    qlattice::QuantumLattice

    ############################
    ## HOLSTEIN PHONON FIELDS ##
    ############################

    "phonon fields ϕᵢ[site,τ]"
    ϕ::Matrix{T}

    ################################
    ## SPECIFIES ON-SITE ENERGIES ##
    ################################

    "chemical potential for each site in lattice"
    μ::Vector{T}

    #############################################################
    ## SPECIFIES ELECTRON KINETIC ENERGY (TIGHT BINDING MODEL) ##
    #############################################################

    "electron hopping energies in tight binding model"
    tij::Vector{T}

    "neighboring sites in tight binding model"
    neighbor_table_tij::Matrix{Int}

    ##############################
    ## SPECIFIES HOLSTEIN MODEL ##
    ##############################

    "frequency of each phonon"
    ω::Vector{T}

    "local electron-phonon coupling"
    λ::Vector{T}

    ###################################
    ## SPECIFIES HOLSTEIN DISPERSION ##
    ###################################

    "extended holstein frequency of the form ωij(ϕᵢ±ϕⱼ)²"
    ωij::Vector{T}

    "specifies which two sites i,j that are coupled in ωij(ϕᵢ±ϕⱼ)²"
    neighbor_table_ωij::Matrix{Int}

    "specifies the sign: ωij(ϕᵢ+ϕⱼ)² or ωij(ϕᵢ-ϕⱼ)²"
    sign_ωij::Vector{Int}

    ###############################################
    ## SPECIFIES HOLSTEIN LONG RANGE INTERACTION ##
    ###############################################

    "extended holstein el-pho coupling of the form λij(ϕᵢ⋅nⱼ)"
    λij::Vector{T}

    "sepcifies which two sites neighbor_table_λij[i,j] in term λij(ϕᵢ⋅nⱼ)"
    neighbor_table_λij::Matrix{Int}

    #######################
    ## INNER CONSTRUCTOR ##
    #######################

    """
    Constructor for Holstein type.
    """
    function HolsteinModel(geom::Geometry{T},lattice::Lattice{T},β::Float64,Δτ::Float64,is_complex::Bool) where {T<:AbstractFloat}

        # cosntruct quantum lattice object
        qlattice = QuantumLattice(lattice,β,Δτ)

        # data type of parameter
        Tp = T
        if is_complex
            Tp = Complex{T}
        end

        # number of sites in lattice
        nsites = lattice.nsites

        # intializing phonon fields to zero
        ϕ = zeros(T,nsites,qlattice.Lτ)

        # initializing all on-site energies to zero
        μ = zeros(Tp,nsites)

        # initialize hopping parameters to empty vector
        tij = Vector{Tp}(undef,0)

        # initializing empty matrix to contain tight binding model neighbor_table
        neighbor_table_tij = Matrix{Int}(undef,2,0)

        # intializing phonon frequencies to zero
        ω = zeros(Tp,nsites)

        # initialize electron-phonon coupling to zero
        λ = zeros(Tp,nsites)

        # initizlize empty vector for inter-site phonon frequencies
        ωij = Vector{Tp}(undef,0)

        # intialize empty matrix for storing inter-site phonon frequency neighbor_table
        neighbor_table_ωij = Matrix{Int}(undef,2,0)

        # intialize empty vector for sign_ωij
        sign_ωij = Vector{Int}(undef,0)

        # initialize empty vector for λij
        λij = Vector{Tp}(undef,0)

        # initialize empty matrix vector for neighbor_table_λij
        neighbor_table_λij = Matrix{Int}(undef,2,0)

        # constructing holstein model
        new{Tp}(geom, lattice, qlattice,
                ϕ, μ, 
                tij, neighbor_table_tij,
                ω, λ,
                ωij, neighbor_table_ωij, sign_ωij,
                λij, neighbor_table_λij)

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

##################################################################
## DEFINING METHODS TO INCREMENTALLY SPECIFY THE HOLSTEIN MODEL ##
##################################################################

# GENERATE THE FOLLOWING FUNCTIONS: assign_μ!, assign_ω!, assign_λ!
for param in [:μ,:ω,:λ]

    # constructing symbol for function name
    op = Symbol(:assign_,param,:!)

    # defining functions assuming parameter value μ0 is a real number
    @eval begin
        function $op(holstein::HolsteinModel,μ0::AbstractFloat,σ0::AbstractFloat,orbit::Int=0)
            
            if orbit==0 # assigning parameter values for all sites
                holstein.$param .= μ0 .+ σ0 .* randn(length(holstein.$param))
            else # assigning paramerter values for only sites of certain kind of orbital
                for i in 1:length(holstein.$param)
                    if holstein.lattice.site_to_orbit[i]==orbit
                        holstein.$param[i] = abs(μ0) + σ0 * randn()
                    end
                end
            end
            return nothing
        end
    end

    # defining function assuming parameter values μ0 is a complex number
    @eval begin
        function $op(holstein::HolsteinModel,μ0::Complex,σ0::AbstractFloat,orbit::Int=0)
            
            # getting phase of complex number
            phase = angle(μ0)

            # getting maginiute of complex number
            mag = abs(μ0)

            # assigning values to parameters based on magninute of μ0
            $op(holstein,mag,σ0,orbit)

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
    
    # defining functions assuming μ0 is a real number
    # disorder is only applied to |μ0|, not the phase angle(μ0).
    @eval begin
        function $op(holstein::HolsteinModel, μ0::AbstractFloat, σ0::AbstractFloat,
                     orbit1::Int, orbit2::Int, displacement::Vector{Int})
            
            # checking dimensions
            @assert length(displacement)==3

            # recasting 0 displacement in unit cell to displacement by L (width of lattice)
            # so that indexing in next step works
            for i in 1:3
                if displacement[i] == 0
                    displacement[i] = holstein.lattice.dims[i]
                end
            end 
            
            # getting new neighbors
            neighbors = holstein.qlattice.trans_equiv_sets[:,:,orbit2,orbit1,displacement[1],displacement[2],displacement[3]]
            
            # getting number of new neighbors
            nneighbors = size(neighbors,2)
            
            # adding new neighbors to neighbor table
            holstein.$neighbor_table = hcat(holstein.$neighbor_table,neighbors)
            
            # getting parameter value associated with each new neighbor
            for i in 1:nneighbors
                push!( holstein.$param , μ0 + σ0 * randn() )
            end
            
            return nothing
        end
    end
    
    # defining functions assuming μ0 is a complex number.
    # disorder is only applied to |μ0|, not the phase angle(μ0). 
    @eval begin
        function $op(holstein::HolsteinModel, μ0::Complex, σ0::AbstractFloat,
                     orbit1::Int, orbit2::Int, displacement::Vector{Int})
            
            # phase of μ0
            phase = angle(μ0)
            
            # amplitude of μ0
            mag = abs(μ0)
            
            # setting parameter values using the amplitude of μ0
            $op(holstein,mag,σ0,orbit1,orbit2,displacement)
            
            # number of new neighbors constructed
            nnewneighbors = div(holstein.lattice.nsites,holstein.lattice.norbits)
            
            # total number of neighbors
            nneighbors = size(holstein.$neighbor_table,2)
            
            # reapplying phase of complex number
            holstein.$neighbor_table[nneighbors-nnewneighbors:end] .*= exp(im*phase)
            
            return nothing
        end
    end
    
end

# adding functionality to assign_ωij! function so that the
# array holsteinmodel.sign_ωij is also modified
function assign_ωij!(holstein::HolsteinModel, μ0::AbstractFloat, σ0::AbstractFloat, sgn::Int,
                     orbit1::Int, orbit2::Int, displacement::Vector{Int})
    
    @assert abs(sgn)==1

    # updating neighbor table and parameter values
    assign_ωij!(holstein,μ0,σ0,orbit1,orbit2,displacement)
    
    # number of new neighbors constructed
    nnewneighbors = div(holstein.lattice.nsites,holstein.lattice.norbits)
    
    # modifying holsteinmodel.sign_ωij array
    append!( holsteinmodel.sign_ωij , zeros(sgn,nnewneighbors) )
    
    return nothing
end

end