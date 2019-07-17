module HolsteinModels

using Langevin.Geometries: Geometry
using Langevin.Lattices: Lattice, translationally_equivalent_sets, sort_neighbors!

export HolsteinModel
export assign_μ!, assign_ω!, assign_λ!
export assign_tij!, assign_ωij!, assign_λij!
export sort_neighbor_tables!

# allow for both Float64 and Complex{Float64} hopping parameters
mutable struct HolsteinModel{T}

    ################################
    ## COMPLETE MODEL HAMILTONIAN ##
    ################################

    # H =  ∑ Pᵢ²/2 + ∑ (ωᵢ²/2) ϕᵢ²  [Einstein Phonons]
    #   +  ∑ λᵢ ϕᵢ nᵢ              [El-Ph Coupling]
    #   +  ∑ ωᵢⱼ(ϕᵢ ± ϕⱼ)²         [Phonon Dispersion]
    #   +  ∑ λᵢⱼ ϕᵢ nⱼ             [Extended El-Ph Coupling]
    #   -  ∑ μᵢ nᵢ                 [Chemical Potential]

    ###########################
    ## SPECIFIES TEMPERATURE ##
    ###########################

    "inverse temperature"
    β::Float64

    "imaginary time step"
    Δτ::Float64

    "length of imaginary time axis"
    Lτ::Int

    #######################################
    ## FOR REPRESENTING LATTICE GEOMETRY ##
    #######################################

    "represents lattice geometry"
    geom::Geometry

    "represents lattice"
    lattice::Lattice

    "maps index_to_τ[index]=τ"
    index_to_τ::Vector{Int}

    "maps index_to_site[index]=site"
    index_to_site::Vector{Int}

    "map to_index[site,τ]=index"
    to_index::Matrix{Int}

    # this array will be very useful when making measurements that average over translational symmentry.
    "stores sets of translationally equivalent pairs of sites in lattice."
    trans_equiv_sets::Array{UInt16,7} # use datatype UInt16 to save memory.

    ############################
    ## HOLSTEIN PHONON FIELDS ##
    ############################

    "phonon fields ϕᵢ[site,τ]"
    ϕ::Matrix{Float64}

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

end

function HolsteinModel(geom::Geometry,lattice::Lattice,β::Float64,Δτ::Float64,T::DataType=Float64)::HolsteinModel

    # calculate length of imaginary time axis
    Lτ=Int(β/Δτ)

    # number of sites in physical lattice
    nsites = lattice.nsites

    # number of indices in D+1 dimensional lattice
    nindices = Lτ*nsites

    # constructing arrays for mapping between [site,τ]⇆[index]
    index_to_τ    = zeros(Int,nindices)
    index_to_site = zeros(Int,nindices)
    to_index      = zeros(Int,nsites,Lτ)
    τ = 0
    site = 0
    index = 1
    for τ in 1:Lτ
        for site in 1:nsites
            index_to_τ[index]    = τ
            index_to_site[index] = site
            to_index[site,τ]     = index
            index += 1
        end
    end

    # constructing translationally equivalent sets of sites
    trans_equiv_sets = Array{UInt16,7}(translationally_equivalent_sets(lattice))

    # intializing phonon fields to zero
    ϕ = zeros(Float64,nsites,Lτ)

    # initializing all on-site energies to zero
    μ = zeros(T,nsites)

    # initialize hopping parameters to empty vector
    tij = Vector{T}(undef,0)

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

    # initialize empty vector for λij
    λij = Vector{Float64}(undef,0)

    # initialize empty matrix vector for neighbor_table_λij
    neighbor_table_λij = Matrix{Int}(undef,2,0)

    # constructing holstein model
    holstein = HolsteinModel{T}(β, Δτ, Lτ,
                                geom, lattice, index_to_τ, index_to_site, to_index, trans_equiv_sets,
                                ϕ, 
                                μ, 
                                tij, neighbor_table_tij,
                                ω, λ,
                                ωij, neighbor_table_ωij, sign_ωij,
                                λij, neighbor_table_λij)

    return holstein
end


# GENERATE THE FOLLOWING FUNCTIONS: assign_μ!, assign_ω!, assign_λ!
for param in [:μ,:ω,:λ]

    # constructing symbol for function name
    op = Symbol(:assign_,param,:!)

    # defining functions assuming parameter value μ0 is a real number
    @eval begin
        function $op(holstein::HolsteinModel,μ0::Float64,σ0::Float64,orbit::Int=0)
            
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
        function $op(holstein::HolsteinModel,μ0::Complex{Float64},σ0::Float64,orbit::Int=0)
            
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
    @eval begin
        function $op(holstein::HolsteinModel, μ0::Float64, σ0::Float64,
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
            neighbors = holstein.trans_equiv_sets[:,:,orbit2,orbit1,displacement[1],displacement[2],displacement[3]]
            
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
    
    # defining functions assuming μ0 is a complex number
    @eval begin
        function $op(holstein::HolsteinModel, μ0::Complex{Float64}, σ0::Float64,
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
function assign_ωij!(holstein::HolsteinModel, μ0::Number, σ0::Float64, sgn::Int,
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


"""
    sort_neighbors!(holstein::HolsteinModel)

Function for sorting all the neighbor tables in HolsteinModel object
"""
function sort_neighbor_tables!(holstein::HolsteinModel)

    # sort tight binding hopping neighbors
    if size(holstein.neighbor_table_tij,2)>2
        perm = sort_neighbors!(holstein.neighbor_table_tij)
        holstein.tij .= holstein.tij[perm]
    end

    # sort phonon dispersion neighors
    if size(holstein.neighbor_table_ωij,2)>2
        perm = sort_neighbors!(holstein.neighbor_table_ωij)
        holstein.ωij .= holstein.ωij[perm]
        holstein.sign_ωij .= holstein.sign_ωij[perm]
    end

    # sort extended El-Ph coupling neighbors
    if size(holstein.neighbor_table_λij,2)>2
        perm = sort_neighbors!(holstein.neighbor_table_λij)
        holstein.λij .= holstein.λij[perm]
    end

    return nothing
end

end