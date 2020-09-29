module Measurements

using Printf
using FFTW
using LinearAlgebra
using Parameters

using ..Utilities: get_index, get_site, get_τ, θ, δ, translational_shift!, reshaped
using ..Models: HolsteinModel, SSHModel, AbstractModel
using ..SimulationParams: SimulationParameters
using ..GreensFunctions: EstimateGreensFunction, update!, measure_GΔ0, measure_GΔ0_GΔ0, measure_GΔΔ_G00, measure_GΔ0_G0Δ

export initialize_measurements_container
export initialize_measurement_files!
export make_measurements!
export process_measurements!

"""
Describes a displacement vector in the lattice.
"""
struct DisplacementVector

    "Starting orbital"
    o₁::Int

    "Ending orbital"
    o₂::Int

    "Displacement vector in terms of unit cells."
    v::Vector{Int}

    function DisplacementVector(o₁::Int,o₂::Int,v::AbstractVector{Int})

        @assert length(v)==3
        @assert o₁>0
        @assert o₂>0
        v′ = zeros(Int,3)
        copyto!(v′,v)
        return new(o₁,o₂,v′)
    end
end

"""
Construct a container to hold the measurements. The input `info` is a dictionary containing the information
from the `measurements` table in the config file.
"""
function initialize_measurements_container(holstein::HolsteinModel{T1,T2,T3},info::Dict) where {T1,T2,T3}

    Lₜ = holstein.Lτ
    L₁ = holstein.lattice.L1
    L₂ = holstein.lattice.L2
    L₂ = holstein.lattice.L3
    nₒ = holstein.lattice.norbits

    # number of random vectors used to make measurements
    if haskey(info,"num_random_vectors")
        num_random_vectors = info["num_random_vectors"]
    else
        num_random_vectors = 1
    end

    container = (global_meas    = Dict(),
                 onsite_meas    = Dict(),
                 onsite_corr    = Dict(),
                 intersite_meas = Dict(),
                 intersite_corr = Dict()
                 n_rand_vecs    = num_random_vectors)

    
    #########################
    ## GLOBAL MEASUREMENTS ##
    #########################
    
    container.global_meas["compressibility"] = 0.0
    container.global_meas["density"]         = 0.0

    ##########################    
    ## ON-SITE MEASUREMENTS ##
    ##########################

    container.onsite_meas["density"]     = zeros(T2,nₒ)
    container.onsite_meas["double_occ"]  = zeros(Complex{T1},nₒ)
    container.onsite_meas["x2"]          = zeros(T2,nₒ)
    container.onsite_meas["x4"]          = zeros(T2,nₒ)
    container.onsite_meas["phonon_pot"]  = zeros(T2,nₒ)
    container.onsite_meas["phonon_kin"]  = zeros(T2,nₒ)
    container.onsite_meas["elph_energy"] = zeros(T2,nₒ)

    # determining whether s-wave susceptibility is measured
    if haskey(info,"PairGreens")
        if info["PairGreens"]["measure"]==true
            if info["PairGreens"]["time_dependent"]==true
                container.onsite_meas["swave_susc"] = zeros(Complex{T1},nₒ)
            end
        end
    end

    #############################
    ## INTER-SITE MEASUREMENTS ##
    #############################

    # NONE FOR THE HOLSTEIN MODEL CURRENTLY

    ###################################
    ## ON-SITE CORRELATION FUNCTIONS ##
    ###################################

    # electron green function
    init_onsite_corr_container!(container.onsite_corr,"Greens",info,holstein,nₒ,L₁,L₂,L₃,Lₜ)

    # density-density correlation
    init_onsite_corr_container!(container.onsite_corr,"DenDen",info,holstein,nₒ,L₁,L₂,L₃,Lₜ)

    # pair green function
    init_onsite_corr_container!(container.onsite_corr,"PairGreens",info,holstein,nₒ,L₁,L₂,L₃,Lₜ)

    # phonon green function
    init_onsite_corr_container!(container.onsite_corr,"PhononGreens",info,holstein,nₒ,L₁,L₂,L₃,Lₜ)

    ######################################
    ## INTER-SITE CORRELATION FUNCTIONS ##
    ######################################

    # bond-bond correlation function
    intersite_corr_container!(container.intersite_corr,"BondBond",info,holstein,L₁,L₂,L₃,Lₜ)

    return container
end

function initialize_measurements_container(ssh::SSHModel{T1,T2,T3},info::Dict) where {T1,T2,T3}

    Lₜ = ssh.Lτ
    L₁ = ssh.lattice.L1
    L₂ = ssh.lattice.L2
    L₂ = ssh.lattice.L3
    nₒ = ssh.lattice.norbits

    # number of phonon defintions/types of phonons
    nᵥ = length(ssh.phonon_definitions)

    container = (global_meas    = Dict(),
                 onsite_meas    = Dict(),
                 onsite_corr    = Dict(),
                 intersite_meas = Dict(),
                 intersite_corr = Dict())

    
    #########################
    ## GLOBAL MEASUREMENTS ##
    #########################
    
    container.global_meas["compressibility"] = 0.0
    container.global_meas["density"]         = 0.0

    ##########################    
    ## ON-SITE MEASUREMENTS ##
    ##########################

    container.onsite_meas["density"]    = zeros(Complex{T1},nₒ)
    container.onsite_meas["double_occ"] = zeros(Complex{T1},nₒ)

    # determining whether s-wave susceptibility is measured
    if haskey(info,"PairGreens")
        if info["PairGreens"]["measure"]==true
            if info["PairGreens"]["time_dependent"]==true
                container.onsite_meas["swave_susc"] = zeros(Complex{T1},nₒ)
            end
        end
    end

    #############################
    ## INTER-SITE MEASUREMENTS ##
    #############################

    # construct displacement vectors associated with each type of phonon
    phonon_definitions = ssh.phonon_definitions
    phonon_vectors     = Vector{DisplacementVector}(undef,0)
    for i in 1:nᵥ
        phonon_definition = phonon_definitions[i]
        phonon_vector     = DisplacementVector(phonon_definition.o₁,phonon_definition.o₂,phonon_definition.v)
        push!(phonon_vectors,phonon_vector)
    end
    container.intersite_meas["vectors"] = phonon_vectors

    # intialize containers for measurements
    container.intersite_meas["x2"]          = zeros(T2,nₒ)
    container.intersite_meas["x4"]          = zeros(T2,nₒ)
    container.intersite_meas["phonon_pot"]  = zeros(T2,nₒ)
    container.intersite_meas["phonon_kin"]  = zeros(T2,nₒ)
    container.intersite_meas["elph_energy"] = zeros(T2,nₒ)

    ###################################
    ## ON-SITE CORRELATION FUNCTIONS ##
    ###################################

    # electron green function
    init_onsite_corr_container!(container.onsite_corr,"Greens",info,ssh,nₒ,L₁,L₂,L₃,Lₜ)

    # density-density correlation
    init_onsite_corr_container!(container.onsite_corr,"DenDen",info,ssh,nₒ,L₁,L₂,L₃,Lₜ)

    # pair green function
    init_onsite_corr_container!(container.onsite_corr,"PairGreens",info,ssh,nₒ,L₁,L₂,L₃,Lₜ)

    ######################################
    ## INTER-SITE CORRELATION FUNCTIONS ##
    ######################################

    # phonon green function
    init_intersite_corr_container!(container.intersite_corr,"PhononGreens",info,ssh,phonon_vectors,nₒ,L₁,L₂,L₃,Lₜ)

    # bond-bond correlation function
    init_intersite_corr_container!(container.intersite_corr,"BondBond",info,ssh,L₁,L₂,L₃,Lₜ)

    return container
end


"""
Initialize Measurement Files.
"""
function initialize_measurement_files!(container::NamedTuple,sim_params::SimulationParameters)

    #############################################
    ## Initialize File For Global Measurements ##
    #############################################

    open(sim_params.datafolder*"global_measurements.out", "w") do file
        for key in keys(container.global_meas)
            measurement = String(key)
            write(file, ",", measurement)
        end
        write(file, "\n")
    end

    ##############################################
    ## Initialize File For On-Site Measurements ##
    ##############################################

    open(sim_params.datafolder*"onsite_measurements.out", "w") do file
        write(file, "orbit")
        for key in keys(container.onsite_meas)
            measurement = String(key)
            write(file, ",", measurement)
        end
        write(file, "\n")
    end

    #################################################
    ## Initialize File For Inter-Site Measurements ##
    #################################################

    open(sim_params.datafolder*"intersite_measurements.out", "w") do file
        write(file, "vector")
        for key in keys(container.intersite_meas)
            measurement = String(key)
            write(file, ",", measurement)
        end
        write(file, "\n")
    end

    ###############################################
    ## Initialize Files For On-Site Correlations ##
    ###############################################

    # iterate over on-site correlation functions
    for measurement in keys(container.onsite_corr)
        # initialize file for position-space data
        open(sim_params.datafolder * measurement * "_position.out", "w") do file
            # writing file header
            write(file, "orbit1", ",", "orbit2", ",", "r1",  ",", "r2",  ",", "r3", ",", "tau", ",", measurement, "\n")
        end
        # initialize file for position-space data
        open(sim_params.datafolder * measurement * "_momentum.out", "w") do file
            # writing file header
            write(file, "orbit1", ",", "orbit2", ",", "k1",  ",", "k2",  ",", "k3", ",", "tau", ",", measurement, "\n")
        end
    end

    ##################################################
    ## Initialize Files For Inter-Site Correlations ##
    ##################################################

    # iterate over on-site correlation functions
    for measurement in keys(container.onsite_corr)
        # initialize file for position-space data
        open(sim_params.datafolder * measurement * "_position.out", "w") do file
            # writing file header
            write(file, "vector1", ",", "vector2", ",", "r1",  ",", "r2",  ",", "r3", ",", "tau", ",", measurement, "\n")
        end
        # initialize file for position-space data
        open(sim_params.datafolder * measurement * "_momentum.out", "w") do file
            # writing file header
            write(file, "vector1", ",", "vector2", ",", "k1",  ",", "k2",  ",", "k3", ",", "tau", ",", measurement, "\n")
        end
    end

    return nothing
end


"""
Make measurements.
"""
function make_measurements!(container::NamedTuple,model::AbstractModel,Gr::EstimateGreensFunction,preconditioner)

    for i in 1:container.n_rand_vecs
        update!(Gr,model,preconditioner)
        make_global_measurements!(container,model,Gr)
        measure_onsite_correlations!(container,model,Gr)
        measure_intersite_correlations!(container,model,Gr)
        make_onsite_measurements!(container,model,Gr)
        make_intersite_measurements!(container,model,Gr)
    end
    return nothing
end

"""
Process Measurments. This includes:
1. fourier transforming measurements from position to momentum space.
2. normalizing measurements by bin size.
3. calcating integrated quantitites like the pair-susceptibility.
"""
function process_measurements!(container::NamedTuple,sim_params::SimulationParameters,model::AbstractModel)

    #########################################
    ## FOURIER TRANSFORM TO MOMENTUM SPACE ##
    #########################################

    # fourier transform on-site correlation functions
    fourier_transform_correlations!(container.onsite_corr)
    
    # fourier transform inter-site correlation functions
    fourier_transform_correlations!(container.onsite_corr)

    ############################
    ## NORMALIZE MEASUREMENTS ##
    ############################

    # bin size
    bin_size = sim_params.bin_size

    # number of random vectors used to make measurements
    n_rand_vecs = sim_params.bin_size

    # normalization constant
    V = bin_size * n_rand_vecs


    return nothing
end

#####################
## PRIVATE METHODS ##
#####################

"""
Intialize multi-dimensional array to contain an inter-site correlation measurement.
"""
function init_intersite_corr_container!(container::Dict,measurement::Strng,info::Dict,model::AbstractModel{T1,T2,T3},
                                   displacement_vectors::Vector{DisplacementVector},L₁::Int,L₂::Int,L₃::Int,Lₜ::Int) where {T1,T2,T3}
    
    if haskey(info,measurement)
        if info[measurement]["measure"]==true
            container[measurement] = Dict()
            # get displacement vectors associated with measurement
            container[measurement]["vectors"] = displacement_vectors
            # number of displacement vector definitions
            nᵥ = length(displacement_vectors)
            # declare multi-dimnesional arrays to hold measurement
            if info[measurement]["time_dependent"]==true
                container[measurement]["position"] = zeros(Complex{T1},Lₜ+1,L₁,L₂,L₃,nᵥ,nᵥ)
                container[measurement]["momentum"] = zeros(Complex{T1},Lₜ+1,L₁,L₂,L₃,nᵥ,nᵥ)
            else
                container[measurement]["position"] = zeros(Complex{T1},1,L₁,L₂,L₃,nᵥ,nᵥ)
                container[measurement]["momentum"] = zeros(Complex{T1},1,L₁,L₂,L₃,nᵥ,nᵥ)
            end
        end
    end
    return nothing
end

function init_intersite_corr_container!(container::Dict,measurement::Strng,info::Dict,model::AbstractModel{T1,T2,T3},
                                   L₁::Int,L₂::Int,L₃::Int,Lₜ::Int) where {T1,T2,T3}
    
    if haskey(info,measurement)
        if info[measurement]["measure"]==true
            # get relevant displacement vectors
            displacement_vectors = Vector{DisplacementVector}(undef,0)
            for vector in info[measurement["vector"]
                displacement_vector = DisplacementVector(vector["orbit"][1],vector["orbit"][2],vector["dL"])
                push!(displacement_vectors,displacement_vector)
            end
            # initialize container
            intersite_corr_container!(container,measurement,info,model,displacement_vectors,L₁,L₂,L₃,Lₜ,time_dependent)
        end
    end
    return nothing
end

"""
Intialize multi-dimensional array to contain an inter-site correlation measurement.
"""
function init_onsite_corr_container!(container::Dict,measurement::Strng,info::Dict,model::AbstractModel{T1,T2,T3},
                                nₒ::Int,L₁::Int,L₂::Int,L₃::Int,Lₜ::Int) where {T1,T2,T3}
    
    if haskey(info,measurement)
        if info[measurement]["measure"]==true
            # declare multi-dimnesional arrays to hold measurement
            if info[measurement]["time_dependent"]==true
                container[measurement]["position"] = zeros(Complex{T1},Lₜ+1,L₁,L₂,L₃,nₒ,nₒ)
                container[measurement]["momentum"] = zeros(Complex{T1},Lₜ+1,L₁,L₂,L₃,nₒ,nₒ)
            else
                container[measurement]["position"] = zeros(Complex{T1},1,L₁,L₂,L₃,nₒ,nₒ)
                container[measurement]["momentum"] = zeros(Complex{T1},1,L₁,L₂,L₃,nₒ,nₒ)
            end
        end
    end
    return nothing
end

"""
Make global measurements.
"""
function make_global_measurements!(container::NamedTuple,model::AbstractModel,Gr::EstimateGreensFunction)

    @unpack r₁, M⁻¹r₁, r₂, M⁻¹r₂, NL, N, L = Gr
    @unpack β = model

    # measure ⟨N⟩
    nup = (NL - dot(M⁻¹r₁,r₁))/L
    ndn = (NL - dot(M⁻¹r₂,r₂))/L
    n   = nup + ndn
    container.global_meas["density"] += n/N

    # measure ⟨N²⟩
    n² = 4/L^2 * (NL-dot(r₁,M⁻¹r₁)) * (NL-dot(r₂,M⁻¹r₂))

    # measure compressibility κ=(β/N)⋅(⟨N²⟩-⟨N⟩²)
    κ = β*(n²-n^2)/N
    container.global_meas["compressibility"] += κ

    return nothing
end

"""
Measure on-site correlation functions
"""
function measure_onsite_correlations!(container::NamedTuple,model::HolsteinModel,Gr::EstimateGreensFunction)

    onsite_corr = container.onsite_corr

    for measurement in keys(onsite_corr)
        if measurement=="Greens"
            measure_Greens!(onsite_corr["Greens"],model,Gr)
        elseif measurement=="DenDen"
            measure_DenDen!(onsite_corr["DenDen"],model,Gr)
        elseif measurement=="PairGreens"
            measure_PairGreens!(onsite_corr["PairGreens"],model,Gr)
        elseif measurement=="PhononGreens"
            measure_PhononGreens!(onsite_corr["PhononGreens"],model,Gr)
        end
    end

    return nothing
end

function measure_onsite_correlations!(container::NamedTuple,model::SSHModel,Gr::EstimateGreensFunction)

    onsite_corr = container.onsite_corr

    for measurement in keys(onsite_corr)
        if measurement=="Greens"
            measure_Greens!(onsite_corr["Greens"],model,Gr)
        elseif measurement=="DenDen"
            measure_DenDen!(onsite_corr["DenDen"],model,Gr)
        elseif measurement=="PairGreens"
            measure_PairGreens!(onsite_corr["PairGreens"],model,Gr)
        end
    end

    return nothing
end

"""
Measure inter-site correlation functions
"""
function measure_intersite_correlations!(container::NamedTuple,model::HolsteinModel,Gr::EstimateGreensFunction)

    intersite_corr = container.intersite_corr

    for measurement in keys(intersite_corr)
        if measurement=="BondBond"
            measure_BondBond!(onsite_corr["BondBond"],model,Gr)
        end
    end

    return nothing
end

function measure_intersite_correlations!(container::NamedTuple,model::SSHModel,Gr::EstimateGreensFunction)

    onsite_corr = container.onsite_corr

    for measurement in keys(intersite_corr)
        if measurement=="BondBond"
            measure_BondBond!(onsite_corr["BondBond"],model,Gr)
        elseif measurement=="PhononGreens"
            measure_PhononGreens!(onsite_corr["PhononGreens"],model,Gr)
        end
    end

    return nothing
end

"""
Make on-site measurements.
"""
function make_onsite_measurements!(container::NamedTuple,model::HolsteinModel,Gr::EstimateGreensFunction)

    # measurements container
    onsite_meas = container.onsite_meas

    # phonon fields
    x = model.x

    # getting number of orbitals
    norbits = model.lattice.unit_cell.norbits::Int

    # number of physical sites in lattice
    nsites = model.Nsites::Int

    # length of imaginary time axis
    Lτ = model.Lτ::Int

    # for measuring phonon kinetic energy
    Δτ  = model.Δτ
    Δτ² = Δτ * Δτ

    # normalization
    normalization = div(nsites,norbits)*Lτ

    # iterating over orbital types
    for orbit in 1:norbits
        # measure density
        onsite_meas["density"][orbit] += 2.0 * (1.0-real(measure_Greens(Gr,0,0,0,orbit,orbit,0)))
        # measure double occupancy
        onsite_meas["double_occ"][orbit] += real(measure_DenDen(Gr,0,0,0,orbit,orbit,0))
        # iterating over orbits of the current type
        for site in orbit:norbits:nsites
            # iterating over time slices
            for τ in 1:Lτ
                # getting current index
                index = get_index(τ,site,Lτ)
                # estimate ⟨cᵢ(τ)c⁺ᵢ(τ)⟩
                G1 = estimate(Gr,site,site,τ,τ,1)
                G2 = estimate(Gr,site,site,τ,τ,2)
                # measuring phonon kinetic energy such that
                # ⟨KE⟩ = 1/(2Δτ) - ⟨(1/2)[xᵢ(τ+1)-xᵢ(τ)]²/Δτ²⟩
                Δx = x[get_index(τ%Lτ+1,site,Lτ)]-x[index]
                onsite_meas["phonon_kin"][orbit] += (0.5/Δτ-(Δx^2)/Δτ²/2) / normalization
                # measuring phonon potential energy
                onsite_meas["phonon_pot"][orbit] += (model.ω[site]^2*x[index]^2/2 + model.ω4[site]*x[index]^4) / normalization
                # measuring the electron phonon energy λ⟨x⋅(n₊+n₋)⟩
                onsite_meas["elph_energy"][orbit] += model.λ[site]*x[index]*(2.0-G1-G2) / normalization
                # measure ⟨x⟩
                onsite_meas.["x"][orbit] += x[index] / normalization
                # measure ⟨x²⟩
                onsite_meas["x2"][orbit] += x[index]^2 / normalization
                # measure ⟨x⁴⟩
                onsite_meas["x4"][orbit] += x[index]^4 / normalization
            end
        end
    end

    return nothing
end

function make_onsite_measurements!(container::NamedTuple,model::SSHModel,Gr::EstimateGreensFunction)

    # measurements container
    onsite_meas = container.onsite_meas

    # phonon fields
    x = model.x

    # getting number of orbitals
    norbits = model.lattice.unit_cell.norbits::Int

    # number of physical sites in lattice
    nsites = model.Nsites::Int

    # length of imaginary time axis
    Lτ = model.Lτ::Int

    # for measuring phonon kinetic energy
    Δτ  = model.Δτ
    Δτ² = Δτ * Δτ

    # normalization
    normalization = div(nsites,norbits)*Lτ

    # iterating over orbital types
    for orbit in 1:norbits
        # measure density
        onsite_meas["density"][orbit] += 2.0 * (1.0-real(measure_Greens(Gr,0,0,0,orbit,orbit,0)))
        # measure double occupancy
        onsite_meas["double_occ"][orbit] += real(measure_DenDen(Gr,0,0,0,orbit,orbit,0))
            end
        end
    end

    return nothing
end

"""
Make inter-site measurements.
"""
function make_intersite_measurements!(container::NamedTuple,model::HolsteinModel,Gr::EstimateGreensFunction)

    # null function as no inter-site measurements to make for Holstein model currently

    return nothing
end

function make_intersite_measurements!(container::NamedTuple,model::SSHModel{T1,T2,T3},Gr::EstimateGreensFunction{T1}) where {T1,T2,T3}

    # container for measurements
    intersite_meas = container.intersite_meas

    # number of phonon fields
    Ndof = model.Ndof

    # length of imaginary time axis
    Lτ = model.Lτ

    # number of types of phonons
    n = length(model.phonon_definitions)

    # number of each type of phonon
    N = div(Ndof,n*Lτ)

    # phonon fields
    x = reshaped(model.x,(Lτ,N,n))

    # imaginary time step
    Δτ = model.Δτ

    # normalization
    V = (N*Lτ)

    # iterate over types phonon
    for d in 1:n
        # phonon/bond parameters
        @unpack v, o₁, o₂, t, α, ω = model.phonon_definitions[d]
        # iterate of phonons
        for phonon in 1:N
            # get bond associated with phonon
            bond = model.phonon_to_bond[phonon]
            # get pair of sites associated with bond/phonon
            s₁ = model.neighbor_table[1,bond]
            s₂ = model.neighbor_table[2,bond]
            # iterate over time slices for each phonon
            for τ in 1:Lτ
                # get xᵢ[τ] phonon field
                xτ   = x[τ,phonon,d]
                # get xᵢ[τ+1] phonon field
                xτp1 = x[mod1(τ+1,Lτ),phonon,d]
                # Δx = xᵢ[τ+1]-xᵢ[τ]
                Δx   = xτp1-xτ
                # get hopping amplitude h = ∑ₛ⟨c⁺ₛᵢcₛⱼ+h.c.⟩
                G↑  = estimate(Gr,s₁,s₂,τ,τ,1)
                G†↑ = estimate(Gr,s₂,s₁,τ,τ,1)
                G↓  = estimate(Gr,s₂,s₁,τ,τ,2)
                G†↓ = estimate(Gr,s₂,s₁,τ,τ,2)
                h   = (1.0-G↑) + (1.0-G†↑) + (1.0-G↓) + (1.0-G†↓)
                # phonon potential energy
                intersite_meas["phonon_pot"]  += (ω^2*xτ^2/2)/V
                # phonon kinetic energy
                intersite_meas["phonon_kin"]  += (0.5/Δτ-(Δx/Δτ)^2/2.0)/V
                # electron-phonon energy
                intersite_meas["elph_energy"] += (α*h*xτ)/V
                # ⟨x²⟩
                intersite_meas["x2"]          += (xτ^2)/V
                # ⟨x⁴⟩
                intersite_meas["x2"]          += (xτ^4)/V
            end
        end
    end

    return nothing
end

"""
Fourier transform position space correlation functions to momentum space.
"""
function fourier_transform_correlations!(container::Dict)

    for measurement in keys(container)
        position = container[measurement]["position"]
        momentum = container[measurement]["momentum"]
        copyto!(momentum,position)
        fft!(momentum,(2,3,4))
    end

    return nothing
end

############################################################
## IMPLEMENTING ON-SITE CORRELATION FUNCTION MEASUREMENTS ##
############################################################

"""
Measure time-ordered single-particle electron Green's function ⟨cᵢ₊ᵣ(τ)⋅c⁺ᵢ(0)⟩ for 0≤τ<β
"""
function measure_Greens(model::AbstractModel,estimator::EstimateGreensFunction,
                        l₁::Int,l₂::Int,l₃::Int,o₁::Int,o₂::Int,τ::Int)

    L   = model.Lτ
    Gᵣτ = measure_GΔ0(estimator,l₁,l₂,l₃,o₁,o₂,τ%L)

    # Gᵣ(β) = δᵣ - Gᵣ(0)
    if τ==L
        Gᵣτ = δ(l₁)*δ(l₂)*δ(l₃)*δ(o₁,o₂) - Gᵣτ
    end

    return Gᵣτ
end

"""
Measure the density-density correlation function ⟨nᵢ₊ᵣ(τ)⋅nᵢ(0)⟩ for 0≤τ<β
"""
function measure_DenDen(model::AbstractModel,estimator::EstimateGreensFunction,
                        l₁::Int,l₂::Int,l₃::Int,o₁::Int,o₂::Int,τ::Int)

    L           = model.Lτ
    Gᵣ₀τ0       = measure_GΔ0(estimator,l₁,l₂,l₃,o₁,o₂,τ%L)    
    G₀₀00       = measure_GΔ0(estimator,0,0,0,o₁,o₁,0)
    Gᵣᵣττ       = measure_GΔ0(estimator,0,0,0,o₂,o₂,0)
    Gᵣᵣττ_G₀₀00 = measure_GΔΔ_G00(estimator,l₁,l₂,l₃,o₁,o₂,τ%L)
    Gᵣ₀τ0_G₀ᵣ0τ = measure_GΔ0_G0Δ(estimator,l₁,l₂,l₃,o₁,o₂,τ%L)
    δᵣ          = δ(l₁)*δ(l₂)*δ(l₃)*δ(o₁,o₂)
    nᵣτ_n₀0     = 4.0 * ( 1.0 - Gᵣᵣττ - G₀₀00 + Gᵣᵣττ_G₀₀00 + 0.5 * ( δᵣ*δ(τ)*Gᵣ₀τ0 - Gᵣ₀τ0_G₀ᵣ0τ ) )

    return nᵣτ_n₀0
end

"""
Measure pair Green's function ⟨Δᵢ₊ᵣ(τ)⋅Δ⁺ᵢ(0)⟩ where Δᵢ(τ) = cᵢ₊(τ)cᵢ₋(τ) for 0≤τ<β
"""
function measure_PairGreens(model::AbstractModel,estimator::EstimateGreensFunction,
                            l₁::Int,l₂::Int,l₃::Int,o₁::Int,o₂::Int,τ::Int)

    L = model.Lτ

    # Pᵣ(τ) = ⟨Δᵢ₊ᵣ(τ)⋅Δ⁺ᵢ(0)⟩ = ⟨c↑ᵢ₊ᵣ(τ)⋅c⁺↑ᵢ(0)⟩⋅⟨c↓ᵢ₊ᵣ(τ)⋅c⁺↓ᵢ(0)⟩
    Pᵣτ = measure_GΔ0_GΔ0(estimator,l₁,l₂,l₃,o₁,o₂,τ%L)

    # Pᵣ(β) = Pᵣ(0) - 2δᵣ[Gᵣ↑(0)+Gᵣ↓(0)-P₀(0)-1/2]
    if τ==L
        δᵣ  = δ(l₁)*δ(l₂)*δ(l₃)*δ(o₁,o₂)
        if δᵣ==1
            G₀0 = measure_Greens(model,estimator,0,0,0,o₁,o₂,0)
            Pᵣτ = 3*Pᵣτ - 4*G₀0 + 1.0
        end
    end

    return Pᵣτ
end

# defining functions measure_Greens!, measure_DenDen!, measure_PairGreens!
for measurement in [ :Greens , :DenDen , :PairGreens ]

    # constructing symbol for function name
    op = Symbol(:measure_,measurement,:!)

    # function to make measurement
    measure = Symbol(:measure_,measurement)

    @eval begin
        function $op(container::Array{Complex{T1},6}, model::AbstractModel{T1,T2,T3},
                     Gr::EstimateGreensFunction{T1}) where {T1,T2,T3}

            # getting size of system
            L  = size(container,1)-1
            L₁ = size(container,2)
            L₂ = size(container,3)
            L₃ = size(container,4)
            n₀ = size(container,5)

            # iterate over all relevant space-time displacement vectors
            for l₃ in 0:L₃-1
                for l₂ in 0:L₂-1
                    for l₁ in 0:L₁-1
                        for o₂ in 1:n₀
                            for o₁ in 1:n₀
                                for τ in 0:L
                                    container[ τ+1, l₁+1, l₂+1, l₃+1, o₁, o₂] += $measure(model,Gr,l₁,l₂,l₃,o₁,o₂,τ)
                                end
                            end
                        end
                    end
                end
            end
            return nothing
        end
    end
end

"""
Measure Phonon Green's function for Holstein model.
"""
function measure_PhononGreens!(container::Array{Complex{T1},6}, model::HolsteinModel{T1,T2,T3},
                               Gr::EstimateGreensFunction{T1}) where {T1,T2,T3}

    @unpack a, b, ab′ = Gr

    # size of lattice
    Lₜ = model.Lτ::Int
    L₁ = model.lattice.L1::Int
    L₂ = model.lattice.L2::Int
    L₂ = model.lattice.L3::Int
    nₒ = model.lattice.norbits::Int
    nᵤ = model.lattice.ncells::Int

    # length of axis corresponding to imaginary time.
    L₀ = size(container,1)

    # phonon fields
    x = reshaped(model.x,(Lₜ,nₒ,L₁,L₂,L₃))

    # containers for performing calculation
    x₁x₂ = reshaped(view(ab′,1:nᵤ),(Lₜ,L₁,L₂,L₃))
    x₁   = reshaped(view(a,1:nᵤ),  (Lₜ,L₁,L₂,L₃))
    x₂   = reshaped(view(b,1:nᵤ),  (Lₜ,L₁,L₂,L₃))

    for o₂ in 1:nₒ
        # phonon fields for orbital o₂
        @views @. x₂ = x[:,o₂,:,:,:]
        for o₁ in 1:n₁
            # phonon fields for orbital o₁
            @views @. x₁ = x[:,o₁,:,:,:]
            # translationally average
            translational_average!(x₁x₂,x₁,x₂)
            # record measurement
            @views @. container[1:L,:,:,:,o₁,o₂] += x₁x₂[1:L,:,:,:]
            # save the results
            if L₀==1 # equal time measurment
                @views @. measurements[1,:,:,:,o₁,o₂] += x₁x₂[1,:,:,:]
            else # unequal time measurement
                @views @. measurements[1:Lₜ,:,:,:,o₁,o₂] += x₁x₂
                # dealing with τ=β time slice
                @views @. measurements[Lₜ+1,:,:,:,o₁,o₂] += x₁x₂[1,:,:,:]
            end
        end
    end

    return nothing
end

###############################################################
## IMPLEMENTING INTER-SITE CORRELATION FUNCTION MEASUREMENTS ##
###############################################################

"""
Measure Bond-Bond correlation function between all possible pairings of types of bonds.
The bond/hopping operator is K[a,b,r′](τ,r)=[a⁺(↑,i+r+r′,τ)⋅b(↑,i+r,τ)+a⁺(↓,i+r+r′,τ)⋅b(↓,i+r,τ)].
Therefore, the bond-bond correlation is given by
B[a,b,r′;c,d,r″](τ,r) = ⟨K[a,b,r′](τ,r)⋅K[c,d,r″](0,0)⟩
B[a,b,r′;c,d,r″](τ,r) = ⟨[a⁺(↑,i+r+r′,τ)⋅b(↑,i+r,τ)+a⁺(↓,i+r+r′,τ)⋅b(↓,i+r,τ)]⋅[c⁺(↑,i+r″,0)⋅d(↑,i,0)+c⁺(↓,i+r″,0)⋅d(↓,i,0)]⟩
"""
function measure_BondBond!(container::Dict,model::AbstractModel{T1,T2,T3},estimator::EstimateGreensFunction{T1}) where {T1,T2,T3}

    # size of lattice
    Lₜ = model.Lτ::Int
    L₁ = model.lattice.L1::Int
    L₂ = model.lattice.L2::Int
    L₂ = model.lattice.L3::Int
    nₒ = model.lattice.norbits::Int
    nᵤ = model.lattice.ncells::Int

    r₁    = reshaped(estimator.r₁,    (Lₜ,nₒ,L₁,L₂,L₃))
    M⁻¹r₁ = reshaped(estimator.M⁻¹r₁, (Lₜ,nₒ,L₁,L₂,L₃))
    r₂    = reshaped(estimator.r₂,    (Lₜ,nₒ,L₁,L₂,L₃))
    M⁻¹r₂ = reshaped(estimator.M⁻¹r₂, (Lₜ,nₒ,L₁,L₂,L₃))

    # containers for performing calculation
    bondbond = reshaped(view(estimator.ab″,1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
    G₁G₂     = reshaped(view(estimator.ab′,1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
    G₁       = reshaped(view(estimator.a,  1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
    G₂       = reshaped(view(estimator.b,  1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
    R₁′      = G₁
    R₁″      = G₁
    R₂′      = G₂
    R₂″      = G₂

    # displacement vectors describing each type of bond
    vectors = container["vectors"]

    # container to hold bond-bond correlation measurements
    measurement = container["position"]

    # length of axis corresponding to imaginary time axis
    L₀ = size(measurement,1)

    # number of vectors
    nᵥ = length(vectors)

    # initialize bond-bond correlation to zero
    fill!(bondbond,0.0)

    # iterate over first bond
    for n″ in 1:nᵥ

        # vector associated with bond going from orbitals d ⟶ c displaced r″ unit cells
        r″ = vectors[n″].v  # displacement in unit cells
        d  = vectors[n″].o₁ # starting orbital
        c  = vectors[n″].o₂ # ending   orbital

        # iterate over second bond
        for n′ in 1:nᵥ

            # vector associated with bond going from orbitals b ⟶ a displaced r′ unit cells
            r′ = vectors[n′].v  # displacement in unit cells
            b  = vectors[n′].o₁ # starting orbital
            a  = vectors[n′].o₂ # ending   orbital

            # CALCULATE G₁⋅G₂ = ⟨T⋅b(i+r,τ)⋅a⁺(i+r+r′,τ)⟩⋅⟨T⋅d(i,τ)⋅c⁺(i+r″,τ)⟩
            M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
            R₁    = @view    r₁[:,a,:,:,:]
            M⁻¹R₂ = @view M⁻¹r₂[:,d,:,:,:]
            R₂    = @view    r₂[:,c,:,:,:]
            circshift!(R₁′,R₁,(0,r′[1],r′[2],r′[2])) # R₁′   = shift(R₁,r′)
            circshift!(R₂″,R₂,(0,r″[1],r″[2],r″[2])) # R₂″   = shift(R₂,r″)
            @. G₁ = M⁻¹R₁ * R₁′                      # G₁    = [M⁻¹R₁⋅R₁′]
            @. G₂ = M⁻¹R₂ * R₂″                      # G₂    = [M⁻¹R₂⋅R₂″]
            translational_average!(G₁G₂,G₁,G₂)       # G₁⋅G₂ = [M⁻¹R₁⋅R₁′]⋆[M⁻¹R₂⋅R₂″]

            # B = B + 4*⟨T⋅b(i+r,τ)⋅a⁺(i+r+r′,τ)⟩⋅⟨T⋅d(i,τ)⋅c⁺(i+r″,τ)⟩
            @. bondbond += 4*G₁G₂

            # CALCULATE G₁⋅G₂ = ⟨T⋅b(i+r,τ)⋅c⁺(i+r″,0)⟩⋅⟨T⋅d(i,0)⋅a⁺(i+r+r′,τ)⟩
            M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
            R₁    = @view    r₁[:,c,:,:,:]
            M⁻¹R₂ = @view M⁻¹r₂[:,d,:,:,:]
            R₂    = @view    r₂[:,a,:,:,:]
            circshift!(R₂′,R₂,(0,r′[1],r′[2],r′[2])) # R₂′   = shift(R₂,r′)
            circshift!(R₁″,R₁,(0,r″[1],r″[2],r″[2])) # R₁″   = shift(R₁,r″)
            @. G₁ = M⁻¹R₁ * R₂′                      # G₁    = [M⁻¹R₁⋅R₂′]
            @. G₂ = M⁻¹R₂ * R₁″                      # G₂    = [M⁻¹R₂⋅R₁″]
            translational_average!(G₁G₂,G₁,G₂)       # G₁⋅G₂ = [M⁻¹R₁⋅R₂′]⋆[M⁻¹R₂⋅R₁″]

            # B = B - 2*⟨T⋅b(i+r,τ)⋅c⁺(i+r″,0)⟩⋅⟨T⋅d(i,0)⋅a⁺(i+r+r′,τ)⟩
            @. bondbond -= 2*G₁G₂

            # CALCULATE G₁ = δ(τ)⋅δ(r′+r)⋅δ(a,d)⋅⟨b(i+r,τ)⋅c⁺(i+r″,0)⟩
            #           G₁ = δ(τ)⋅δ(r+r′)⋅δ(a,d)⋅⟨b(i+r-r″,τ)⋅c⁺(i,0)⟩
            if a==d
                # δ(r+r′)               ⟶ r = -r′
                # ⟨b(i+r-r″,τ)⋅c⁺(i,0)⟩ ⟶ r = r - r″
                # therefore             ⟶ r = -r′-r″
                l₁ = mod(-r′[1]-r″[1],L₁)
                l₂ = mod(-r′[2]-r″[2],L₂)
                l₃ = mod(-r′[3]-r″[3],L₃)
                G₁ = measure_GΔ0(0,c,b,l₁,l₂,l₃)

                # B = B + 2⋅δ(τ)⋅δ(r′+r)⋅δ(a,d)⋅⟨b(i+r,τ)⋅c⁺(i+r″,0)⟩
                bondbond[τ+1,l₁+1,l₂+1,l₃+1] += 2*G₁
            else

            # record measurements
            if L₀==1 # if time independent measurement
                @views @. measurement[1,:,:,:,n′,n″]    += bondbond[1,:,:,:]
            else # if time dependent measurement
                @views @. measurement[1:Lₜ,:,:,:,n′,n″] += bondbond
                # deal with τ=β time slice
                for l₃ in 0:L₃-1
                    for l₂ in 0:L₂-1
                        for l₁ in 0:L₁-1
                            nl₁ = mod(-l₁,L₁)
                            nl₂ = mod(-l₂,L₂)
                            nl₃ = mod(-l₃,L₃)
                            # B[a,b,r′;c,d,r″](β,r) = B[c,d,r″;a,b,r′](0,-r)
                            measure_BondBond![Lₜ+1,l₁+1,l₂+1,l₃+1,n″,n′] += bondbond[1,nl₁+1,nl₂+1,nl₃+1]
                        end
                    end
                end
            end
        end
    end

    return nothing
end

"""
Measure Phonon Green's function for SSH model.
"""
function measure_PhononGreens!(container::Dict,model::SSHModel{T1,T2,T3},Gr::EstimateGreensFunction{T1}) where {T1,T2,T3}

    # size of lattice
    Lₜ = model.Lτ::Int
    L₁ = model.lattice.L1::Int
    L₂ = model.lattice.L2::Int
    L₂ = model.lattice.L3::Int
    nₒ = model.lattice.norbits::Int
    nᵤ = model.lattice.ncells::Int

    # displacement vectors describing each type of bond
    vectors = container["vectors"]

    # container to hold bond-bond correlation measurements
    measurement = container["position"]

    # length of axis corresponding to imaginary time axis
    L₀ = size(measurement,1)

    # number of vectors
    nᵥ = length(vectors)

    # phonon fields
    x = reshaped(model.x,(Lₜ,L₁,L₂,L₃,nᵥ))

    # containers
    x₁   = reshaped(view(Gr.a,  1:nᵤ*Lₜ),(Lₜ,L₁,L₂,L₃))
    x₂   = reshaped(view(Gr.b,  1:nᵤ*Lₜ),(Lₜ,L₁,L₂,L₃))
    x₁x₂ = reshaped(view(Gr.ab′,1:nᵤ*Lₜ),(Lₜ,L₁,L₂,L₃))

    # iterate over bonds
    for b₂ in 1:nᵥ
        # associated phonon fields
        @views @. x₂ = x[:,:,:,:,b₂]
        # iterate over bonds
        for b₁ in 1:nᵥ
            # associated phonon fields
            @views @. x₁ = x[:,:,:,:,b₁]
            # calculate translation average
            translational_average!(x₁x₂,x₁,x₂)
            # save the results
            if L₀==1 # equal time measurment
                @views @. measurements[1,:,:,:,b₁,b₂] += x₁x₂[1,:,:,:]
            else # unequal time measurement
                @views @. measurements[1:Lₜ,:,:,:,b₁,b₂] += x₁x₂
                # dealing with τ=β time slice
                @views @. measurements[Lₜ+1,:,:,:,b₁,b₂] += x₁x₂[1,:,:,:]
            end
        end
    end

    return nothing
end

end