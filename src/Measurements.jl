module Measurements

using Printf
using FFTW
using LinearAlgebra
using Parameters
using Statistics
using UnsafeArrays
using Parameters

using ..Utilities: get_index, get_site, get_τ, θ, δ, reshaped, translational_average!, simpson
using ..Models: HolsteinModel, SSHModel, AbstractModel
using ..SimulationParams: SimulationParameters
using ..GreensFunctions: EstimateGreensFunction, update!, estimate
using ..GreensFunctions: measure_GΔ0, measure_GΔ0_GΔ0, measure_GΔΔ_G00, measure_GΔ0_G0Δ

export initialize_measurements_container
export initialize_measurement_files!
export make_measurements!
export process_measurements!
export write_measurements!
export reset_measurements!

"""
Construct a container to hold the measurements. The input `info` is a dictionary containing the information
from the `measurements` table in the config file.
"""
function initialize_measurements_container(holstein::HolsteinModel{T1,T2,T3},info::Dict) where {T1,T2,T3}

    Lₜ = holstein.Lτ
    L₁ = holstein.lattice.L1
    L₂ = holstein.lattice.L2
    L₃ = holstein.lattice.L3
    nₒ = holstein.lattice.unit_cell.norbits

    # number of random vectors used to make measurements
    if haskey(info,"num_random_vectors")
        num_random_vectors = info["num_random_vectors"]
    else
        num_random_vectors = 1
    end

    container = Dict()
    container["global_meas"]    = Dict()
    container["onsite_meas"]    = Dict()
    container["onsite_corr"]    = Dict()
    container["intersite_meas"] = Dict()
    container["intersite_corr"] = Dict()
    container["n_rand_vecs"]    = num_random_vectors
    
    #########################
    ## GLOBAL MEASUREMENTS ##
    #########################
    
    container["global_meas"]["Nsqr"]    = Complex{T1}(0.0)
    container["global_meas"]["density"] = Complex{T1}(0.0)
    container["global_meas"]["mu"]      = Complex{T1}(0.0)

    ##########################    
    ## ON-SITE MEASUREMENTS ##
    ##########################

    container["onsite_meas"]["density"]     = zeros(Complex{T1},nₒ)
    container["onsite_meas"]["double_occ"]  = zeros(Complex{T1},nₒ)
    container["onsite_meas"]["x"]           = zeros(Complex{T1},nₒ)
    container["onsite_meas"]["x2"]          = zeros(Complex{T1},nₒ)
    container["onsite_meas"]["x4"]          = zeros(Complex{T1},nₒ)
    container["onsite_meas"]["phonon_pe"]   = zeros(Complex{T1},nₒ)
    container["onsite_meas"]["phonon_ke"]   = zeros(Complex{T1},nₒ)
    container["onsite_meas"]["elph_energy"] = zeros(Complex{T1},nₒ)
    container["onsite_meas"]["mu"]          = zeros(Complex{T1},nₒ)

    # determining whether s-wave susceptibility is measured
    if haskey(info,"PairGreens")
        if info["PairGreens"]["measure"]==true
            if info["PairGreens"]["time_dependent"]==true
                container["onsite_meas"]["swave_susc"] = zeros(Complex{T1},nₒ)
            end
        end
    end

    #############################
    ## INTER-SITE MEASUREMENTS ##
    #############################

    container["intersite_meas"]["el_ke"] = zeros(Complex{T1},length(holstein.bond_definitions))

    ###################################
    ## ON-SITE CORRELATION FUNCTIONS ##
    ###################################

    # electron green function
    init_corr_container!(container["onsite_corr"],"Greens",info,holstein,nₒ,L₃,L₂,L₁,Lₜ)

    # density-density correlation
    init_corr_container!(container["onsite_corr"],"DenDen",info,holstein,nₒ,L₃,L₂,L₁,Lₜ)

    # pair green function
    init_corr_container!(container["onsite_corr"],"PairGreens",info,holstein,nₒ,L₃,L₂,L₁,Lₜ)

    # phonon green function
    init_corr_container!(container["onsite_corr"],"PhononGreens",info,holstein,nₒ,L₃,L₂,L₁,Lₜ)

    ######################################
    ## INTER-SITE CORRELATION FUNCTIONS ##
    ######################################

    # bond-bond correlation function
    init_corr_container!(container["intersite_corr"],"BondBond",info,holstein,holstein.nbonds,L₃,L₂,L₁,Lₜ)

    # Current-Current correlation function
    init_corr_container!(container["intersite_corr"],"CurrentCurrent",info,holstein,holstein.nbonds,L₃,L₂,L₁,Lₜ)

    ##########################################
    ## CONVERTING FROM DICTS TO NAMEDTUPLES ##
    ##########################################

    container["onsite_meas"]    = (;(Symbol(k)=>container["onsite_meas"][k] for k in keys(container["onsite_meas"]))...)
    container["intersite_meas"] = (;(Symbol(k)=>container["intersite_meas"][k] for k in keys(container["intersite_meas"]))...)
    container["onsite_corr"]    = (;(Symbol(k)=>container["onsite_corr"][k] for k in keys(container["onsite_corr"]))...)
    container["intersite_corr"] = (;(Symbol(k)=>container["intersite_corr"][k] for k in keys(container["intersite_corr"]))...)
    container                   = (;(Symbol(k) => container[k] for k in keys(container))...)

    return container
end

function initialize_measurements_container(ssh::SSHModel{T1,T2,T3},info::Dict) where {T1,T2,T3}

    Lₜ = ssh.Lτ
    L₁ = ssh.lattice.L1
    L₂ = ssh.lattice.L2
    L₃ = ssh.lattice.L3
    nₒ = ssh.lattice.unit_cell.norbits

    # number of random vectors used to make measurements
    if haskey(info,"num_random_vectors")
        num_random_vectors = info["num_random_vectors"]
    else
        num_random_vectors = 1
    end

    container = Dict()
    container["global_meas"]    = Dict()
    container["onsite_meas"]    = Dict()
    container["onsite_corr"]    = Dict()
    container["intersite_meas"] = Dict()
    container["intersite_corr"] = Dict()
    container["n_rand_vecs"]    = num_random_vectors

    
    #########################
    ## GLOBAL MEASUREMENTS ##
    #########################
    
    container["global_meas"]["density"] = Complex{T1}(0.0)
    container["global_meas"]["Nsqr"]    = Complex{T1}(0.0)
    container["global_meas"]["mu"]      = Complex{T1}(0.0)

    ##########################    
    ## ON-SITE MEASUREMENTS ##
    ##########################

    container["onsite_meas"]["density"]    = zeros(Complex{T1},nₒ)
    container["onsite_meas"]["double_occ"] = zeros(Complex{T1},nₒ)
    container["onsite_meas"]["mu"]         = zeros(Complex{T1},nₒ)

    # determining whether s-wave susceptibility is measured
    if haskey(info,"PairGreens")
        if info["PairGreens"]["measure"]==true
            if info["PairGreens"]["time_dependent"]==true
                container["onsite_meas"]["swave_susc"] = zeros(Complex{T1},nₒ)
            end
        end
    end


    #############################
    ## INTER-SITE MEASUREMENTS ##
    #############################

    # number of phonon species
    nᵥ = ssh.nbonds

    # intialize containers for measurements
    container["intersite_meas"]["x"]           = zeros(Complex{T1},nᵥ)
    container["intersite_meas"]["x2"]          = zeros(Complex{T1},nᵥ)
    container["intersite_meas"]["x4"]          = zeros(Complex{T1},nᵥ)
    container["intersite_meas"]["phonon_pe"]   = zeros(Complex{T1},nᵥ)
    container["intersite_meas"]["phonon_ke"]   = zeros(Complex{T1},nᵥ)
    container["intersite_meas"]["elph_energy"] = zeros(Complex{T1},nᵥ)
    container["intersite_meas"]["el_ke"]       = zeros(Complex{T1},nᵥ)

    ###################################
    ## ON-SITE CORRELATION FUNCTIONS ##
    ###################################

    # electron green function
    init_corr_container!(container["onsite_corr"],"Greens",info,ssh,nₒ,L₃,L₂,L₁,Lₜ)

    # density-density correlation
    init_corr_container!(container["onsite_corr"],"DenDen",info,ssh,nₒ,L₃,L₂,L₁,Lₜ)

    # pair green function
    init_corr_container!(container["onsite_corr"],"PairGreens",info,ssh,nₒ,L₃,L₂,L₁,Lₜ)

    ######################################
    ## INTER-SITE CORRELATION FUNCTIONS ##
    ######################################

    # phonon greens function
    if ssh.nph>0
        init_corr_container!(container["intersite_corr"],"PhononGreens",info,ssh,ssh.nph,L₃,L₂,L₁,Lₜ)
    end

    # bond-bond correlation function
    init_corr_container!(container["intersite_corr"],"BondBond",info,ssh,ssh.nbonds,L₃,L₂,L₁,Lₜ)

    # current-current correlation function
    init_corr_container!(container["intersite_corr"],"CurrentCurrent",info,ssh,ssh.nbonds,L₃,L₂,L₁,Lₜ)
    
    ##########################################
    ## CONVERTING FROM DICTS TO NAMEDTUPLES ##
    ##########################################

    container["onsite_meas"]    = (;(Symbol(k)=>container["onsite_meas"][k] for k in keys(container["onsite_meas"]))...)
    container["intersite_meas"] = (;(Symbol(k)=>container["intersite_meas"][k] for k in keys(container["intersite_meas"]))...)
    container["onsite_corr"]    = (;(Symbol(k)=>container["onsite_corr"][k] for k in keys(container["onsite_corr"]))...)
    container["intersite_corr"] = (;(Symbol(k)=>container["intersite_corr"][k] for k in keys(container["intersite_corr"]))...)
    container                   = (;(Symbol(k) => container[k] for k in keys(container))...)

    return container
end

"""
Initialize Measurement Files.
"""
function initialize_measurement_files!(container::NamedTuple,sim_params::SimulationParameters)

    #############################################
    ## Initialize File For Global Measurements ##
    #############################################

    open(joinpath(sim_params.datafolder,"global_measurements.out"), "w") do file
        write( file, "bin")
        for key in keys(container.global_meas)
            write(file, ",", string(key))
        end
        write(file,"\n")
    end

    ##############################################
    ## Initialize File For On-Site Measurements ##
    ##############################################

    open(joinpath(sim_params.datafolder,"onsite_measurements.out"), "w") do file
        write(file, "bin,orbit")
        for key in keys(container.onsite_meas)
            measurement = string(key)
            write(file, ",", measurement)
        end
        write(file, "\n")
    end

    #################################################
    ## Initialize File For Inter-Site Measurements ##
    #################################################

    open(joinpath(sim_params.datafolder,"intersite_measurements.out"), "w") do file
        write(file, "bin,vector")
        for key in keys(container.intersite_meas)
            measurement = string(key)
            write(file, ",", measurement)
        end
        write(file, "\n")
    end

    ###############################################
    ## Initialize Files For On-Site Correlations ##
    ###############################################

    # iterate over on-site correlation functions
    for k in keys(container.onsite_corr)
        # measurement name
        measurement = string(k)
        # initialize file for position-space data
        open(joinpath(sim_params.datafolder,measurement*"_position.out"), "w") do file
            # writing file header
            write(file, "bin", ",", "orbit1", ",", "orbit2", ",", "r3",  ",", "r2",  ",", "r1", ",", "tau", ",", measurement, "\n")
        end
        # initialize file for position-space data
        open(joinpath(sim_params.datafolder,measurement*"_momentum.out"), "w") do file
            # writing file header
            write(file, "bin", ",", "orbit1", ",", "orbit2", ",", "k3",  ",", "k2",  ",", "k1", ",", "tau", ",", measurement, "\n")
        end
    end

    ##################################################
    ## Initialize Files For Inter-Site Correlations ##
    ##################################################

    # iterate over on-site correlation functions
    for k in keys(container.intersite_corr)
        # measurement name
        measurement = string(k)
        # initialize file for position-space data
        open(joinpath(sim_params.datafolder,measurement*"_position.out"), "w") do file
            # writing file header
            write(file, "bin", ",", "vector1", ",", "vector2", ",", "r3",  ",", "r2",  ",", "r1", ",", "tau", ",", measurement, "\n")
        end
        # initialize file for position-space data
        open(joinpath(sim_params.datafolder,measurement*"_momentum.out"), "w") do file
            # writing file header
            write(file, "bin", ",", "vector1", ",", "vector2", ",", "k3",  ",", "k2",  ",", "k1", ",", "tau", ",", measurement, "\n")
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
    fourier_transform_correlations!(container.intersite_corr)

    ############################
    ## NORMALIZE MEASUREMENTS ##
    ############################

    # bin size
    bin_size = sim_params.bin_size

    # number of random vectors used to make measurements
    n_rand_vecs = container.n_rand_vecs

    # normalization constant
    V = bin_size * n_rand_vecs

    # normalize global measurements
    global_meas = container.global_meas
    for key in keys(global_meas)
        global_meas[key] /= V
    end

    # normalize on-site measurements
    onsite_meas = container.onsite_meas
    for key in keys(onsite_meas)
        onsite_meas[key] ./= V
    end

    # normalize inter-site measurements
    intersite_meas = container.intersite_meas
    for key in keys(intersite_meas)
        intersite_meas[key] ./= V
    end

    # normalize on-site correlation
    onsite_corr = container.onsite_corr
    for key in keys(onsite_corr)
        onsite_corr[key].position ./= V
        onsite_corr[key].momentum ./= V
    end

    # normalize inter-site correlation
    intersite_corr = container.intersite_corr
    for key in keys(intersite_corr)
        intersite_corr[key].position ./= V
        intersite_corr[key].momentum ./= V
    end

    ###################################
    ## MEASURE INTEGRATED QUANTITIES ##
    ###################################

    measure_swave!(container,model)

    return nothing
end

"""
Write measurements to file.
"""
function write_measurements!(container::NamedTuple,sim_params::SimulationParameters,model::AbstractModel{T1,T2},bin::Int) where {T1,T2}

    write_global_measurements!(    container.global_meas,    sim_params, model, bin)
    write_onsite_measurements!(    container.onsite_meas,    sim_params, model, bin)
    write_intersite_measurements!( container.intersite_meas, sim_params, model, bin)
    write_correlations!(           container.onsite_corr,    sim_params, model, bin)
    write_correlations!(           container.intersite_corr, sim_params, model, bin)

    return nothing
end

"""
Reset the measurements container i.e. resent all values to zero.
"""
function reset_measurements!(container::NamedTuple,model::AbstractModel{T1,T2}) where {T1,T2}

    # reset global measurements
    global_measurements = container.global_meas
    for key in keys(global_measurements)
        global_measurements[key] = 0.0
    end

    # reset on-site measurements
    onsite_measurements = container.onsite_meas
    for key in keys(onsite_measurements)
        measurement = onsite_measurements[key]::Vector{Complex{T1}}
        fill!(measurement,0.0)
    end

    # reset inter-site measurements
    intersite_measurements = container.intersite_meas
    for key in keys(intersite_measurements)
        measurement = intersite_measurements[key]::Vector{Complex{T1}}
        fill!(measurement,0.0)
    end

    # reset on-site correlations
    onsite_correlations = container.onsite_corr
    for key  in keys(onsite_correlations)
        position = onsite_correlations[key].position::Array{Complex{T1},6}
        momentum = onsite_correlations[key].momentum::Array{Complex{T1},6}
        fill!(position,0.0)
        fill!(momentum,0.0)
    end

    # reset inter-site correlations
    intersite_correlations = container.intersite_corr
    for key in keys(intersite_correlations)
        position = intersite_correlations[key].position::Array{Complex{T1},6}
        momentum = intersite_correlations[key].momentum::Array{Complex{T1},6}
        fill!(position,0.0)
        fill!(momentum,0.0)
    end

    return nothing
end

#####################
## PRIVATE METHODS ##
#####################

"""
Intialize multi-dimensional array to contain correlation measurement.
"""
function init_corr_container!(container::Dict,measurement::String,info::Dict,model::AbstractModel{T1,T2,T3},
                                        n::Int,L₃::Int,L₂::Int,L₁::Int,Lₜ::Int) where {T1,T2,T3}
    
    if haskey(info,measurement)
        if info[measurement]["measure"]==true
            container[measurement] = Dict()
            # declare multi-dimnesional arrays to hold measurement
            if info[measurement]["time_dependent"]==true
                position = zeros(Complex{T1},Lₜ+1,L₁,L₂,L₃,n,n)
                momentum = zeros(Complex{T1},Lₜ+1,L₁,L₂,L₃,n,n)
            else
                position = zeros(Complex{T1},1,L₁,L₂,L₃,n,n)
                momentum = zeros(Complex{T1},1,L₁,L₂,L₃,n,n)
            end
            container[measurement] = (position=position,momentum=momentum)
        end
    end
    return nothing
end

"""
Make global measurements.
"""
function make_global_measurements!(container::NamedTuple,model::AbstractModel,Gr::EstimateGreensFunction)

    global_meas = container.global_meas::Dict

    # measure density ⟨n̂⟩
    n  = measure_density(model,Gr)
    global_meas["density"] += n

    # measure ⟨N̂²⟩
    N² = measure_N²(model,Gr)
    global_meas["Nsqr"] += N²

    # measure μ
    global_meas["mu"] += mean(model.μ)

    return nothing
end

"""
Measure on-site correlation functions
"""
function measure_onsite_correlations!(container::NamedTuple,model::HolsteinModel,Gr::EstimateGreensFunction)

    onsite_corr = container.onsite_corr

    for measurement in keys(onsite_corr)
        if measurement == :Greens
            measure_Greens!(onsite_corr.Greens.position,model,Gr)
        elseif measurement == :DenDen
            measure_DenDen!(onsite_corr.DenDen.position,model,Gr)
        elseif measurement == :PairGreens
            measure_PairGreens!(onsite_corr.PairGreens.position,model,Gr)
        elseif measurement == :PhononGreens
            measure_PhononGreens!(onsite_corr.PhononGreens.position,model,Gr)
        end
    end

    return nothing
end

function measure_onsite_correlations!(container::NamedTuple,model::SSHModel,Gr::EstimateGreensFunction)

    onsite_corr = container.onsite_corr

    for measurement in keys(onsite_corr)
        if measurement == :Greens
            measure_Greens!(onsite_corr.Greens.position,model,Gr)
        elseif measurement == :DenDen
            measure_DenDen!(onsite_corr.DenDen.position,model,Gr)
        elseif measurement == :PairGreens
            measure_PairGreens!(onsite_corr.PairGreens.position,model,Gr)
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
        if measurement == :BondBond
            measure_BondBond!(intersite_corr.BondBond.position,model,Gr)
        elseif measurement == :CurrentCurrent
            measure_CurrentCurrent!(intersite_corr.CurrentCurrent.position,model,Gr)
        end
    end

    return nothing
end

function measure_intersite_correlations!(container::NamedTuple,model::SSHModel,Gr::EstimateGreensFunction)

    intersite_corr = container.intersite_corr

    for measurement in keys(intersite_corr)
        if measurement == :BondBond
            measure_BondBond!(intersite_corr.BondBond.position,model,Gr)
        elseif measurement == :PhononGreens
            measure_PhononGreens!(intersite_corr.PhononGreens.position,model,Gr)
        elseif measurement == :CurrentCurrent
            measure_CurrentCurrent!(intersite_corr.CurrentCurrent.position,model,Gr)
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
        # iterating over orbits of the current type
        for site in orbit:norbits:nsites
            # iterating over time slices
            for τ in 1:Lτ
                # getting current index
                index = get_index(τ,site,Lτ)
                # estimate ⟨cᵢ(τ)c⁺ᵢ(τ)⟩
                G1 = estimate(Gr,site,site,τ,τ,1)
                G2 = estimate(Gr,site,site,τ,τ,2)
                # measure density
                onsite_meas.density[orbit] += ((1.0-G1)+(1.0-G2)) / normalization
                # measure double occupancy
                onsite_meas.double_occ[orbit] += (1.0-G1)*(1.0-G2) / normalization
                # measuring phonon kinetic energy such that
                # ⟨KE⟩ = 1/(2Δτ) - ⟨(1/2)[xᵢ(τ+1)-xᵢ(τ)]²/Δτ²⟩
                Δx = x[get_index(τ%Lτ+1,site,Lτ)]-x[index]
                onsite_meas.phonon_ke[orbit] += (0.5/Δτ-(Δx^2)/Δτ²/2) / normalization
                # measuring phonon potential energy
                onsite_meas.phonon_pe[orbit] += (model.ω[site]^2*x[index]^2/2 + model.ω₄[site]*x[index]^4) / normalization
                # measuring the electron phonon energy λ⟨x⋅(n₊+n₋)⟩
                onsite_meas.elph_energy[orbit] += model.λ[site]*x[index]*(2.0-G1-G2) / normalization
                # measure ⟨x⟩
                onsite_meas.x[orbit]  += x[index] / normalization
                # measure ⟨x²⟩
                onsite_meas.x2[orbit] += x[index]^2 / normalization
                # measure ⟨x⁴⟩
                onsite_meas.x4[orbit] += x[index]^4 / normalization
                # measure chemical potential
                onsite_meas.mu[orbit] += model.μ[site] / normalization
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
        # iterating over orbits of the current type
        for site in orbit:norbits:nsites
            # iterating over time slices
            for τ in 1:Lτ
                # getting current index
                index = get_index(τ,site,Lτ)
                # estimate ⟨cᵢ(τ)c⁺ᵢ(τ)⟩
                G1 = estimate(Gr,site,site,τ,τ,1)
                G2 = estimate(Gr,site,site,τ,τ,2)
                # measure density
                onsite_meas.density[orbit] += ((1.0-G1)+(1.0-G2)) / normalization
                # measure double occupancy
                onsite_meas.double_occ[orbit] += (1.0-G1)*(1.0-G2) / normalization
                # measure chemical potential
                onsite_meas.mu[orbit] += model.μ[site] / normalization
            end
        end
    end

    return nothing
end

"""
Make inter-site measurements.
"""
function make_intersite_measurements!(container::NamedTuple,model::HolsteinModel{T1,T2,T3},Gr::EstimateGreensFunction) where{T1,T2,T3}

    # number of types of bonds in lattice
    nbonds = length(model.bond_definitions)

    # number of unit cells in lattice
    ncells = model.lattice.ncells::Int

    # length of imaginarty time axis
    Lτ = model.Lτ

    # normalization constant
    V = ncells*Lτ

    # iterate of types of bonds
    for bond_def in 1:nbonds
        # iterate over unit cells
        for cell in 1:ncells
            # get bond
            bond = (bond_def-1)*ncells + cell
            # get pair of neighboring sites assoicated with bond
            index = model.checkerboard_perm[bond]
            s₁    = model.neighbor_table[1,bond]
            s₂    = model.neighbor_table[2,bond]
            # get hopping amplitude
            t = model.t[bond]
            # iterate over imaginary time slices
            for τ in 1:Lτ
                # get hopping amplitude h = ∑ₛ⟨c⁺ₛᵢcₛⱼ+h.c.⟩
                G1 = estimate(Gr,s₁,s₂,τ,τ,1)
                G2 = estimate(Gr,s₂,s₁,τ,τ,1)
                G3 = estimate(Gr,s₁,s₂,τ,τ,2)
                G4 = estimate(Gr,s₂,s₁,τ,τ,2)
                h  = -G1-G2-G3-G4
                # calculate electron kinetic energy
                container.intersite_meas.el_ke[bond_def] += -t*h/V
            end
        end
    end

    return nothing
end

function make_intersite_measurements!(container::NamedTuple,ssh::SSHModel{T1,T2,T3},Gr::EstimateGreensFunction{T1}) where {T1,T2,T3}

    # container for measurements
    intersite_meas = container.intersite_meas

    # length of imaginary time axis
    Lτ = ssh.Lτ

    # number of bonds in lattices
    Nbonds = ssh.Nbonds

    # number of types of bonds lattice
    nbonds = ssh.nbonds

    # phonon fields
    x = reshaped(ssh.x,(Lτ,ssh.Nph))

    # imaginary time step
    Δτ = ssh.Δτ

    # normalization
    V = div(Nbonds,nbonds)*Lτ

    # keeps track of phonon species
    phonon_species = 0

    # iterate over bonds in lattice
    for bond in 1:ssh.Nbonds
        # get bond definitions
        bond_def = ssh.bond_to_definition[bond]
        # get phonon
        phonon = ssh.bond_to_phonon[bond]
        # get hopping energy
        t = ssh.t[bond]
        # get pair of sites associated with bond
        index = ssh.checkerboard_perm[bond]
        s₁    = ssh.neighbor_table[1,index]
        s₂    = ssh.neighbor_table[2,index]
        # get phonon parameters
        if phonon!=0
            ω = ssh.ω[phonon]
            α = ssh.α[phonon]
        else
            ω = 0.0
            α = 0.0
        end
        # iterate over time slices
        for τ in 1:Lτ
            # get phonon field info
            if phonon!=0
                # get xᵢ[τ] phonon field
                xτ   = x[τ,phonon]
                # get xᵢ[τ+1] phonon field
                xτp1 = x[mod1(τ+1,Lτ),phonon]
                # Δx = xᵢ[τ+1]-xᵢ[τ]
                Δx   = xτp1-xτ
            else
                xτ   = 0.0
                xτp1 = 0.0
                Δx   = 0.0
            end
            # get hopping amplitude h = ∑ₛ⟨c⁺ₛᵢcₛⱼ+h.c.⟩
            G1 = estimate(Gr,s₁,s₂,τ,τ,1)
            G2 = estimate(Gr,s₂,s₁,τ,τ,1)
            G3 = estimate(Gr,s₁,s₂,τ,τ,2)
            G4 = estimate(Gr,s₂,s₁,τ,τ,2)
            h   = -G1-G2-G3-G4
            # calculate modulated hopping amplitude
            t′ = ssh.t′[τ,bond]
            # phonon potential energy
            intersite_meas.phonon_pe[bond_def]   += (ω^2*xτ^2/2)/V
            # phonon kinetic energy
            intersite_meas.phonon_ke[bond_def]   += (0.5/Δτ-(Δx/Δτ)^2/2.0)/V
            # electron-phonon energy
            intersite_meas.elph_energy[bond_def] += (α*h*xτ)/V
            # ⟨x⟩
            intersite_meas.x[bond_def]           += xτ/V
            # ⟨x²⟩
            intersite_meas.x2[bond_def]          += (xτ^2)/V
            # ⟨x⁴⟩
            intersite_meas.x4[bond_def]          += (xτ^4)/V
            # calculate electron kinetic energy
            intersite_meas.el_ke[bond_def]       += -t′*h/V
        end
    end

    return nothing
end

"""
Fourier transform position space correlation functions to momentum space.
"""
function fourier_transform_correlations!(container::NamedTuple)

    for measurement in keys(container)
        position = container[measurement].position
        momentum = container[measurement].momentum
        copyto!(momentum,position)
        fft!(momentum,(2,3,4))
    end

    return nothing
end

"""
Write global measurements to file.
"""
function write_global_measurements!(container::Dict,sim_params::SimulationParameters,model::AbstractModel{T1,T2,T3},bin::Int) where {T1,T2,T3}

    filename = joinpath(sim_params.datafolder,"global_measurements.out")

    open(filename,"a") do file

        write(file,string(bin),",")
        line = join((real(container[k]) for k in keys(container)), ",")
        write(file,line,"\n")
    end

    return nothing
end

"""
Write on-site measurements to file.
"""
function write_onsite_measurements!(container::NamedTuple,sim_params::SimulationParameters,model::AbstractModel{T1,T2,T3},bin::Int) where {T1,T2,T3}

    # number of orbitals per unit cell
    nₒ = model.lattice.unit_cell.norbits::Int

    # filename
    filename = joinpath(sim_params.datafolder,"onsite_measurements.out")

    open(filename,"a") do file
        # iterate over orbitals
        for o in 1:nₒ
            write(file,string(bin),",")
            write(file, string(o))
            for measurement in keys(container)
                write( file , @sprintf(",%.6f",real(container[measurement][o])) )
            end
            write(file, "\n")
        end
    end

    return nothing
end

"""
Write inter-site measurements to file.
"""
function write_intersite_measurements!(container::NamedTuple,sim_params::SimulationParameters,model::AbstractModel{T1,T2,T3},bin::Int) where {T1,T2,T3}

    # filename
    filename = joinpath(sim_params.datafolder,"intersite_measurements.out")

    open(filename,"a") do file
        # iterate over bonds
        for bond in 1:model.nbonds
            write(file,string(bin),",")
            write(file, string(bond))
            for measurement in keys(container)
                write( file , @sprintf(",%.6f",real(container[measurement][bond])) )
            end
            write(file, "\n")
        end
    end

    return nothing
end

"""
Write all different correlation functions to file.
"""
function write_correlations!(container::NamedTuple,sim_params::SimulationParameters,model::AbstractModel{T1,T2,T3},bin::Int) where {T1,T2,T3}

    # iterate over on-site correlation functions
    for key in keys(container)

        # write position space correlations to file
        filename = joinpath(sim_params.datafolder,string(key)*"_position.out")
        write_correlation!(filename,container[key].position,bin)

        # write momemtum space correlations to file
        filename = joinpath(sim_params.datafolder,string(key)*"_momentum.out")
        write_correlation!(filename,container[key].momentum,bin)
    end

    return nothing
end

"""
Write a correlation function to file.
"""
function write_correlation!(filename::String,correlations::Array{Complex{T},6},bin::Int) where {T<:AbstractFloat}

    open(filename,"a") do file
        Lₜ = size(correlations,1)
        L₁ = size(correlations,2)
        L₂ = size(correlations,3)
        L₃ = size(correlations,4)
        n  = size(correlations,5)
        for n₁ in 1:n
            for n₂ in 1:n
                for l₃ in 1:L₃
                    for l₂ in 1:L₂
                        for l₁ in 1:L₁
                            for τ in 1:Lₜ
                                line = @sprintf("%d,%d,%d,%d,%d,%d,%d,%.6f\n",bin,n₁,n₂,l₃-1,l₂-1,l₁-1,τ-1,real(correlations[τ,l₁,l₂,l₃,n₂,n₁]))
                                write(file,line)
                            end
                        end
                    end
                end
            end
        end
    end
    return nothing
end

######################################
## IMPLEMENTING GLOBAL MEASUREMENTS ##
######################################

"""
Measure density ⟨n̂⟩.
"""
function measure_density(model::AbstractModel,estimator::EstimateGreensFunction)

    @unpack r₁, M⁻¹r₁, r₂, M⁻¹r₂, N, L = estimator

    N₁ = 2*(N - dot(M⁻¹r₁,r₁)/L)
    N₂ = 2*(N - dot(M⁻¹r₂,r₂)/L)
    n  = (N₁+N₂)/(2*N)

    return n
end

"""
Measure ⟨N̂²⟩.
"""
function measure_N²(model::AbstractModel,estimator::EstimateGreensFunction)

    @unpack r₁, M⁻¹r₁, r₂, M⁻¹r₂, GΔ0, GΔ0_G0Δ, L, N, NL, nₛ = estimator
    @unpack β = model

    N² = 0.0
    @uviews GΔ0_G0Δ begin
        TrG₁    = dot(M⁻¹r₁,r₁)/L
        TrG₂    = dot(M⁻¹r₂,r₂)/L
        N₁      = 2*(N - TrG₁)
        N₂      = 2*(N - TrG₂)
        Gr0_G0r = @view GΔ0_G0Δ[1,:,:,:,:,:]
        N²     += N₁*N₂ + TrG₁ + TrG₂ - 2*(N/nₛ)*sum(Gr0_G0r)
    end

    return N²
end

"""
Measure the compressibility κ.
β  = inverse temperature
N   = system size/sites in lattice
N²  = ⟨N̂²⟩
ΔN² = error in measurement of ⟨N̂²⟩
n   = density ⟨n̂⟩
Δn  = error in measurement of density ⟨n̂⟩
"""
function measure_κ(β,N,N²,ΔN²,n,Δn)

    # calculate ⟨N̂⟩
    N̄ = N * n

    # calculate ⟨N̂⟩²
    N̄² = (N̄)^2

    # calculate error in measurement of ⟨N̂⟩
    ΔN̄ = N*Δn

    # calculate error in measurement of ⟨N̂⟩²
    ΔN̄² = 2*N̄*ΔN̄

    # calculate κ = β⋅(⟨N̂²⟩-⟨N̂⟩²)
    κ = β*(N²-N̄²)

    # calculate error in measurement of κ
    Δκ = sqrt( ΔN²^2 + ΔN̄²^2 )

    return κ/N, Δκ/N
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

    if τ==L
        # Pᵣ(β) = Pᵣ(0) + δᵣ(1-G↑₀(0)-G↓₀(0))
        Pᵣτ = measure_GΔ0_GΔ0(estimator,l₁,l₂,l₃,o₁,o₂,0)
        if l₁==0 && l₂==0 && l₃==0 && o₁==o₂
            G₀  = measure_GΔ0(estimator,0,0,0,o₁,o₁,0)
            Pᵣτ = Pᵣτ + 1.0 - 2*G₀
        end
    else
        # Pᵣ(τ) = ⟨Δᵢ₊ᵣ(τ)⋅Δ⁺ᵢ(0)⟩ = ⟨c↑ᵢ₊ᵣ(τ)⋅c⁺↑ᵢ(0)⟩⋅⟨c↓ᵢ₊ᵣ(τ)⋅c⁺↓ᵢ(0)⟩
        Pᵣτ = measure_GΔ0_GΔ0(estimator,l₁,l₂,l₃,o₁,o₂,τ)
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
            for o₂ in 1:n₀
                for o₁ in 1:n₀
                    for l₃ in 0:L₃-1
                        for l₂ in 0:L₂-1
                            for l₁ in 0:L₁-1
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
    L₃ = model.lattice.L3::Int
    nₒ = model.lattice.unit_cell.norbits::Int
    nᵤ = model.lattice.ncells::Int

    # length of axis corresponding to imaginary time.
    L₀ = size(container,1)

    # get phonon fields
    x₀ = model.x

    @uviews a b ab′ x₀ container begin

        # phonon fields
        x = reshaped(x₀,(Lₜ,nₒ,L₁,L₂,L₃))

        # containers for performing calculation
        x₁x₂ = reshaped(view(ab′,1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        x₁   = reshaped(view(a,  1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        x₂   = reshaped(view(b,  1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))

        for o₂ in 1:nₒ
            # phonon fields for orbital o₂
            @views @. x₂ = x[:,o₂,:,:,:]
            for o₁ in 1:nₒ
                # phonon fields for orbital o₁
                @views @. x₁ = x[:,o₁,:,:,:]
                # translationally average
                translational_average!(x₁x₂,x₁,x₂)
                # save the results
                if L₀==1 # equal time measurment
                    @views @. container[1,:,:,:,o₁,o₂] += x₁x₂[1,:,:,:]
                else # unequal time measurement
                    @views @. container[1:Lₜ,:,:,:,o₁,o₂] += x₁x₂
                    # dealing with τ=β time slice
                    @views @. container[Lₜ+1,:,:,:,o₁,o₂] += x₁x₂[1,:,:,:]
                end
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
function measure_BondBond!(container::Array{Complex{T1},6},model::AbstractModel{T1,T2,T3},estimator::EstimateGreensFunction{T1}) where {T1,T2,T3}

    # size of lattice
    Lₜ = model.Lτ::Int
    L₁ = model.lattice.L1::Int
    L₂ = model.lattice.L2::Int
    L₃ = model.lattice.L3::Int
    nₒ = model.lattice.unit_cell.norbits::Int
    nᵤ = model.lattice.ncells::Int

    @unpack r₁, M⁻¹r₁, r₂, M⁻¹r₂, ab″, ab′, a, b, = estimator

    @uviews container r₁ M⁻¹r₁ r₂ M⁻¹r₂ ab″ ab′ a b begin

        r₁    = reshaped(r₁,    (Lₜ,nₒ,L₁,L₂,L₃))
        M⁻¹r₁ = reshaped(M⁻¹r₁, (Lₜ,nₒ,L₁,L₂,L₃))
        r₂    = reshaped(r₂,    (Lₜ,nₒ,L₁,L₂,L₃))
        M⁻¹r₂ = reshaped(M⁻¹r₂, (Lₜ,nₒ,L₁,L₂,L₃))

        # containers for performing calculation
        bondbond = reshaped(view(ab″,1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        G₁G₂     = reshaped(view(ab′,1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        G₁       = reshaped(view(a,  1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        G₂       = reshaped(view(b,  1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        R₁′      = G₁
        R₁″      = G₁
        R₂′      = G₂
        R₂″      = G₂

        # displacement vectors describing each type of bond
        bonds = model.bond_definitions

        # length of axis corresponding to imaginary time axis
        L₀ = size(container,1)

        # number of vectors
        nᵥ = length(bonds)

        # iterate over first bond
        for n″ in 1:nᵥ

            # vector associated with bond going from orbitals d ⟶ c displaced r″ unit cells
            r″ = bonds[n″].v::Vector{Int} # displacement in unit cells
            d  = bonds[n″].o₁::Int # starting orbital
            c  = bonds[n″].o₂::Int # ending   orbital

            # iterate over second bond
            for n′ in 1:nᵥ

                # initialize bond-bond correlation to zero
                fill!(bondbond,0.0)

                # vector associated with bond going from orbitals b ⟶ a displaced r′ unit cells
                r′ = bonds[n′].v::Vector{Int} # displacement in unit cells
                b  = bonds[n′].o₁::Int # starting orbital
                a  = bonds[n′].o₂::Int # ending   orbital

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
                @. G₂ = M⁻¹R₁ * R₂′                      # G₂    = [M⁻¹R₁⋅R₂′]
                @. G₁ = M⁻¹R₂ * R₁″                      # G₁    = [M⁻¹R₂⋅R₁″]
                translational_average!(G₁G₂,G₁,G₂)       # G₂⋅G₁ = [M⁻¹R₁⋅R₂′]⋆[M⁻¹R₂⋅R₁″]

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
                    G  = measure_GΔ0(estimator,l₁,l₂,l₃,c,b,0)

                    # B = B + 2⋅δ(τ)⋅δ(r′+r)⋅δ(a,d)⋅⟨b(i+r,τ)⋅c⁺(i+r″,0)⟩
                    bondbond[1,l₁+1,l₂+1,l₃+1] += 2*G
                end

                # record measurements
                if L₀==1 # if time independent measurement
                    @views @. container[1,:,:,:,n′,n″]    += bondbond[1,:,:,:]
                else # if time dependent measurement
                    @views @. container[1:Lₜ,:,:,:,n′,n″] += bondbond
                    # deal with τ=β time slice
                    for l₃ in 0:L₃-1
                        for l₂ in 0:L₂-1
                            for l₁ in 0:L₁-1
                                nl₁ = mod(-l₁,L₁)
                                nl₂ = mod(-l₂,L₂)
                                nl₃ = mod(-l₃,L₃)
                                # B[a,b,r′;c,d,r″](β,r) = B[c,d,r″;a,b,r′](0,-r)
                                container[Lₜ+1,l₁+1,l₂+1,l₃+1,n″,n′] += bondbond[1,nl₁+1,nl₂+1,nl₃+1]
                            end
                        end
                    end
                end
            end
        end
    end

    return nothing
end

"""
Measure current-current correlation.
"""
function measure_CurrentCurrent!(container::Array{Complex{T1},6},model::HolsteinModel{T1,T2,T3},estimator::EstimateGreensFunction{T1}) where {T1,T2,T3}

    # size of lattice
    Lₜ = model.Lτ::Int
    L₁ = model.lattice.L1::Int
    L₂ = model.lattice.L2::Int
    L₃ = model.lattice.L3::Int
    nₒ = model.lattice.unit_cell.norbits::Int
    nᵤ = model.lattice.ncells::Int # number of unit cells in lattice

    @unpack r₁, M⁻¹r₁, r₂, M⁻¹r₂, ab″, ab′, a, b, = estimator
    t₀ = model.t

    @uviews container r₁ M⁻¹r₁ r₂ M⁻¹r₂ ab″ ab′ a b t₀ begin

        r₁    = reshaped(r₁,    (Lₜ,nₒ,L₁,L₂,L₃))
        M⁻¹r₁ = reshaped(M⁻¹r₁, (Lₜ,nₒ,L₁,L₂,L₃))
        r₂    = reshaped(r₂,    (Lₜ,nₒ,L₁,L₂,L₃))
        M⁻¹r₂ = reshaped(M⁻¹r₂, (Lₜ,nₒ,L₁,L₂,L₃))

        # containers for performing calculation
        crntcrnt = reshaped(view(ab″,1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        G₁G₂     = reshaped(view(ab′,1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        G₁       = reshaped(view(a,  1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        G₂       = reshaped(view(b,  1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        R₁′      = G₁
        R₁″      = G₁
        M⁻¹R₁′   = G₁
        M⁻¹R₁″   = G₁
        R₂′      = G₂
        R₂″      = G₂
        M⁻¹R₂′   = G₂
        M⁻¹R₂″   = G₂

        # displacement vectors describing each type of bond
        bonds = model.bond_definitions

        # length of axis corresponding to imaginary time axis
        L₀ = size(container,1)

        # number of vectors
        nᵥ = length(bonds)

        # hopping ampltidues
        t = reshaped(t₀,(L₁,L₂,L₃,nᵥ))

        # iterate over first bond
        for n″ in 1:nᵥ

            # vector associated with bond going from orbitals d ⟶ c displaced r″ unit cells
            r″ = bonds[n″].v::Vector{Int} # displacement in unit cells
            d  = bonds[n″].o₁::Int # starting orbital
            c  = bonds[n″].o₂::Int # ending   orbital
            t″ = @view t[:,:,:,n″] # hopping amplitudes associated with bond

            # iterate over second bond
            for n′ in 1:nᵥ

                # initialize bond-bond correlation to zero
                fill!(crntcrnt,0.0)

                # vector associated with bond going from orbitals b ⟶ a displaced r′ unit cells
                r′ = bonds[n′].v::Vector{Int} # displacement in unit cells
                b  = bonds[n′].o₁::Int # starting orbital
                a  = bonds[n′].o₂::Int # ending   orbital
                t′ = @view t[:,:,:,n′] # hopping amplitudes associated with bond

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) a⁺(r′+r+i,τ)⟩⋅⟨c(r″+i,0) d⁺(i,0)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                R₁    = @view    r₁[:,a,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,c,:,:,:]
                R₂    = @view    r₂[:,d,:,:,:]
                circshift!(R₁′,   R₁   ,(0,r′[1],r′[2],r′[3])) # shift(R₁,r′)
                circshift!(M⁻¹R₂″,M⁻¹R₂,(0,r″[1],r″[2],r″[3])) # shift(M⁻¹R₂,r″)
                @. G₁ = M⁻¹R₁  * R₁′ # G₁ = [M⁻¹R₁ ⋅R₁′]
                @. G₂ = M⁻¹R₂″ * R₂  # G₂ = [M⁻¹R₂″⋅R₂ ]
                @views for τ in Lₜ
                    @. G₁[τ,:,:,:] *= t′ # G₁ = [t′⋅M⁻¹R₁ ⋅R₁′]
                    @. G₂[τ,:,:,:] *= t″ # G₂ = [t″⋅M⁻¹R₂″⋅R₂ ]
                end
                translational_average!(G₁G₂,G₁,G₂) # G₁⋅G₂ = [t′⋅M⁻¹R₁ ⋅R₁′]⋆[t″⋅M⁻¹R₂″⋅R₂ ]

                # J += 4⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) a⁺(r′+r+i,τ)⟩⋅⟨c(r″+i,0) d⁺(i,0)⟩
                @. crntcrnt += 4*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) a⁺(r′+r+i,τ)⟩⋅⟨d(i,0) c⁺(r″+i,0)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                R₁    = @view    r₁[:,a,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,d,:,:,:]
                R₂    = @view    r₂[:,c,:,:,:]
                circshift!(R₁′,R₁,(0,r′[1],r′[2],r′[3])) # shift(R₁,r′)
                circshift!(R₂″,R₂,(0,r″[1],r″[2],r″[3])) # shift(R₂,r″)
                @. G₁ = M⁻¹R₁ * R₁′ # G₁ = [M⁻¹R₁⋅R₁′]
                @. G₂ = M⁻¹R₂ * R₂″ # G₂ = [M⁻¹R₂⋅R₂″]
                @views for τ in Lₜ
                    @. G₁[τ,:,:,:] *= t′ # G₁ = [t′⋅M⁻¹R₁⋅R₁′]
                    @. G₂[τ,:,:,:] *= t″ # G₂ = [t″⋅M⁻¹R₂⋅R₂″]
                end
                translational_average!(G₁G₂,G₁,G₂) # G₁⋅G₂ = [t′⋅M⁻¹R₁⋅R₁′]⋆[t″⋅M⁻¹R₂⋅R₂″]

                # J -= 4⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) a⁺(r′+r+i,τ)⟩⋅⟨d(i,0) c⁺(r″+i,0)⟩
                @. crntcrnt -= 4*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) b⁺(r+i,τ)⟩⋅⟨c(r″+i,0) d⁺(i,0)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,a,:,:,:]
                R₁    = @view    r₁[:,b,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,c,:,:,:]
                R₂    = @view    r₂[:,d,:,:,:]
                circshift!(M⁻¹R₁′,M⁻¹R₁,(0,r′[1],r′[2],r′[3])) # shift(M⁻¹R₁,r′)
                circshift!(M⁻¹R₂″,M⁻¹R₂,(0,r″[1],r″[2],r″[3])) # shift(M⁻¹R₂,r″)
                @. G₁ = M⁻¹R₁′ * R₁ # G₁ = [M⁻¹R₁′⋅R₁]
                @. G₂ = M⁻¹R₂″ * R₂ # G₂ = [M⁻¹R₂″⋅R₂]
                @views for τ in Lₜ
                    @. G₁[τ,:,:,:] *= t′ # G₁ = [t′⋅M⁻¹R₁′⋅R₁]
                    @. G₂[τ,:,:,:] *= t″ # G₂ = [t″⋅M⁻¹R₂″⋅R₂]
                end
                translational_average!(G₁G₂,G₁,G₂) # G₁⋅G₂ = [t′⋅M⁻¹R₁′⋅R₁]⋆[t″⋅M⁻¹R₂″⋅R₂]

                # J -= 4⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) b⁺(r+i,τ)⟩⋅⟨c(r″+i,0) d⁺(i,0)⟩
                @. crntcrnt -= 4*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) b⁺(r+i,τ)⟩⋅⟨d(i,0) c⁺(r″+i,0)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,a,:,:,:]
                R₁    = @view    r₁[:,b,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,d,:,:,:]
                R₂    = @view    r₂[:,c,:,:,:]
                circshift!(M⁻¹R₁′,M⁻¹R₁,(0,r′[1],r′[2],r′[3])) # shift(M⁻¹R₁,r′)
                circshift!(R₂″,R₂,(0,r″[1],r″[2],r″[3]))       # shift(R₂,r″)
                @. G₁ = M⁻¹R₁′* R₁  # G₁ = [M⁻¹R₁′⋅R₁]
                @. G₂ = M⁻¹R₂ * R₂″ # G₂ = [M⁻¹R₂⋅R₂″]
                @views for τ in Lₜ
                    @. G₁[τ,:,:,:] *= t′ # G₁ = [t′⋅M⁻¹R₁′⋅R₁]
                    @. G₂[τ,:,:,:] *= t″ # G₂ = [t″⋅M⁻¹R₂⋅R₂″]
                end
                translational_average!(G₁G₂,G₁,G₂) # G₁⋅G₂ = [t′⋅M⁻¹R₁′⋅R₁]⋆[t″⋅M⁻¹R₂⋅R₂″]

                # J += 4⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) b⁺(r+i,τ)⟩⋅⟨d(i,0) c⁺(r″+i,0)⟩
                @. crntcrnt -= 4*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) d⁺(i,0)⟩⋅⟨c(r″+i,0) a⁺(r′+r+i,τ)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                R₁    = @view    r₁[:,d,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,c,:,:,:]
                R₂    = @view    r₂[:,a,:,:,:]
                circshift!(M⁻¹R₂″,M⁻¹R₂,(0,r″[1],r″[2],r″[3]))
                circshift!(   R₁′,   R₂,(0,r′[1],r′[2],r′[3]))
                @. G₁ = M⁻¹R₁  * R₁′
                @. G₂ = M⁻¹R₂″ * R₁
                @views for τ in Lₜ
                    @. G₁[τ,:,:,:] *= t′
                    @. G₂[τ,:,:,:] *= t″
                end
                translational_average!(G₁G₂,G₁,G₂)

                # J -= 2⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) d⁺(i,0)⟩⋅⟨c(r″+i,0) a⁺(r′+r+i,τ)⟩
                @. crntcrnt -= 2*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) c⁺(r″+i,0)⟩⋅⟨d(i,0) a⁺(r′+r+i,τ)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                R₁    = @view    r₁[:,c,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,d,:,:,:]
                R₂    = @view    r₂[:,a,:,:,:]
                circshift!(R₁″,R₁,(0,r″[1],r″[2],r″[3]))
                circshift!(R₂′,R₂,(0,r′[1],r′[2],r′[3]))
                @. G₁ = R₁″   * M⁻¹R₂
                @. G₂ = M⁻¹R₁ * R₂′
                @views for τ in Lₜ
                    @. G₁[τ,:,:,:] *= t″
                    @. G₂[τ,:,:,:] *= t′
                end
                translational_average!(G₁G₂,G₁,G₂)

                # J += 2⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) c⁺(r″+i,0)⟩⋅⟨d(i,0) a⁺(r′+r+i,τ)⟩
                @. crntcrnt += 2*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) d⁺(i,0)⟩⋅⟨c(r″+i,0) b⁺(r+i,τ)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,a,:,:,:]
                R₁    = @view    r₁[:,d,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,c,:,:,:]
                R₂    = @view    r₂[:,b,:,:,:]
                circshift!(M⁻¹R₁′,M⁻¹R₁,(0,r′[1],r′[2],r′[3]))
                circshift!(M⁻¹R₂″,M⁻¹R₂,(0,r″[1],r″[2],r″[3]))
                @. G₁ = M⁻¹R₁′ * R₂
                @. G₂ = R₁     * M⁻¹R₂″
                @views for τ in Lₜ
                    @. G₁[τ,:,:,:] *= t′
                    @. G₂[τ,:,:,:] *= t″
                end
                translational_average!(G₁G₂,G₁,G₂)

                # J += 2⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) d⁺(i,0)⟩⋅⟨c(r″+i,0) b⁺(r+i,τ)⟩
                @. crntcrnt += 2*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) c⁺(r″+i,0)⟩⋅⟨d(i,0) b⁺(r+i,τ)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,a,:,:,:]
                R₁    = @view    r₁[:,c,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,d,:,:,:]
                R₂    = @view    r₂[:,b,:,:,:]
                circshift!(M⁻¹R₁′,M⁻¹R₁,(0,r′[1],r′[2],r′[3]))
                circshift!(R₂″,   R₁,   (0,r″[1],r″[2],r″[3]))
                @. G₁ = M⁻¹R₁′ * R₂
                @. G₂ = R₂″    * M⁻¹R₂
                @views for τ in Lₜ
                    @. G₁[τ,:,:,:] *= t′
                    @. G₂[τ,:,:,:] *= t″
                end
                translational_average!(G₁G₂,G₁,G₂)

                # J -= 2⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) c⁺(r″+i,0)⟩⋅⟨d(i,0) b⁺(r+i,τ)⟩
                @. crntcrnt -= 2*G₁G₂

                # CALCULATE 2⋅δ(τ)⋅δ(a,c)⋅δ(r″,r′+r)⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,0) d⁺(i,0)⟩
                if a==c
                    # r = r″-r′
                    l₁ = mod(r″[1]-r′[1],L₁)
                    l₂ = mod(r″[2]-r′[2],L₂)
                    l₃ = mod(r″[3]-r′[3],L₃)
                    # ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,0) d⁺(i,0)⟩
                    M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                    R₁    = @view    r₁[:,d,:,:,:]
                    copyto!(G₁,M⁻¹R₁)
                    copyto!(G₂,R₁)
                    @views for τ in Lₜ
                        @. G₁[τ,:,:,:] *= t′
                        @. G₂[τ,:,:,:] *= t″
                    end
                    circshift!(G₁G₂,G₁,(0,l₁,l₂,l₃))
                    @. G₁G₂ *= G₂
                    crntcrnt[1,l₁+1,l₂+1,l₃+1] += 2 * sum(G₁G₂) / length(G₁G₂)
                end

                # CALCULATE -2⋅δ(τ)⋅δ(a,d)⋅δ(r′+r)⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,0) c⁺(r″+i,0)⟩
                if a==d
                    # r = -r′
                    l₁ = mod(-r′[1],L₁)
                    l₂ = mod(-r′[2],L₂)
                    l₃ = mod(-r′[3],L₃)
                    # t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,0) c⁺(r″+i,0)⟩
                    M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                    R₁    = @view    r₁[:,c,:,:,:]
                    copyto!(G₁,M⁻¹R₁)
                    circshift!(G₂,R₁,(0,r″[1],r″[2],r″[3]))
                    @views for τ in Lₜ
                        @. G₁[τ,:,:,:] *= t′
                        @. G₂[τ,:,:,:] *= t″
                    end
                    circshift!(G₁G₂,G₁,(0,l₁,l₂,l₃))
                    @. G₁G₂ *= G₂
                    crntcrnt[1,l₁+1,l₂+1,l₃+1] -= 2 * sum(G₁G₂) / length(G₁G₂)
                end

                # CALCULATE -2⋅δ(τ)⋅δ(b,c)⋅δ(r-r″)⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,0) d⁺(i,0)⟩
                if b==c
                    # r = r″
                    l₁ = r″[1]
                    l₂ = r″[2]
                    l₃ = r″[3]
                    # ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,0) d⁺(i,0)⟩
                    M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                    R₁    = @view    r₁[:,d,:,:,:]
                    circshift!(G₁,M⁻¹R₁,(0,r′[1],r′[2],r′[3]))
                    copyto!(G₂,R₁)
                    @views for τ in Lₜ
                        @. G₁[τ,:,:,:] *= t′
                        @. G₂[τ,:,:,:] *= t″
                    end
                    circshift!(G₁G₂,G₁,(0,l₁,l₂,l₃))
                    @. G₁G₂ *= G₂
                    crntcrnt[1,l₁+1,l₂+1,l₃+1] -= 2 * sum(G₁G₂) / length(G₁G₂)
                end

                # CALCULATE 2⋅δ(τ)⋅δ(b,d)⋅δ(r)⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,0) c⁺(r″+i)⟩
                if b==d
                    M⁻¹R₁ = @view M⁻¹r₁[:,a,:,:,:]
                    R₁    = @view    r₁[:,c,:,:,:]
                    circshift!(G₁,M⁻¹R₁,(0,r′[1],r′[2],r′[3]))
                    circshift!(G₂,   R₁,(0,r″[1],r″[2],r″[3]))
                    @views for τ in Lₜ
                        @. G₁[τ,:,:,:] *= t′
                        @. G₂[τ,:,:,:] *= t″
                    end
                    @. G₁G₂ = G₁ * G₂
                    crntcrnt[1,1,1,1] += 2 * sum(G₁G₂) / length(G₁G₂)
                end

                # record measurements
                if L₀==1 # if time independent measurement
                    @views @. container[:,:,:,:,n′,n″]    += crntcrnt[1,:,:,:]
                else # if time dependent measurement
                    @views @. container[1:Lₜ,:,:,:,n′,n″] += crntcrnt
                    # deal with τ=β time slice
                    for l₃ in 0:L₃-1
                        for l₂ in 0:L₂-1
                            for l₁ in 0:L₁-1
                                nl₁ = mod(-l₁,L₁)
                                nl₂ = mod(-l₂,L₂)
                                nl₃ = mod(-l₃,L₃)
                                # J[a,b,r′;c,d,r″](β,r) = J[c,d,r″;a,b,r′](0,-r)
                                container[Lₜ+1,l₁+1,l₂+1,l₃+1,n″,n′] += crntcrnt[1,nl₁+1,nl₂+1,nl₃+1]
                            end
                        end
                    end
                end
            end
        end
    end

    return nothing
end

function measure_CurrentCurrent!(container::Array{Complex{T1},6},model::SSHModel{T1,T2,T3},estimator::EstimateGreensFunction{T1}) where {T1,T2,T3}

    # size of lattice
    Lₜ = model.Lτ::Int
    L₁ = model.lattice.L1::Int
    L₂ = model.lattice.L2::Int
    L₃ = model.lattice.L3::Int
    nₒ = model.lattice.unit_cell.norbits::Int
    nᵤ = model.lattice.ncells::Int # number of unit cells in lattice

    @unpack r₁, M⁻¹r₁, r₂, M⁻¹r₂, ab″, ab′, a, b = estimator
    t₀ = model.t′

    @uviews container r₁ M⁻¹r₁ r₂ M⁻¹r₂ ab″ ab′ a b t₀ begin

        r₁    = reshaped(r₁,    (Lₜ,nₒ,L₁,L₂,L₃))
        M⁻¹r₁ = reshaped(M⁻¹r₁, (Lₜ,nₒ,L₁,L₂,L₃))
        r₂    = reshaped(r₂,    (Lₜ,nₒ,L₁,L₂,L₃))
        M⁻¹r₂ = reshaped(M⁻¹r₂, (Lₜ,nₒ,L₁,L₂,L₃))

        # containers for performing calculation
        crntcrnt = reshaped(view(ab″,1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        G₁G₂     = reshaped(view(ab′,1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        G₁       = reshaped(view(a,  1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        G₂       = reshaped(view(b,  1:Lₜ*nᵤ), (Lₜ,L₁,L₂,L₃))
        R₁′      = G₁
        R₁″      = G₁
        M⁻¹R₁′   = G₁
        M⁻¹R₁″   = G₁
        R₂′      = G₂
        R₂″      = G₂
        M⁻¹R₂′   = G₂
        M⁻¹R₂″   = G₂

        # displacement vectors describing each type of bond
        bonds = model.bond_definitions

        # length of axis corresponding to imaginary time axis
        L₀ = size(container,1)

        # number of vectors
        nᵥ = length(bonds)

        # hopping ampltidues
        t = reshaped(t₀,(Lₜ,L₁,L₂,L₃,nᵥ))

        # iterate over first bond
        for n″ in 1:nᵥ

            # vector associated with bond going from orbitals d ⟶ c displaced r″ unit cells
            r″ = bonds[n″].v::Vector{Int} # displacement in unit cells
            d  = bonds[n″].o₁::Int # starting orbital
            c  = bonds[n″].o₂::Int # ending   orbital
            t″ = @view t[:,:,:,:,n″] # hopping amplitudes associated with bond

            # iterate over second bond
            for n′ in 1:nᵥ

                # initialize bond-bond correlation to zero
                fill!(crntcrnt,0.0)

                # vector associated with bond going from orbitals b ⟶ a displaced r′ unit cells
                r′ = bonds[n′].v::Vector{Int} # displacement in unit cells
                b  = bonds[n′].o₁::Int # starting orbital
                a  = bonds[n′].o₂::Int # ending   orbital
                t′ = @view t[:,:,:,:,n′] # hopping amplitudes associated with bond

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) a⁺(r′+r+i,τ)⟩⋅⟨c(r″+i,0) d⁺(i,0)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                R₁    = @view    r₁[:,a,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,c,:,:,:]
                R₂    = @view    r₂[:,d,:,:,:]
                circshift!(R₁′,   R₁   ,(0,r′[1],r′[2],r′[3])) # shift(R₁,r′)
                circshift!(M⁻¹R₂″,M⁻¹R₂,(0,r″[1],r″[2],r″[3])) # shift(M⁻¹R₂,r″)
                @. G₁ = M⁻¹R₁  * R₁′ # G₁ = [M⁻¹R₁ ⋅R₁′]
                @. G₂ = M⁻¹R₂″ * R₂  # G₂ = [M⁻¹R₂″⋅R₂ ]
                @. G₁ *= t′ # G₁ = [t′⋅M⁻¹R₁ ⋅R₁′]
                @. G₂ *= t″ # G₂ = [t″⋅M⁻¹R₂″⋅R₂ ]
                translational_average!(G₁G₂,G₁,G₂) # G₁⋅G₂ = [t′⋅M⁻¹R₁ ⋅R₁′]⋆[t″⋅M⁻¹R₂″⋅R₂ ]

                # J += 4⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) a⁺(r′+r+i,τ)⟩⋅⟨c(r″+i,0) d⁺(i,0)⟩
                @. crntcrnt += 4*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) a⁺(r′+r+i,τ)⟩⋅⟨d(i,0) c⁺(r″+i,0)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                R₁    = @view    r₁[:,a,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,d,:,:,:]
                R₂    = @view    r₂[:,c,:,:,:]
                circshift!(R₁′,R₁,(0,r′[1],r′[2],r′[3])) # shift(R₁,r′)
                circshift!(R₂″,R₂,(0,r″[1],r″[2],r″[3])) # shift(R₂,r″)
                @. G₁ = M⁻¹R₁ * R₁′ # G₁ = [M⁻¹R₁⋅R₁′]
                @. G₂ = M⁻¹R₂ * R₂″ # G₂ = [M⁻¹R₂⋅R₂″]
                @. G₁ *= t′ # G₁ = [t′⋅M⁻¹R₁⋅R₁′]
                @. G₂ *= t″ # G₂ = [t″⋅M⁻¹R₂⋅R₂″]
                translational_average!(G₁G₂,G₁,G₂) # G₁⋅G₂ = [t′⋅M⁻¹R₁⋅R₁′]⋆[t″⋅M⁻¹R₂⋅R₂″]

                # J -= 4⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) a⁺(r′+r+i,τ)⟩⋅⟨d(i,0) c⁺(r″+i,0)⟩
                @. crntcrnt -= 4*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) b⁺(r+i,τ)⟩⋅⟨c(r″+i,0) d⁺(i,0)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,a,:,:,:]
                R₁    = @view    r₁[:,b,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,c,:,:,:]
                R₂    = @view    r₂[:,d,:,:,:]
                circshift!(M⁻¹R₁′,M⁻¹R₁,(0,r′[1],r′[2],r′[3])) # shift(M⁻¹R₁,r′)
                circshift!(M⁻¹R₂″,M⁻¹R₂,(0,r″[1],r″[2],r″[3])) # shift(M⁻¹R₂,r″)
                @. G₁ = M⁻¹R₁′ * R₁ # G₁ = [M⁻¹R₁′⋅R₁]
                @. G₂ = M⁻¹R₂″ * R₂ # G₂ = [M⁻¹R₂″⋅R₂]
                @. G₁ *= t′ # G₁ = [t′⋅M⁻¹R₁′⋅R₁]
                @. G₂ *= t″ # G₂ = [t″⋅M⁻¹R₂″⋅R₂]
                translational_average!(G₁G₂,G₁,G₂) # G₁⋅G₂ = [t′⋅M⁻¹R₁′⋅R₁]⋆[t″⋅M⁻¹R₂″⋅R₂]

                # J -= 4⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) b⁺(r+i,τ)⟩⋅⟨c(r″+i,0) d⁺(i,0)⟩
                @. crntcrnt -= 4*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) b⁺(r+i,τ)⟩⋅⟨d(i,0) c⁺(r″+i,0)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,a,:,:,:]
                R₁    = @view    r₁[:,b,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,d,:,:,:]
                R₂    = @view    r₂[:,c,:,:,:]
                circshift!(M⁻¹R₁′,M⁻¹R₁,(0,r′[1],r′[2],r′[3])) # shift(M⁻¹R₁,r′)
                circshift!(R₂″,R₂,(0,r″[1],r″[2],r″[3]))       # shift(R₂,r″)
                @. G₁ = M⁻¹R₁′* R₁  # G₁ = [M⁻¹R₁′⋅R₁]
                @. G₂ = M⁻¹R₂ * R₂″ # G₂ = [M⁻¹R₂⋅R₂″]
                @. G₁ *= t′ # G₁ = [t′⋅M⁻¹R₁′⋅R₁]
                @. G₂ *= t″ # G₂ = [t″⋅M⁻¹R₂⋅R₂″]
                translational_average!(G₁G₂,G₁,G₂) # G₁⋅G₂ = [t′⋅M⁻¹R₁′⋅R₁]⋆[t″⋅M⁻¹R₂⋅R₂″]

                # J += 4⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) b⁺(r+i,τ)⟩⋅⟨d(i,0) c⁺(r″+i,0)⟩
                @. crntcrnt -= 4*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) d⁺(i,0)⟩⋅⟨c(r″+i,0) a⁺(r′+r+i,τ)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                R₁    = @view    r₁[:,d,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,c,:,:,:]
                R₂    = @view    r₂[:,a,:,:,:]
                circshift!(M⁻¹R₂″,M⁻¹R₂,(0,r″[1],r″[2],r″[3]))
                circshift!(   R₁′,   R₂,(0,r′[1],r′[2],r′[3]))
                @. G₁ = M⁻¹R₁  * R₁′
                @. G₂ = M⁻¹R₂″ * R₁
                @. G₁ *= t′
                @. G₂ *= t″
                translational_average!(G₁G₂,G₁,G₂)

                # J -= 2⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) d⁺(i,0)⟩⋅⟨c(r″+i,0) a⁺(r′+r+i,τ)⟩
                @. crntcrnt -= 2*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) c⁺(r″+i,0)⟩⋅⟨d(i,0) a⁺(r′+r+i,τ)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                R₁    = @view    r₁[:,c,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,d,:,:,:]
                R₂    = @view    r₂[:,a,:,:,:]
                circshift!(R₁″,R₁,(0,r″[1],r″[2],r″[3]))
                circshift!(R₂′,R₂,(0,r′[1],r′[2],r′[3]))
                @. G₁ = R₁″   * M⁻¹R₂
                @. G₂ = M⁻¹R₁ * R₂′
                @. G₁ *= t″
                @. G₂ *= t′
                translational_average!(G₁G₂,G₁,G₂)

                # J += 2⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,τ) c⁺(r″+i,0)⟩⋅⟨d(i,0) a⁺(r′+r+i,τ)⟩
                @. crntcrnt += 2*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) d⁺(i,0)⟩⋅⟨c(r″+i,0) b⁺(r+i,τ)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,a,:,:,:]
                R₁    = @view    r₁[:,d,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,c,:,:,:]
                R₂    = @view    r₂[:,b,:,:,:]
                circshift!(M⁻¹R₁′,M⁻¹R₁,(0,r′[1],r′[2],r′[3]))
                circshift!(M⁻¹R₂″,M⁻¹R₂,(0,r″[1],r″[2],r″[3]))
                @. G₁ = M⁻¹R₁′ * R₂
                @. G₂ = R₁     * M⁻¹R₂″
                @. G₁ *= t′
                @. G₂ *= t″
                translational_average!(G₁G₂,G₁,G₂)

                # J += 2⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) d⁺(i,0)⟩⋅⟨c(r″+i,0) b⁺(r+i,τ)⟩
                @. crntcrnt += 2*G₁G₂

                # CALCULATE ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) c⁺(r″+i,0)⟩⋅⟨d(i,0) b⁺(r+i,τ)⟩
                M⁻¹R₁ = @view M⁻¹r₁[:,a,:,:,:]
                R₁    = @view    r₁[:,c,:,:,:]
                M⁻¹R₂ = @view M⁻¹r₂[:,d,:,:,:]
                R₂    = @view    r₂[:,b,:,:,:]
                circshift!(M⁻¹R₁′,M⁻¹R₁,(0,r′[1],r′[2],r′[3]))
                circshift!(R₂″,   R₁,   (0,r″[1],r″[2],r″[3]))
                @. G₁ = M⁻¹R₁′ * R₂
                @. G₂ = R₂″    * M⁻¹R₂
                @. G₁ *= t′
                @. G₂ *= t″
                translational_average!(G₁G₂,G₁,G₂)

                # J -= 2⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,τ) c⁺(r″+i,0)⟩⋅⟨d(i,0) b⁺(r+i,τ)⟩
                @. crntcrnt -= 2*G₁G₂

                # CALCULATE 2⋅δ(τ)⋅δ(a,c)⋅δ(r″,r′+r)⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,0) d⁺(i,0)⟩
                if a==c
                    # r = r″-r′
                    l₁ = mod(r″[1]-r′[1],L₁)
                    l₂ = mod(r″[2]-r′[2],L₂)
                    l₃ = mod(r″[3]-r′[3],L₃)
                    # ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,0) d⁺(i,0)⟩
                    M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                    R₁    = @view    r₁[:,d,:,:,:]
                    copyto!(G₁,M⁻¹R₁)
                    copyto!(G₂,R₁)
                    @. G₁ *= t′
                    @. G₂ *= t″
                    circshift!(G₁G₂,G₁,(0,l₁,l₂,l₃))
                    @. G₁G₂ *= G₂
                    crntcrnt[1,l₁+1,l₂+1,l₃+1] += 2 * sum(G₁G₂) / length(G₁G₂)
                end

                # CALCULATE -2⋅δ(τ)⋅δ(a,d)⋅δ(r′+r)⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,0) c⁺(r″+i,0)⟩
                if a==d
                    # r = -r′
                    l₁ = mod(-r′[1],L₁)
                    l₂ = mod(-r′[2],L₂)
                    l₃ = mod(-r′[3],L₃)
                    # t′(r+i,τ) t″(i,0)⟩⋅⟨b(r+i,0) c⁺(r″+i,0)⟩
                    M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                    R₁    = @view    r₁[:,c,:,:,:]
                    copyto!(G₁,M⁻¹R₁)
                    circshift!(G₂,R₁,(0,r″[1],r″[2],r″[3]))
                    @. G₁ *= t′
                    @. G₂ *= t″
                    circshift!(G₁G₂,G₁,(0,l₁,l₂,l₃))
                    @. G₁G₂ *= G₂
                    crntcrnt[1,l₁+1,l₂+1,l₃+1] -= 2 * sum(G₁G₂) / length(G₁G₂)
                end

                # CALCULATE -2⋅δ(τ)⋅δ(b,c)⋅δ(r-r″)⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,0) d⁺(i,0)⟩
                if b==c
                    # r = r″
                    l₁ = r″[1]
                    l₂ = r″[2]
                    l₃ = r″[3]
                    # ⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,0) d⁺(i,0)⟩
                    M⁻¹R₁ = @view M⁻¹r₁[:,b,:,:,:]
                    R₁    = @view    r₁[:,d,:,:,:]
                    circshift!(G₁,M⁻¹R₁,(0,r′[1],r′[2],r′[3]))
                    copyto!(G₂,R₁)
                    @. G₁ *= t′
                    @. G₂ *= t″
                    circshift!(G₁G₂,G₁,(0,l₁,l₂,l₃))
                    @. G₁G₂ *= G₂
                    crntcrnt[1,l₁+1,l₂+1,l₃+1] -= 2 * sum(G₁G₂) / length(G₁G₂)
                end

                # CALCULATE 2⋅δ(τ)⋅δ(b,d)⋅δ(r)⋅⟨t′(r+i,τ) t″(i,0)⟩⋅⟨a(r′+r+i,0) c⁺(r″+i)⟩
                if b==d
                    M⁻¹R₁ = @view M⁻¹r₁[:,a,:,:,:]
                    R₁    = @view    r₁[:,c,:,:,:]
                    circshift!(G₁,M⁻¹R₁,(0,r′[1],r′[2],r′[3]))
                    circshift!(G₂,   R₁,(0,r″[1],r″[2],r″[3]))
                    @. G₁ *= t′
                    @. G₂ *= t″
                    @. G₁G₂ = G₁ * G₂
                    crntcrnt[1,1,1,1] += 2 * sum(G₁G₂) / length(G₁G₂)
                end

                # record measurements
                if L₀==1 # if time independent measurement
                    @views @. container[:,:,:,:,n′,n″]    += crntcrnt[1,:,:,:]
                else # if time dependent measurement
                    @views @. container[1:Lₜ,:,:,:,n′,n″] += crntcrnt
                    # deal with τ=β time slice
                    for l₃ in 0:L₃-1
                        for l₂ in 0:L₂-1
                            for l₁ in 0:L₁-1
                                nl₁ = mod(-l₁,L₁)
                                nl₂ = mod(-l₂,L₂)
                                nl₃ = mod(-l₃,L₃)
                                # J[a,b,r′;c,d,r″](β,r) = J[c,d,r″;a,b,r′](0,-r)
                                container[Lₜ+1,l₁+1,l₂+1,l₃+1,n″,n′] += crntcrnt[1,nl₁+1,nl₂+1,nl₃+1]
                            end
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
function measure_PhononGreens!(container::Array{Complex{T1},6},model::SSHModel{T1,T2,T3},Gr::EstimateGreensFunction{T1}) where {T1,T2,T3}

    # size of lattice
    Lₜ = model.Lτ::Int
    L₁ = model.lattice.L1::Int
    L₂ = model.lattice.L2::Int
    L₃ = model.lattice.L3::Int
    nₒ = model.lattice.unit_cell.norbits::Int
    nᵤ = model.lattice.ncells::Int
    nᵥ = model.nph::Int

    # if given lattice dimension only has an extent of 2 unit cells then reduce to 1.
    # need to make things like the two site limit work: in the two site limit there are
    # 2 unit cells but only one phonon.
    if L₁==2
        L₁=1
    end
    if L₂==2
        L₂=1
    end
    if L₃==2
        L₃=1
    end

    # length of axis corresponding to imaginary time axis.
    # L₀ = Lₜ+1 if unequal time measurement
    # L₀ = 1    if   equal time measurement
    L₀ = size(container,1)

    x₀ = model.x
    @unpack a, b, ab′ = Gr

    @uviews a b ab′ x₀ container begin
    
        # phonon fields
        x = reshaped(x₀,(Lₜ,L₁,L₂,L₃,nᵥ))

        # containers
        NL   = Lₜ*L₁*L₂*L₃
        x₁   = reshaped(view(a,  1:NL),(Lₜ,L₁,L₂,L₃))
        x₂   = reshaped(view(b,  1:NL),(Lₜ,L₁,L₂,L₃))
        x₁x₂ = reshaped(view(ab′,1:NL),(Lₜ,L₁,L₂,L₃))

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
                    @views @. container[1,:,:,:,b₁,b₂] += x₁x₂[1,:,:,:]
                else # unequal time measurement
                    @views @. container[1:Lₜ,:,:,:,b₁,b₂] += x₁x₂
                    # dealing with τ=β time slice
                    @views @. container[Lₜ+1,:,:,:,b₁,b₂] += x₁x₂[1,:,:,:]
                end
            end
        end
    end

    return nothing
end

##################################################
## ADDITIONAL FUNCTIONS FOR MAKING MEASUREMENTS ##
##################################################

"""
Measure S-Wave Susceptibility.
"""
function measure_swave!(container::NamedTuple,model::AbstractModel{T1,T2,T3}) where {T1,T2,T3}

    if haskey(container.onsite_meas, :swave_susc)

        # to container S-wave susceptibility measurement
        swave = container.onsite_meas.swave_susc
        
        # number of orbitals per unit cell
        nₒ = model.lattice.unit_cell.norbits::Int
        L₁ = model.lattice.L1
        L₂ = model.lattice.L2
        L₃ = model.lattice.L3
        L  = model.Lτ

        # array contain momentum space pair green's function
        pairs = container.onsite_corr.PairGreens.momentum::Array{Complex{T1},6}

        @uviews pairs begin
            for o in 1:nₒ
                p        = @view pairs[:,1,1,1,o,o]
                swave[o] = simpson(p,model.Δτ)
            end
        end
    end

    return nothing
end

end