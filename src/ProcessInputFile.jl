module ProcessInputFile

using Pkg.TOML
using Random
using LinearAlgebra
using Logging
using LibGit2

using ..Geometries: Geometry
using ..Lattices: Lattice
using ..HolsteinModels: HolsteinModel
using ..HolsteinModels: assign_μ!, assign_ω!, assign_λ!, assign_ω4!
using ..HolsteinModels: assign_tij!, assign_ωij!
using ..HolsteinModels: setup_checkerboard!, construct_expnΔτV!, read_phonons
using ..InitializePhonons: init_phonons_half_filled!
using ..LangevinDynamics: EulerDynamics, RungeKuttaDynamics, HeunsDynamics
using ..HMC: HybridMonteCarlo
using ..FourierAcceleration: FourierAccelerator, update_Q!
using ..SimulationParams: SimulationParameters

using ..KPMPreconditioners: LeftKPMPreconditioner, LeftRightKPMPreconditioner

export process_input_file, initialize_holstein_model


function process_input_file(filename::String)
    
    ########################
    ## READ IN INPUT FILE ##
    ########################
    
    input = TOML.parsefile(filename)

    ##################################
    ## DEFINE SIMULATION PARAMETERS ##
    ##################################

    # default to langevin simulation
    if !haskey(input["simulation"],"is_hybrid_monte_carlo")
        input["simulation"] = false
    end

    # For Hybrid Monte Carlo simulation
    if input["simulation"]["is_hybrid_monte_carlo"]

        burnin_time     = input["simulation"]["burnin_time"]
        simulation_time = input["simulation"]["simulation_time"]
        refresh_time    = input["simulation"]["refresh_time"]

        burnin = round(Int,burnin_time/refresh_time)
        nsteps = round(Int,simulation_time/refresh_time)
        meas_freq = 1

        # construct simulation parameters object.
        sim_params = SimulationParameters(input["simulation"]["dt"],
                                          burnin,
                                          nsteps,
                                          meas_freq,
                                          input["simulation"]["num_bins"],
                                          input["simulation"]["downsample"],
                                          input["simulation"]["filepath"],
                                          input["simulation"]["foldername"])

    # For Langevin simulation
    else

        # construct simulation parameters object.
        sim_params = SimulationParameters(input["simulation"]["dt"],
                                          input["simulation"]["burnin"],
                                          input["simulation"]["nsteps"],
                                          input["simulation"]["meas_freq"],
                                          input["simulation"]["num_bins"],
                                          input["simulation"]["downsample"],
                                          input["simulation"]["filepath"],
                                          input["simulation"]["foldername"])
    end

    # make direcotory data will be written to
    mkdir(sim_params.datafolder)

    # initialize random number generator with seed
    if !("random_seed" in keys(input["simulation"]))
        input["simulation"]["random_seed"] = abs(rand(Int))
    end
    Random.seed!(input["simulation"]["random_seed"])

    # copy input file into data folder
    cp(filename,sim_params.datafolder*filename)

    # create log for simulation
    logfilename = sim_params.datafolder*sim_params.foldername[1:end-1]*".log"
    logio       = open(logfilename,"w+")
    logger      = SimpleLogger(logio)
    global_logger(logger)
    
    # write git commit of code to log file
    @info( "Commit Hash: "*LibGit2.head(abspath(joinpath(dirname(Base.find_package("Langevin")), ".."))) )
    flush(logio)

    ##############################
    ## CONSTRUCT HOLSTEIN MODEL ##
    ##############################
    
    holstein = initialize_holstein_model(filename)

    # intialize phonon field
    if input["holstein"]["read_phonon_config"]
        read_phonons(holstein, input["holstein"]["phonon_config_file"])
    else
        init_phonons_half_filled!(holstein)
    end

    ###########################
    ## DEFINE PRECONDITIONER ##
    ###########################

    # default Identity preconditioner
    preconditioner = I

    if input["simulation"]["use_preconditioner"] && input["simulation"]["is_hybrid_monte_carlo"]
        λ_lo = input["simulation"]["lambda_lo"]
        λ_hi = input["simulation"]["lambda_hi"]
        c1   = input["simulation"]["c1"]
        c2   = input["simulation"]["c2"]
        preconditioner = LeftRightKPMPreconditioner(holstein,λ_lo,λ_hi,c1,c2,false)
    elseif input["simulation"]["use_preconditioner"]
        λ_lo = input["simulation"]["lambda_lo"]
        λ_hi = input["simulation"]["lambda_hi"]
        c1   = input["simulation"]["c1"]
        c2   = input["simulation"]["c2"]
        preconditioner = LeftKPMPreconditioner(holstein,λ_lo,λ_hi,c1,c2,false)
    end
    
    #################################
    ## DEFINE FOURIER ACCELERATION ##
    #################################
    
    # defining FourierAccelerator type
    fa = FourierAccelerator(holstein, 0.5, input["simulation"]["dt"])
    
    # set the mass used to construct fourier acceleration matrix
    for d in input["fourier_acceleration"]
        update_Q!(fa, holstein, d["mass"], input["simulation"]["dt"], d["omega_min"], d["omega_max"])
    end

    ################################
    ## DEFINE DYNAMICS TO BE USED ##
    ################################

    Δt = input["simulation"]["dt"]
    NL = length(holstein)

    dynamics = nothing
    if input["simulation"]["is_hybrid_monte_carlo"]
        dynamics = HybridMonteCarlo(NL,Δt,refresh_time)
    elseif input["simulation"]["update_method"]==1
        dynamics = EulerDynamics(NL,Δt)
    elseif input["simulation"]["update_method"]==2
        dynamics = RungeKuttaDynamics(NL,Δt)
    elseif input["simulation"]["update_method"]==3
        dynamics = HeunsDynamics(NL,Δt)
    else
        error("Did not specify a valid dynamics option.")
    end

    #########################
    ## DEFINE MEASUREMENTS ##
    #########################

    # specify which measurements to make
    measurements     = input["measurements"]
    unequaltime_meas = Vector{String}()
    equaltime_meas   = Vector{String}()
    for k in keys(measurements)
        if measurements[k]["measure"]
            if measurements[k]["time_dependent"]
                push!(unequaltime_meas,k)
            else
                push!(equaltime_meas,k)
            end
        end
    end
    
    return holstein, sim_params, dynamics, fa, preconditioner, unequaltime_meas, equaltime_meas, input
end


function initialize_holstein_model(filename::String)

    # read input file
    input = TOML.parsefile(filename)
    
    # define lattice geometry
    geom = Geometry(input["lattice"]["ndim"],
                    input["lattice"]["norbits"],
                    hcat(input["lattice"]["lattice_vectors"]...),
                    hcat(input["lattice"]["basis_vectors"]...))
    
    # define lattice
    lattice = Lattice(geom, input["lattice"]["L"])
    
    # initialize holstein model
    holstein = HolsteinModel(geom,lattice,
                             input["holstein"]["beta"],
                             input["holstein"]["dtau"],
                             is_complex=false,
                             tol=input["simulation"]["tol"],
                             mul_by_M=input["simulation"]["use_preconditioner"],
                             restart=input["simulation"]["restart"])
    
    # adding phonon frequencies
    for d in input["holstein"]["omega"]
        stddev = 0.0
        if "stddev" in keys(d)
            stddev = d["stddev"]
        end
        for orbit in d["orbit"]
            assign_ω!(holstein,d["val"],stddev,orbit)
        end
    end
    
    # adding chemical potential
    for d in input["holstein"]["mu"]
        stddev = 0.0
        if "stddev" in keys(d)
            stddev = d["stddev"]
        end
        for orbit in d["orbit"]
            assign_μ!(holstein,d["val"],stddev,orbit)
        end
    end

    # check in anharmic term defined
    if "omega4" in keys(input["holstein"])
        # adding anharmoic term to holstein model
        for d in input["holstein"]["omega4"]
            stddev = 0.0
            if "stddev" in keys(d)
                stddev = d["stddev"]
            end
            for orbit in d["orbit"]
                assign_ω4!(holstein,d["val"],stddev,orbit)
            end
        end
    end
    
    # check if any hopping defined
    if "t" in keys(input["holstein"])
        for tij in input["holstein"]["t"]
            stddev = 0.0
            if "stddev" in keys(tij)
                stddev = tij["stddev"]
            end
            assign_tij!(holstein, tij["val"], stddev,
                        tij["orbit"][1], tij["orbit"][2], tij["dL"])
        end
    end

    # organize electron hoppings for checkerboard decomposition
    setup_checkerboard!(holstein)

    # adding electron-phonon coupling
    for d in input["holstein"]["lambda"]
        stddev = 0.0
        if "stddev" in keys(d)
            stddev = d["stddev"]
        end
        for orbit in d["orbit"]
            assign_λ!(holstein,d["val"],stddev,orbit)
        end
    end

    # construct exponentiated interaction matrix
    construct_expnΔτV!(holstein)

    return holstein
end

end