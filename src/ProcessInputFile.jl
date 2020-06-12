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
using ..FourierAcceleration: FourierAccelerator, update_Q!, update_M!
using ..SimulationParams: SimulationParameters

using ..KPMPreconditioners: LeftRightKPMPreconditioner

export process_input_file, initialize_holstein_model


function process_input_file(filename::String)
    
    ########################
    ## READ IN INPUT FILE ##
    ########################
    
    input = TOML.parsefile(filename)

    # Input file must describe either a Langevin or a Hyrbid Monte Carlo Simulation but not both.
    @assert haskey(input,"hmc") ⊻ haskey(input,"langevin")

    ##################################
    ## DEFINE SIMULATION PARAMETERS ##
    ##################################

    if haskey(input,"hmc")
        meas_freq = 1
        nsteps    = input["hmc"]["simulation_updates"]
        burnin    = input["hmc"]["burnin_updates"]
    else
        meas_freq = input["langevin"]["meas_freq"]
        nsteps    = input["langevin"]["simulation_timesteps"]
        burnin    = input["langevin"]["burnin_timesteps"]
    end

    # construct simulation parameters object.
    sim_params = SimulationParameters(burnin,
                                      nsteps,
                                      meas_freq,
                                      input["simulation"]["num_bins"],
                                      input["simulation"]["downsample"],
                                      input["simulation"]["filepath"],
                                      input["simulation"]["foldername"])

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
    
    # write current git commit tag of code to log file
    @info( "Commit Hash: "*LibGit2.head(abspath(joinpath(dirname(Base.find_package("Langevin")), ".."))) )
    flush(logio)

    ##############################
    ## CONSTRUCT HOLSTEIN MODEL ##
    ##############################
    
    holstein = initialize_holstein_model(filename)

    # intialize phonon field
    if input["holstein"]["read_phonon_config"]
        phononfile = input["holstein"]["phonon_config_file"]
        read_phonons(holstein, phononfile)
        cp(filename, sim_params.datafolder * phononfile)
    else
        init_phonons_half_filled!(holstein)
    end

    ###########################
    ## DEFINE PRECONDITIONER ##
    ###########################

    # default preconditioner to Identity operator
    preconditioner = I

    if input["solver"]["use_preconditioner"]
        λ_lo = input["simulation"]["lambda_lo"]
        λ_hi = input["simulation"]["lambda_hi"]
        c1   = input["simulation"]["c1"]
        c2   = input["simulation"]["c2"]
        preconditioner = LeftRightKPMPreconditioner(holstein,λ_lo,λ_hi,c1,c2,false)
    end
    
    #################################
    ## DEFINE FOURIER ACCELERATION ##
    #################################
    
    # defining FourierAccelerator type
    fa = FourierAccelerator(holstein)
    
    # set the mass used to construct fourier acceleration matrix
    for d in input["fourier_acceleration"]
        mass = d["mass"]
        if haskey(d,"c")
            c = d["c"]
        else
            c = 0.0
        end
        update_Q!(fa, holstein, d["omega_min"], d["omega_max"], mass)
        update_M!(fa, holstein, d["omega_min"], d["omega_max"], mass, c)
    end

    #####################
    ## DEFINE DYNAMICS ##
    #####################

    # number of degrees of freedom (phonon fields) to simulate
    NL = length(holstein)

    if haskey(input,"hmc")
        Δt              = input["hmc"]["dt"]
        trajectory_time = input["hmc"]["trajectory_time"]
        construct_guess = input["hmc"]["construct_guess"]
        α               = 1.0 - input["hmc"]["momentum_refresh_fraction"]
        @assert 0.0 <= α < 1.0
        dynamics = HybridMonteCarlo(NL,Δt,trajectory_time,α,construct_guess)
    elseif input["langevin"]["update_method"]==1
        Δt       = input["langevin"]["dt"]
        dynamics = EulerDynamics(NL,Δt)
    elseif input["langevin"]["update_method"]==2
        Δt       = input["langevin"]["dt"]
        dynamics = RungeKuttaDynamics(NL,Δt)
    elseif input["langevin"]["update_method"]==3
        Δt       = input["langevin"]["dt"]
        dynamics = HeunsDynamics(NL,Δt)
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
                             tol=input["solver"]["tol"],
                             mul_by_M=input["solver"]["use_preconditioner"],
                             restart=input["solver"]["restart"])
    
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