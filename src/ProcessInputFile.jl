module ProcessInputFile

using TOML
using Random
using Statistics
using LinearAlgebra
using Logging
using LibGit2
using Serialization
using Parameters

using ..UnitCells: UnitCell
using ..Lattices: Lattice
using ..Models: HolsteinModel, SSHModel, AbstractModel
using ..Models: assign_μ!, assign_ω!, assign_λ!, assign_λ₂!, assign_ω₄!
using ..Models: assign_t!, assign_ωᵢⱼ!, assign_hopping!
using ..Models: assign_datafolder!
using ..Models: initialize_model!, update_model!, read_phonons!, mulM!, mulMᵀ!
using ..MuFinder: MuTuner, update_μ!
using ..GreensFunctions: EstimateGreensFunction, update!
using ..InitializePhonons: init_phonons_half_filled!
using ..LangevinDynamics: EulerDynamics, RungeKuttaDynamics, HeunsDynamics
using ..HMC: HybridMonteCarlo
using ..SpecialUpdates: SpecialUpdate, NullUpdate, ReflectionUpdate, SwapUpdate
using ..FourierAcceleration: FourierAccelerator, update_Q!, update_M!
using ..SimulationParams: SimulationParameters
using ..SimulationSummary: initialize_simulation_summary!
using ..Measurements: initialize_measurements_container, initialize_measurement_files!
using ..KPMPreconditioners: LeftRightKPMPreconditioner, SymmetricKPMPreconditioner

export process_input_file, initialize_holstein_model


function process_input_file(filename::String,input::Dict)

    # Input file must describe either a Langevin or HMC Simulation but not both.
    @assert haskey(input,"hmc") ⊻ haskey(input,"langevin")

    # Input file must describe either a Holstein or SSH model but not both.
    @assert haskey(input,"holstein") ⊻ haskey(input,"ssh")

    ##################################
    ## DEFINE SIMULATION PARAMETERS ##
    ##################################

    # initialize simulation parameters
    sim_params = initialize_simulation_params(input)

    # copy input file into data folder
    cp(filename, joinpath(sim_params.datafolder,filename))

    # # write current git commit tag of code to log file
    # @info( "Commit Hash: "*LibGit2.head(abspath(joinpath(dirname(Base.find_package("ElPhDynamics")), ".."))) )
    # logger = global_logger()
    # flush(logger.stream)

    ######################
    ## INITIALIZE MODEL ##
    ######################

    model = initialize_model(input)

    ##############################
    ## INITIALIZE PHONON FIELDS ##
    ##############################

    initialize_phonon_fields!(input,model)

    #####################################
    ## TUNE DENSITY/CHEMICAL POTENTIAL ##
    #####################################

    # initialize MuTuner
    μ_tuner = initialize_mutuner(input,model)

    ###########################
    ## DEFINE PRECONDITIONER ##
    ###########################

    preconditioner = initialize_preconditioner(input,model)
    
    #################################
    ## DEFINE FOURIER ACCELERATION ##
    #################################
    
    # initialize FourierAccelerator
    fa = initialize_fourieraccelerator(input, model)

    #####################
    ## DEFINE DYNAMICS ##
    #####################

    burnin_dynamics, simulation_dynamics = initialize_dynamics(input,model)

    ############################
    ## DEFINE SPECIAL UDPATES ##
    ############################

    burnin_reflect_update, sim_reflect_update = initialize_reflect_update(input,model,burnin_dynamics,simulation_dynamics)
    burnin_swap_update,    sim_swap_update    = initialize_swap_update(input,model,burnin_dynamics,simulation_dynamics)

    #########################
    ## DEFINE MEASUREMENTS ##
    #########################

    # construct object of estimating Green's function
    Gr = EstimateGreensFunction(model,input["measurements"]["num_random_vectors"])

    # construct measurements container
    container = initialize_measurements_container(model,input["measurements"])

    # initialize measurement files
    initialize_measurement_files!(container,sim_params)

    ########################################
    ## INITIALIZE SIMULATION SUMMARY FILE ##
    ########################################

    initialize_simulation_summary!(model,sim_params,input)
    
    burnin_start     = 1
    sim_start        = 1

    #################################
    ## INITIALIZE SIMULATION STATS ##
    #################################

    sim_stats = initialize_sim_stats()

    
    return (model, Gr, μ_tuner, sim_params, simulation_dynamics, burnin_dynamics,
            burnin_reflect_update, sim_reflect_update,
            burnin_swap_update, sim_swap_update,
            fa, preconditioner, container, burnin_start, sim_start, sim_stats)
end

function process_checkpoint(input::Dict)

    ############################
    ## DESERIALIZE CHECKPOINT ##
    ############################

    chkpnt = deserialize(joinpath(input["simulation"]["datafolder"],"checkpoint.jls"))
    
    #######################
    ## UNPACK CHECKPOINT ##
    #######################

    @unpack model, μ_tuner, container, burnin_start, sim_start, sim_stats = chkpnt

    ######################################
    ## INITIALIZE SIMULATION PARAMETERS ##
    ######################################

    sim_params = initialize_simulation_params(input)

    #########################
    ## INITIALIZE DYNAMICS ##
    #########################

    burnin_dynamics, simulation_dynamics = initialize_dynamics(input,model,burnin_start,sim_start)

    ###########################
    ## DEFINE SPECIAL UDPATE ##
    ###########################

    burnin_special_update, sim_reflect_update = initialize_reflect_update(input,model,burnin_dynamics,simulation_dynamics)
    burnin_swap_update,    sim_swap_update    = initialize_swap_update(input,model,burnin_dynamics,simulation_dynamics)

    ###############################
    ## INITIALIZE PRECONDITIONER ##
    ###############################

    preconditioner = initialize_preconditioner(input,model)
    
    ##########################################
    ## INITIALIZE GREENS FUNCTION ESTIMATOR ##
    ##########################################

    Gr = EstimateGreensFunction(model,input["measurements"]["num_random_vectors"])

    ###################################
    ## INITIALIZE FOUIER ACCELERATOR ##
    ###################################

    fa = initialize_fourieraccelerator(input, model)

    return (model, Gr, μ_tuner, sim_params, simulation_dynamics, burnin_dynamics,
            burnin_reflect_update, sim_reflect_update,
            burnin_swap_update, sim_swap_update,
            fa, preconditioner, container, burnin_start, sim_start, sim_stats)
end

###########################################################
## METHODS INITIALIZING SIMULATION THAT ARE NOT EXPORTED ##
###########################################################

"""
Initialize a hamiltonian.
"""
function initialize_model(input::Dict)

    if haskey(input,"holstein") && haskey(input,"ssh")
        error("Config file cannot include both ssh and holstein tables.")
    end

    if !haskey(input,"holstein") && !haskey(input,"ssh")
        error("No valid model table detected in config file.")
    end

    # initialize random number generator
    rng = initialize_rng(input)

    if haskey(input,"holstein")
        model = initialize_holstein_model(input,rng)
    elseif haskey(input,"ssh")
        model = initialize_ssh_model(input,rng)
    else
        error("Neither holstein or ssh model defined.")
    end

    # assign data folder to model
    assign_datafolder!(model,input["simulation"]["datafolder"])

    return model
end

"""
Initialize Holstein Model from config file.
"""
function initialize_holstein_model(input::Dict,rng::AbstractRNG)
    
    # define lattice geometry
    unit_cell = UnitCell(input["lattice"]["ndim"],
                         input["lattice"]["norbits"],
                         hcat(input["lattice"]["lattice_vectors"]...),
                         hcat(input["lattice"]["basis_vectors"]...))
    
    # define lattice
    lattice = Lattice(unit_cell, input["lattice"]["L"])

    # restart for GMRES solver
    if haskey(input["solver"],"restart")
        restart = input["solver"]["restart"]
    else
        restart = 20
    end
    
    # initialize holstein model
    holstein = HolsteinModel(lattice,
                             input["holstein"]["beta"],
                             input["holstein"]["dtau"],
                             is_complex      = false,
                             iterativesolver = input["solver"]["type"],
                             rng             = rng,
                             tol             = input["solver"]["tol"],
                             maxiter         = input["solver"]["maxiter"],
                             restart         = restart)
    
    # adding phonon frequencies
    if haskey(input["holstein"],"omega")
        for d in input["holstein"]["omega"]
            stddev = 0.0
            if haskey(d,"stddev")
                stddev = d["stddev"]
            end
            for orbit in d["orbit"]
                assign_ω!(holstein,d["val"],stddev,orbit)
            end
        end
    end
    
    # adding chemical potential
    if haskey(input["holstein"],"mu")
        for d in input["holstein"]["mu"]
            stddev = 0.0
            if haskey(d,"stddev")
                stddev = d["stddev"]
            end
            for orbit in d["orbit"]
                assign_μ!(holstein,d["val"],stddev,orbit)
            end
        end
    end

    # check in anharmic term defined
    if haskey(input["holstein"],"omega4")
        # adding anharmoic term to holstein model
        for d in input["holstein"]["omega4"]
            stddev = 0.0
            if haskey(d,"stddev")
                stddev = d["stddev"]
            end
            for orbit in d["orbit"]
                assign_ω₄!(holstein,d["val"],stddev,orbit)
            end
        end
    end
    
    # check if any hopping defined
    if haskey(input["holstein"],"t")
        for t in input["holstein"]["t"]
            stddev = 0.0
            if haskey(t,"stddev")
                stddev = t["stddev"]
            end
            assign_t!(holstein, t["val"], stddev, t["orbit"][1], t["orbit"][2], Vector{Int}(t["dL"]))
        end
    end

    # adding electron-phonon coupling
    if haskey(input["holstein"],"lambda")
        for d in input["holstein"]["lambda"]
            stddev = 0.0
            if haskey(d,"stddev")
                stddev = d["stddev"]
            end
            for orbit in d["orbit"]
                assign_λ!(holstein,d["val"],stddev,orbit)
            end
        end
    end

    # adding electron-phonon coupling
    if haskey(input["holstein"],"lambda2")
        for d in input["holstein"]["lambda2"]
            stddev = 0.0
            if haskey(d,"stddev")
                stddev = d["stddev"]
            end
            for orbit in d["orbit"]
                assign_λ₂!(holstein,d["val"],stddev,orbit)
            end
        end
    end

    # initialize model
    initialize_model!(holstein)

    return holstein
end

"""
Initialize SSH model from config file.
"""
function initialize_ssh_model(input::Dict,rng::AbstractRNG)
    
    # define lattice geometry
    unit_cell = UnitCell(input["lattice"]["ndim"],
                         input["lattice"]["norbits"],
                         hcat(input["lattice"]["lattice_vectors"]...),
                         hcat(input["lattice"]["basis_vectors"]...))
    
    # define lattice
    lattice = Lattice(unit_cell, input["lattice"]["L"])

    # restart for GMRES solver
    if haskey(input["solver"],"restart")
        restart = input["solver"]["restart"]
    else
        restart = 20
    end
    
    # initialize holstein model
    ssh = SSHModel(lattice,
                   input["ssh"]["beta"],
                   input["ssh"]["dtau"],
                   is_complex      = false,
                   iterativesolver = input["solver"]["type"],
                   rng             = rng,
                   tol             = input["solver"]["tol"],
                   maxiter         = input["solver"]["maxiter"],
                   restart         = restart)

    # adding chemical potential
    for d in input["ssh"]["mu"]
        stddev = 0.0
        if "stddev" in keys(d)
            stddev = d["stddev"]
        end
        for orbit in d["orbit"]
            assign_μ!(ssh,d["val"],stddev,orbit)
        end
    end

    # add hoppings
    if haskey(input["ssh"],"hopping")
        for d in input["ssh"]["hopping"]
            if haskey(d,"t_avg")
                t = d["t_avg"]
            else
                t = 0.0
            end
            if haskey(d,"t_std")
                σt = d["t_std"]
            else
                σt = 0.0
            end
            if haskey(d,"alpha_avg")
                α = d["alpha_avg"]
            else
                α = 0.0
            end
            if haskey(d,"alpha_std")
                σα = d["alpha_std"]
            else
                σα = 0.0
            end
            if haskey(d,"alpha2_avg")
                α₂ = d["alpha2_avg"]
            else
                α₂ = 0.0
            end
            if haskey(d,"alpha2_std")
                σα₂ = d["alpha2_std"]
            else
                σα₂ = 0.0
            end
            if haskey(d,"omega_avg")
                ω = d["omega_avg"]
            else
                ω = 0.0
            end
            if haskey(d,"omega_std")
                σω = d["omega_std"]
            else
                σω = 0.0
            end
            if haskey(d,"omega4_avg")
                ω₄ = d["omega4_avg"]
            else
                ω₄ = 0.0
            end
            if haskey(d,"omega4_std")
                σω₄ = d["omega4_std"]
            else
                σω₄ = 0.0
            end
            if haskey(d,"name")
                name = d["name"]
            else
                name = ""
            end
            o₁ = d["orbits"][1]
            o₂ = d["orbits"][2]
            dL = zeros(Int,3)
            dL[1:length(d["dL"])] .= d["dL"]
            assign_hopping!(ssh,t,σt,ω,σω,ω₄,σω₄,α,σα,α₂,σα₂,o₁,o₂,dL,name)
        end
    end

    # initialize model
    initialize_model!(ssh)

    return ssh
end

"""
Initialize Phonon Fields.
"""
function initialize_phonon_fields!(input::Dict,model::AbstractModel)

    # determine type of el-ph hamiltonian
    if haskey(input,"holstein")
        model_type = "holstein"
    else
        model_type= "ssh"
    end

    # intialize phonon field
    if !haskey(input[model_type],"read_phonon_config")
        input[model_type]["read_phonon_config"] = false
    end
    if input[model_type]["read_phonon_config"] # read in phonon field
        phononfile = input[model_type]["phonon_config_file"]
        read_phonons!(model, phononfile)
        cp(phononfile, joinpath(input["simulation"]["datafolder"],basename(phononfile)) )
    else # initialize to random phonon field
        init_phonons_half_filled!(model)
    end

    return nothing
end

"""
Initialize Preconditioner.
"""
function initialize_preconditioner(input::Dict,model::AbstractModel)

    # reconstruct preconditioner
    if haskey(input["solver"],"preconditioner")

        if haskey(input["solver"]["preconditioner"],"n")
            n = input["solver"]["preconditioner"]["n"]
        else
            n = 20
        end

        if haskey(input["solver"]["preconditioner"],"buf")
            buf = input["solver"]["preconditioner"]["buf"]
        else
            buf = 0.05
        end

        if haskey(input["solver"]["preconditioner"],"c1")
            c1 = input["solver"]["preconditioner"]["c1"]
        else
            c1 = 1.0
        end

        if haskey(input["solver"]["preconditioner"],"c2")
            c2 = input["solver"]["preconditioner"]["c2"]
        else
            c2 = 1.0
        end

        if lowercase(input["solver"]["type"])=="cg"
            preconditioner = SymmetricKPMPreconditioner(model,n,buf,c1,c2)
        else
            preconditioner = LeftRightKPMPreconditioner(model,n,buf,c1,c2)
        end
    else

        preconditioner = I
    end

    return preconditioner
end

"""
Initialize FourierAccelerator.
"""
function initialize_fourieraccelerator(input::Dict, model::AbstractModel)

    # defining FourierAccelerator type
    fa = FourierAccelerator(model)
    
    # set the mass used to construct fourier acceleration matrix
    for d in input["fourier_acceleration"]
        mass = d["mass"]
        if haskey(d,"c")
            c = d["c"]
        else
            c = 0.0
        end
        update_Q!(fa, model, d["omega_min"], d["omega_max"], mass)
        update_M!(fa, model, d["omega_min"], d["omega_max"], mass, c)
    end

    return fa
end

"""
Initialize SimulationParams.
"""
function initialize_simulation_params(input::Dict)

    if haskey(input,"hmc")
        meas_freq = input["hmc"]["meas_freq"]
        nsteps    = input["hmc"]["simulation_updates"]
        burnin    = input["hmc"]["burnin_updates"]
    else
        @assert input["langevin"]["burnin_timesteps"]%input["langevin"]["meas_freq"]==0
        meas_freq = input["langevin"]["meas_freq"]
        nsteps    = input["langevin"]["simulation_timesteps"]
        burnin    = input["langevin"]["burnin_timesteps"]
    end

    # set default checkpointing frequency in minutes
    if !haskey(input["simulation"],"checkpoint_freq")
        input["simulation"]["checkpoint_freq"]=10
    end

    # construct simulation parameters object.
    sim_params = SimulationParameters(burnin,
                                      nsteps,
                                      meas_freq,
                                      input["simulation"]["num_bins"],
                                      input["simulation"]["checkpoint_freq"],
                                      input["simulation"]["filepath"],
                                      input["simulation"]["foldername"],
                                      input["simulation"]["datafolder"])

    # make direcotory data will be written to
    if !isdir(sim_params.datafolder)
        mkdir(sim_params.datafolder)
    end

    # create log for simulation
    logfilename = joinpath(sim_params.datafolder, "$(sim_params.foldername).log")
    logio       = open(logfilename,"w+")
    logger      = SimpleLogger(logio)
    global_logger(logger)

    return sim_params
end

"""
Initialize Random Number Generator with Seed
"""
function initialize_rng(input::Dict)

    # initialize random number generator with seed
    if !haskey(input["simulation"],"random_seed")
        input["simulation"]["random_seed"] = abs(rand(Int))
    end
    seed = input["simulation"]["random_seed"]
    rng  = MersenneTwister(seed)

    # write current git commit tag of code to log file
    @info("Random Seed: $seed")
    logger = global_logger()
    flush(logger.stream)

    return rng
end

"""
Initialize MuTuner.
"""
function initialize_mutuner(input::Dict,model::AbstractModel)

    # filename for μ_tuner log
    μ_tuner_logfile = joinpath(input["simulation"]["datafolder"],"mu_tuner_log.out")

    if haskey(input,"tune_density")
        targed_density = input["tune_density"]["density"]
        memory         = input["tune_density"]["memory"]
        κ_min          = input["tune_density"]["kappa_min"]
        # whether or not to write the μ_tuner trajectories to a log file
        if haskey(input["tune_density"],"log")
            log = input["tune_density"]["log"]
        else
            log = false
        end
        μ_tuner = MuTuner(true, mean(model.μ), targed_density*model.Nsites, model.Nsites, model.β, model.Δτ, memory, κ_min*model.Nsites, log, μ_tuner_logfile)
    else
        μ_tuner = MuTuner(false, mean(model.μ), 1.0*model.Nsites, model.Nsites, model.β, model.Δτ, 0.75, 0.1, false, μ_tuner_logfile)
    end

    return μ_tuner
end

"""
Initialize Dynamics.
"""
function initialize_dynamics(input::Dict,model::AbstractModel,burnin_start::Int=1,sim_start::Int=1)

    # check to bugs
    if haskey(input,"hmc") && haskey(input,"langevin")
        error("Config file cannot include both hmc and langevin tables.")
    end

    # number of degrees of freedom (phonon fields) to simulate
    NL = length(model)

    if haskey(input,"hmc")

        Δt              = input["hmc"]["dt"]
        tr              = input["hmc"]["trajectory_time"]
        α               = input["hmc"]["momentum_conservation_fraction"]
        Nb              = input["hmc"]["num_multitimesteps"]

        # log file instructions
        if haskey(input["hmc"],"log")
            log = input["hmc"]["log"]
        else
            log = false
        end
        if log && haskey(input["hmc"],"verbose")
            verbose = input["hmc"]["verbose"]
        else
            verbose = false
        end
        hmc_simulation_logfile = joinpath(input["simulation"]["datafolder"],"hmc_sim_log.out")
        hmc_burnin_logfile     = joinpath(input["simulation"]["datafolder"],"hmc_burnin_log.out")

        @assert 0.0 <= α < 1.0
        simulation_dynamics = HybridMonteCarlo(model, Δt, tr, α, Nb, log=log, verbose=verbose,
                                               logfilename=hmc_simulation_logfile,updates=sim_start)

        # defining burnin dynamics
        if haskey(input["hmc"], "burnin")
            if haskey(input["hmc"]["burnin"],"dt")
                Δt = input["hmc"]["burnin"]["dt"]
            end
            if haskey(input["hmc"]["burnin"],"trajectory_time")
                tr = input["hmc"]["burnin"]["trajectory_time"]
            end
            if haskey(input["hmc"]["burnin"],"momentum_conservation_fraction")
                α = input["hmc"]["burnin"]["momentum_conservation_fraction"]
            end
            if haskey(input["hmc"]["burnin"],"num_multitimesteps")
                Nb = input["hmc"]["burnin"]["num_multitimesteps"]
            end
            @assert 0.0 <= α < 1.0
        end
        burnin_dynamics = HybridMonteCarlo(simulation_dynamics, Δt, tr, α, Nb, log=log, verbose=verbose,
                                           logfilename=hmc_burnin_logfile, updates=burnin_start)

    elseif input["langevin"]["update_method"]==1

        Δt = input["langevin"]["dt"]
        simulation_dynamics = EulerDynamics(model,Δt)
        burnin_dynamics = simulation_dynamics

    elseif input["langevin"]["update_method"]==2

        Δt = input["langevin"]["dt"]
        simulation_dynamics = RungeKuttaDynamics(model,Δt)
        burnin_dynamics = simulation_dynamics

    elseif input["langevin"]["update_method"]==3

        Δt = input["langevin"]["dt"]
        simulation_dynamics = HeunsDynamics(model,Δt)
        burnin_dynamics = simulation_dynamics

    end

    return burnin_dynamics, simulation_dynamics
end

"""
Initialize Reflection Update.
"""
function initialize_reflect_update(input::Dict,model::AbstractModel,burnin_dynaimcs,simulation_dynamics)

    # define special update for simulation updates
    if haskey(input,"langevin")
        sim_reflect_update = NullUpdate()
    else
        if haskey(input,"holstein") && haskey(input["hmc"],"reflection_update")
            freq   = input["hmc"]["reflection_update"]["freq"]
            nsites = input["hmc"]["reflection_update"]["nsites"]
            sim_reflect_update = ReflectionUpdate(model,freq,nsites)
        else
            sim_reflect_update = NullUpdate()
        end
    end

    # define special update for burnin updates
    burnin_reflect_update = sim_reflect_update
    if haskey(input,"hmc") && haskey(input,"holstein")
        if haskey(input["hmc"],"burnin")
            if haskey(input["hmc"]["burnin"],"reflection_update")
                freq   = input["hmc"]["reflection_update"]["freq"]
                nsites = input["hmc"]["reflection_update"]["nsites"]
                burnin_reflect_update = ReflectionUpdate(model,freq,nsites)
            end
        end
    end

    return burnin_reflect_update, sim_reflect_update
end

"""
Initialize Swap Update.
"""
function initialize_swap_update(input::Dict,model::AbstractModel,burnin_dynaimcs,simulation_dynamics)

    # define special update for simulation updates
    if haskey(input,"langevin")
        sim_swap_update = NullUpdate()
    else
        if haskey(input,"holstein") && haskey(input["hmc"],"swap_update")
            freq   = input["hmc"]["swap_update"]["freq"]
            nsites = input["hmc"]["swap_update"]["nsites"]
            sim_swap_update = SwapUpdate(model,freq,nsites)
        else
            sim_swap_update = NullUpdate()
        end
    end

    # define special update for burnin updates
    burnin_swap_update = sim_swap_update
    if haskey(input,"hmc") && haskey(input,"holstein")
        if haskey(input["hmc"],"burnin")
            if haskey(input["hmc"]["burnin"],"swap_update")
                freq   = input["hmc"]["swap_update"]["freq"]
                nsites = input["hmc"]["swap_update"]["nsites"]
                burnin_swap_update = SwapUpdate(model,freq,nsites)
            end
        end
    end

    return burnin_swap_update, sim_swap_update
end

"""
Initialize Simulation Stats Dictionary.
"""
function initialize_sim_stats()::Dict

    sim_stats = Dict("simulation_time" => 0.0, "measurement_time" => 0.0, "write_time" => 0.0,
                     "iters" => 0.0, "acceptance_rate" => 0.0, "reflect_acceptance_rate" => 0.0,
                     "swap_acceptance_rate" => 0.0)

    return sim_stats
end

end