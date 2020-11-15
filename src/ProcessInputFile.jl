module ProcessInputFile

using Pkg.TOML
using Random
using Statistics
using LinearAlgebra
using Logging
using LibGit2

using ..UnitCells: UnitCell
using ..Lattices: Lattice
using ..Models: HolsteinModel, SSHModel
using ..MuFinder: MuTuner, update_μ!
using ..Models: assign_μ!, assign_ω!, assign_λ!, assign_λ₂!, assign_ω₄!
using ..Models: assign_t!, assign_ωᵢⱼ!, assign_hopping!
using ..Models: initialize_model!, update_model!, read_phonons!, mulM!, mulMᵀ!
using ..GreensFunctions: EstimateGreensFunction, update!
using ..InitializePhonons: init_phonons_half_filled!
using ..LangevinDynamics: EulerDynamics, RungeKuttaDynamics, HeunsDynamics
using ..HMC: HybridMonteCarlo
using ..FourierAcceleration: FourierAccelerator, update_Q!, update_M!
using ..SimulationParams: SimulationParameters
using ..SimulationSummary: initialize_simulation_summary!

using ..KPMPreconditioners: LeftRightKPMPreconditioner, SymmetricKPMPreconditioner

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
        meas_freq = input["hmc"]["meas_freq"]
        nsteps    = input["hmc"]["simulation_updates"]
        burnin    = input["hmc"]["burnin_updates"]
    else
        @assert input["langevin"]["burnin_timesteps"]%input["langevin"]["meas_freq"]==0
        meas_freq = input["langevin"]["meas_freq"]
        nsteps    = input["langevin"]["simulation_timesteps"]
        burnin    = input["langevin"]["burnin_timesteps"]
    end

    # construct simulation parameters object.
    sim_params = SimulationParameters(burnin,
                                      nsteps,
                                      meas_freq,
                                      input["simulation"]["num_bins"],
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
    cp(filename, joinpath(sim_params.datafolder,filename))

    # create log for simulation
    logfilename = joinpath(sim_params.datafolder, sim_params.foldername*".log")
    logio       = open(logfilename,"w+")
    logger      = SimpleLogger(logio)
    global_logger(logger)
    
    # write current git commit tag of code to log file
    @info( "Commit Hash: "*LibGit2.head(abspath(joinpath(dirname(Base.find_package("Langevin")), ".."))) )
    flush(logio)

    ######################
    ## INITIALIZE MODEL ##
    ######################

    # initialize model
    model = initialize_model(filename)
    
    # if hosltein model
    if haskey(input,"holstein")

        if !haskey(input["holstein"],"read_phonon_config")
            input["holstein"]["read_phonon_config"] = false
        end

        # intialize phonon field
        if input["holstein"]["read_phonon_config"] # read in phonon field
            phononfile = input["holstein"]["phonon_config_file"]
            read_phonons!(model, phononfile)
            cp(filename, joinpath(sim_params.datafolder,phononfile))
        else # initialize to random phonon field
            init_phonons_half_filled!(model)
        end
    
    # if ssh model
    elseif haskey(input,"ssh")

        if !haskey(input["ssh"],"read_phonon_config")
            input["ssh"]["read_phonon_config"] = false
        end

        # intialize phonon field
        if input["ssh"]["read_phonon_config"] # read in phonon field
            phononfile = input["ssh"]["phonon_config_file"]
            read_phonons!(model, phononfile)
            cp(filename, joinpath(sim_params.datafolder,phononfile))
        else # initialize to random phonon field
            init_phonons_half_filled!(model)
        end
    end

    #####################################
    ## TUNE DENSITY/CHEMICAL POTENTIAL ##
    #####################################

    # filename for μ_tuner log
    μ_tuner_logfile = joinpath(sim_params.datafolder,"mu_tuner_log.out")

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

    ###########################
    ## DEFINE PRECONDITIONER ##
    ###########################

    if lowercase(input["solver"]["type"])=="cg"
        if haskey(input["solver"],"preconditioner")
            λ_lo = input["solver"]["preconditioner"]["lambda_lo"]
            λ_hi = input["solver"]["preconditioner"]["lambda_hi"]
            c1   = input["solver"]["preconditioner"]["c1"]
            c2   = input["solver"]["preconditioner"]["c2"]
            preconditioner = SymmetricKPMPreconditioner(model,λ_lo,λ_hi,c1,c2,false)
        else
            preconditioner = I
        end
    else
        λ_lo = input["solver"]["preconditioner"]["lambda_lo"]
        λ_hi = input["solver"]["preconditioner"]["lambda_hi"]
        c1   = input["solver"]["preconditioner"]["c1"]
        c2   = input["solver"]["preconditioner"]["c2"]
        preconditioner = LeftRightKPMPreconditioner(model,λ_lo,λ_hi,c1,c2,false)
    end
    
    #################################
    ## DEFINE FOURIER ACCELERATION ##
    #################################
    
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

    #####################
    ## DEFINE DYNAMICS ##
    #####################

    # check to bugs
    if haskey(input,"hmc") && haskey(input,"langevin")
        error("Config file cannot include both hmc and langevin tables.")
    end

    # number of degrees of freedom (phonon fields) to simulate
    NL = length(model)

    if haskey(input,"hmc")

        Δt              = input["hmc"]["dt"]
        tr              = input["hmc"]["trajectory_time"]
        construct_guess = input["hmc"]["construct_guess"]
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
        hmc_simulation_logfile = joinpath(sim_params.datafolder,"hmc_sim_log.out")
        hmc_burnin_logfile     = joinpath(sim_params.datafolder,"hmc_burnin_log.out")

        @assert 0.0 <= α < 1.0
        simulation_dynamics = HybridMonteCarlo(model, Δt, tr, α, Nb, construct_guess, log=log, verbose=verbose,
                                               logfilename=hmc_simulation_logfile)

        # defining burnin dynamics
        if haskey(input["hmc"], "burnin")
            if haskey(input["hmc"]["burnin"],"dt")
                Δt = input["hmc"]["burnin"]["dt"]
            end
            if haskey(input["hmc"]["burnin"],"trajectory_time")
                tr = input["hmc"]["burnin"]["trajectory_time"]
            end
            if haskey(input["hmc"]["burnin"],"construct_guess")
                construct_guess = input["hmc"]["burnin"]["construct_guess"]
            end
            if haskey(input["hmc"]["burnin"],"momentum_conservation_fraction")
                α = input["hmc"]["burnin"]["momentum_conservation_fraction"]
            end
            if haskey(input["hmc"]["burnin"],"num_multitimesteps")
                Nb = input["hmc"]["burnin"]["num_multitimesteps"]
            end
            @assert 0.0 <= α < 1.0
        end
        burnin_dyanmics = HybridMonteCarlo(simulation_dynamics, Δt, tr, α, Nb, construct_guess,
                                           log=log, verbose=verbose, logfilename=hmc_burnin_logfile)

    elseif input["langevin"]["update_method"]==1

        Δt = input["langevin"]["dt"]
        simulation_dynamics = EulerDynamics(model,Δt)
        burnin_dyanmics = simulation_dynamics

    elseif input["langevin"]["update_method"]==2

        Δt = input["langevin"]["dt"]
        simulation_dynamics = RungeKuttaDynamics(model,Δt)
        burnin_dyanmics = simulation_dynamics

    elseif input["langevin"]["update_method"]==3

        Δt = input["langevin"]["dt"]
        simulation_dynamics = HeunsDynamics(model,Δt)
        burnin_dyanmics = simulation_dynamics

    end

    #########################
    ## DEFINE MEASUREMENTS ##
    #########################

    # construct object of estimating Green's function
    Gr = EstimateGreensFunction(model)

    ########################################
    ## INITIALIZE SIMULATION SUMMARY FILE ##
    ########################################

    initialize_simulation_summary!(model,sim_params,input)
    
    return model, Gr, μ_tuner, sim_params, simulation_dynamics, burnin_dyanmics, fa, preconditioner, input
end

#####################################
## METHODS FOR INITIALIZING MODELS ##
## BASED ON THE CONFIGURATION FILE ##
#####################################

"""
Initialize a hamiltonian.
"""
function initialize_model(filename::String)

    # read input file
    input = TOML.parsefile(filename)

    if haskey(input,"holstein") && haskey(input,"ssh")
        error("Config file cannot include both ssh and holstein tables.")
    end

    if haskey(input,"holstein")
        return initialize_holstein_model(input)
    elseif haskey(input,"ssh")
        return initialize_ssh_model(input)
    else
        error("Neither holstein or ssh model defined.")
    end
end

"""
Initialize Holstein Model from config file.
"""
function initialize_holstein_model(input::Dict)
    
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
function initialize_ssh_model(input::Dict)
    
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

end