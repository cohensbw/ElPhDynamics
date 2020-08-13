module ProcessInputFile

using Pkg.TOML
using Random
using Statistics
using LinearAlgebra
using Logging
using LibGit2

using ..UnitCells: UnitCell
using ..Lattices: Lattice
using ..Models: HolsteinModel
using ..MuFinder: MuTuner, update_μ!
using ..Models: assign_μ!, assign_ω!, assign_λ!, assign_ω4!
using ..Models: assign_tij!, assign_ωij!
using ..Models: setup_checkerboard!, construct_expnΔτV!, read_phonons
using ..GreensFunctions: EstimateGreensFunction, update!
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

    #####################################
    ## TUNE DENSITY/CHEMICAL POTENTIAL ##
    #####################################

    if haskey(input,"tune_density")
        targed_density = input["tune_density"]["density"]
        memory         = input["tune_density"]["memory"]
        buffer         = input["tune_density"]["buffer"]
        κ₀             = input["tune_density"]["init_compresibility"]
        μ_tuner = MuTuner(true, mean(holstein.μ), targed_density*holstein.nsites, holstein.β, holstein.Δτ, κ₀, memory, buffer)
    else
        μ_tuner = MuTuner(false, mean(holstein.μ), 1.0*holstein.nsites, holstein.β, holstein.Δτ, 1.0, 0.75, 10)
    end

    ###########################
    ## DEFINE PRECONDITIONER ##
    ###########################

    if lowercase(input["solver"]["type"])=="cg"
        preconditioner = I
    else
        λ_lo = input["solver"]["preconditioner"]["lambda_lo"]
        λ_hi = input["solver"]["preconditioner"]["lambda_hi"]
        c1   = input["solver"]["preconditioner"]["c1"]
        c2   = input["solver"]["preconditioner"]["c2"]
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
        tr              = input["hmc"]["trajectory_time"]
        construct_guess = input["hmc"]["construct_guess"]
        α               = input["hmc"]["momentum_conservation_fraction"]
        Nb              = input["hmc"]["num_multitimesteps"]
        if haskey(input["hmc"],"tau")
            τ = input["hmc"]["tau"]
        else
            # default value is infinity, recovering normal hamiltonian dynamics
            τ = Inf
        end
        # if τ is meant to be infinity, recovering normal hamiltonian dynamics
        if typeof(τ)==String
            if startswith(lowercase(τ),"inf")
                τ = Inf
            else
                throw(DomainError(τ,"invalid value for tau"))
            end
        end
        @assert τ >= 0.0
        @assert 0.0 <= α < 1.0
        @assert !((α>0)&(isfinite(τ)))
        simulation_dynamics = HybridMonteCarlo(NL,Δt,tr,τ,α,Nb,construct_guess)

        if haskey(input["hmc"], "burnin")
            if haskey(input["hmc"]["burnin"],"dt")
                Δt = input["hmc"]["burnin"]["dt"]
            end
            if haskey(input["hmc"]["burnin"],"trajectory_time")
                tr = input["hmc"]["burnin"]["trajectory_time"]
            end
            if haskey(input["hmc"]["burnin"],"tau")
                τ = input["hmc"]["burnin"]["tau"]
                # if τ is meant to be infinity, recovering normal hamiltonian dynamics
                if typeof(τ)==String
                    if startswith(lowercase(τ),"inf")
                        τ = Inf
                    else
                        throw(DomainError(τ,"invalid value for tau"))
                    end
                end
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
            @assert τ >= 0.0
            @assert 0.0 <= α < 1.0
            @assert !((α>0)&(isfinite(τ)))
        end
        burnin_dyanmics = HybridMonteCarlo(simulation_dynamics,Δt=Δt,tr=tr,τ=τ,α=α,Nb=Nb,construct_guess=construct_guess)

    elseif input["langevin"]["update_method"]==1

        Δt       = input["langevin"]["dt"]
        simulation_dynamics = EulerDynamics(NL,Δt)
        burnin_dyanmics = simulation_dynamics

    elseif input["langevin"]["update_method"]==2

        Δt       = input["langevin"]["dt"]
        simulation_dynamics = RungeKuttaDynamics(NL,Δt)
        burnin_dyanmics = simulation_dynamics

    elseif input["langevin"]["update_method"]==3

        Δt       = input["langevin"]["dt"]
        simulation_dynamics = HeunsDynamics(NL,Δt)
        burnin_dyanmics = simulation_dynamics

    end

    #########################
    ## DEFINE MEASUREMENTS ##
    #########################

    # specify which measurements to make
    measurements     = input["measurements"]
    unequaltime_meas = Vector{String}()
    equaltime_meas   = Vector{String}()
    for k in keys(measurements)
        if k != "num_random_vectors"
            if measurements[k]["measure"]
                if measurements[k]["time_dependent"]
                    push!(unequaltime_meas,k)
                else
                    push!(equaltime_meas,k)
                end
            end
        end
    end

    # construct object of estimating Green's function
    if haskey(input["measurements"],"num_random_vectors")
        num_random_vectors = input["measurements"]["num_random_vectors"]
    else
        num_random_vectors = 1
    end
    Gr = EstimateGreensFunction(holstein,num_random_vectors)
    
    return holstein, Gr, μ_tuner, sim_params, simulation_dynamics, burnin_dyanmics, fa, preconditioner, unequaltime_meas, equaltime_meas, input
end


function initialize_holstein_model(filename::String)

    # read input file
    input = TOML.parsefile(filename)
    
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
                        tij["orbit"][1], tij["orbit"][2], Vector{Int}(tij["dL"]))
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