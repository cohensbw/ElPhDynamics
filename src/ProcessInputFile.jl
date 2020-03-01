module ProcessInputFile

using Pkg.TOML
using Random
using LinearAlgebra
using IterativeSolvers

using ..Geometries: Geometry
using ..Lattices: Lattice
using ..HolsteinModels: HolsteinModel
using ..HolsteinModels: assign_μ!, assign_ω!, assign_λ!, assign_ω4!
using ..HolsteinModels: assign_tij!, assign_ωij!
using ..HolsteinModels: setup_checkerboard!, construct_expnΔτV!, read_phonons
using ..InitializePhonons: init_phonons_half_filled!
using ..FourierAcceleration: FourierAccelerator, update_Q!
using ..LangevinSimulationParameters: SimulationParameters
using ..BlockPreconditioners: BlockPreconditioner, solve!, setup!

export process_input_file

function process_input_file(filename::String)
    
    ########################
    ## READ IN INPUT FILE ##
    ########################
    
    input = TOML.parsefile(filename)

    # checking correct version for input file is specified
    @assert input["input_format_version"] == "0.1.0"

    ##################################
    ## DEFINE SIMULATION PARAMETERS ##
    ##################################

    # construct simulation parameters object.
    sim_params = SimulationParameters(input["simulation"]["dt"],
                                      input["simulation"]["euler"],
                                      input["simulation"]["burnin"],
                                      input["simulation"]["nsteps"],
                                      input["simulation"]["meas_freq"],
                                      input["simulation"]["num_bins"],
                                      input["simulation"]["downsample"],
                                      input["simulation"]["filepath"],
                                      input["simulation"]["foldername"])

    # initialize random number generator with seed
    Random.seed!(input["simulation"]["random_seed"])

    ##############################
    ## CONSTRUCT HOLSTEIN MODEL ##
    ##############################
    
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
                             use_gmres=input["simulation"]["use_block_preconditioner"],
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

    # intialize phonon field
    if input["holstein"]["read_phonon_config"]
        read_phonons(holstein, input["holstein"]["phonon_config_file"])
    else
        init_phonons_half_filled!(holstein)
    end

    # construct exponentiated interaction matrix
    construct_expnΔτV!(holstein)

    ###########################
    ## DEFINE PRECONDITIONER ##
    ###########################

    # default Identity preconditioner
    preconditioner = Identity()

    if input["simulation"]["use_block_preconditioner"]
        preconditioner = BlockPreconditioner(holstein,
                                             tol=input["simulation"]["block_tol"],
                                             restart=input["simulation"]["block_restart"])
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
    
    return holstein, sim_params, fa, preconditioner, input
end

end