module ProcessInputFile

using Pkg.TOML

using ..Geometries: Geometry
using ..Lattices: Lattice
using ..HolsteinModels: HolsteinModel
using ..HolsteinModels: assign_μ!, assign_ω!, assign_λ!
using ..HolsteinModels: assign_tij!, assign_ωij!
using ..HolsteinModels: setup_checkerboard!, construct_expnΔτV!
using ..InitializePhonons: init_phonons_single_site!
using ..FourierAcceleration: FourierAccelerator, update_Q!
using ..LangevinSimulationParameters: SimulationParameters

export process_input_file

function process_input_file(filename::String)
    
    ########################
    ## READ IN INPUT FILE ##
    ########################
    
    input = TOML.parsefile(filename)

    # checking correct version for input file is specified
    @assert input["input_format_version"] == "0.1.0"
    
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
                             input["holstein"]["dtau"])
    
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

    # intialize phonon field
    init_phonons_single_site!(holstein)

    # construct exponentiated interaction matrix
    construct_expnΔτV!(holstein)
    
    #################################
    ## DEFINE FOURIER ACCELERATION ##
    #################################
    
    # defining FourierAccelerator type
    fa = FourierAccelerator(holstein, 0.5, input["simulation"]["dt"])
    
    # set the mass used to construct fourier acceleration matrix
    for d in input["fourier_acceleration"]
        update_Q!(fa, holstein, d["mass"], input["simulation"]["dt"], d["omega_min"], d["omega_max"])
    end
    
    ##################################
    ## DEFINE SIMULATION PARAMETERS ##
    ##################################

    # define initial annealing temperature
    annealing_init_temp = 1.0
    if "annealing_init_temp" in keys(input["simulation"])
        annealing_init_temp = input["simulation"]["annealing_init_temp"]
    end
    @assert annealing_init_temp >= 1.0

    # define annealing exponent
    annealing_exponent = 1.0
    if "annealing_exponent" in keys(input["simulation"])
        annealing_exponent = input["simulation"]["annealing_exponent"]
    end

    # construct simulation parameters object.
    sim_params = SimulationParameters(input["simulation"]["dt"],
                                      input["simulation"]["euler"],
                                      input["simulation"]["tol"],
                                      input["simulation"]["burnin"],
                                      input["simulation"]["nsteps"],
                                      input["simulation"]["meas_freq"],
                                      input["simulation"]["num_bins"],
                                      input["simulation"]["filepath"],
                                      input["simulation"]["foldername"],
                                      annealing_init_temp,
                                      annealing_exponent)
    
    
    return holstein, sim_params, fa, input
end

end