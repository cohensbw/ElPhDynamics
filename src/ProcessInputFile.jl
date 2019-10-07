module ProcessInputFile

using TOML

using ..Geometries: Geometry
using ..Lattices: Lattice
using ..HolsteinModels: HolsteinModel
using ..HolsteinModels: assign_μ!, assign_ω!, assign_λ!
using ..HolsteinModels: assign_tij!, assign_ωij!
using ..HolsteinModels: setup_checkerboard!
using ..InitializePhonons: init_phonons_single_site!
using ..FourierAcceleration: FourierAccelerator, update_Q!
using ..LangevinSimulationParameters: SimulationParameters

export process_input_file

function process_input_file(filename::String)
    
    ########################
    ## READ IN INPUT FILE ##
    ########################
    
    input = TOML.parsefile(filename)
    
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
        for orbit in d["orbit"]
            assign_ω!(holstein,d["mean"],d["std"],orbit)
        end
    end
    
    # adding electron-phonon coupling
    for d in input["holstein"]["lambda"]
        for orbit in d["orbit"]
            assign_λ!(holstein,d["mean"],d["std"],orbit)
        end
    end
    
    # adding chemical potential
    for d in input["holstein"]["mu"]
        for orbit in d["orbit"]
            assign_μ!(holstein,d["mean"],d["std"],orbit)
        end
    end
    
    # check if any hopping defined
    if "t" in keys(input["holstein"])
        # adding electron hopping
        for tij in input["holstein"]["t"]
            assign_tij!(holstein, tij["mean"], tij["std"],
                        tij["orbit"][1], tij["orbit"][2], tij["dL"])
        end
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
    
    ##################################
    ## DEFINE SIMULATION PARAMETERS ##
    ##################################
    
    # construct simulation parameters object.
    sim_params = SimulationParameters(input["simulation"]["dt"],
                                      input["simulation"]["euler"],
                                      input["simulation"]["tol"],
                                      input["simulation"]["burnin"],
                                      input["simulation"]["nsteps"],
                                      input["simulation"]["meas_freq"],
                                      input["simulation"]["num_bins"],
                                      input["simulation"]["filepath"],
                                      input["simulation"]["foldername"])
    
    
    return holstein, sim_params, fa, input
end

end