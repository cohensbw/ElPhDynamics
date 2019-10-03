module ProcessInputFile

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
    
    #############################
    ## DEFINING HOLSTEIN MODEL ##
    #############################
    
    # read in contents of input file into a dictionary
    info = file_to_dict(filename)
    
    # contruct lattice geometry
    ndim    = parse(Int,info["ndim"][1])
    norbits = parse(Int,info["norbits"][1])
    lvecs   = [parse.(Float64,info["lattice_vector_"*string(i)]) for i in 1:ndim]
    bvecs   = [parse.(Float64,info["basis_vector_"*string(i)]) for i in 1:norbits]
    geom    = Geometry(ndim, norbits, lvecs, bvecs)
    
    # construct lattice
    L       = parse(Int,info["L"][1])
    lattice = Lattice(geom,L)
    
    # declaring holstein model
    β  = parse(Float64,info["beta"][1])
    Δτ = parse(Float64,info["dtau"][1])
    holstein = HolsteinModel(geom,lattice,β,Δτ)
    
    for key in keys(info)
        
        # adding hopping parameter
        if startswith(key,"t") && isdigit(key[2])
            assign_tij!(holstein,
                        parse(Float64,info[key][1]),
                        parse(Float64,info[key][2]),
                        parse(Int,info[key][3]),
                        parse(Int,info[key][4]),
                        [parse(Int,info[key][5]),parse(Int,info[key][6]),parse(Int,info[key][7])])
            
        # assigning phonon frequency
        elseif startswith(key,"omega")
            assign_ω!(holstein, parse(Float64,info[key][1]),
                                parse(Float64,info[key][2]),
                                parse(Int,info[key][3]))
            
        # assigning electron-phonon coupling
        elseif startswith(key,"lambda")
            assign_λ!(holstein, parse(Float64,info[key][1]),
                                parse(Float64,info[key][2]),
                                parse(Int,info[key][3]))
            
        # assigning chemical potential
        elseif startswith(key,"mu")
            assign_μ!(holstein, parse(Float64,info[key][1]),
                                parse(Float64,info[key][2]),
                                parse(Int,info[key][3]))
        end
    end
    
    # organize electron hoppings for checkerboard decomposition
    setup_checkerboard!(holstein)

    # intialize phonon fields
    init_phonons_single_site!(holstein)
    
    ####################################
    ## DEFINING SIMULATION PARAMETERS ##
    ####################################
    
    # langevin time step
    Δt = parse(Float64,info["dt"][1])

    # tolerace of IterativeSolvers
    tol = parse(Float64,info["tol"][1])

    # number of thermalization steps
    burnin = parse(Int,info["burnin"][1])

    # total number of steps
    nsteps = parse(Int,info["nsteps"][1])

    # measurement frequency
    meas_freq = parse(Int,info["meas_freq"][1])

    # number of bins
    num_bins = parse(Int,info["num_bins"][1])

    # euler or runge-kutta updates
    euler = parse(Bool,info["euler"][1])

    # filepath to where to write data
    filepath = string(info["filepath"][1])

    # name of folder for data to get dumped into
    foldername = string(info["foldername"][1])
    
    # construct simulation parameters object.
    sim_params = SimulationParameters(Δt,euler,tol,burnin,nsteps,meas_freq,num_bins,filepath,foldername)
    
    ###################################
    ## DEFINING FOURIER ACCELERATION ##
    ###################################
    
    # defining FourierAccelerator type
    fa = FourierAccelerator(holstein,0.5,Δt)
    
    # setting mass term in fourier acceleration
    for key in keys(info)
        if startswith(key,"mass")
            update_Q!(fa,holstein,parse(Float64,info[key][1]), # mass
                                  Δt,
                                  parse(Float64,info[key][2]), # omega_min
                                  parse(Float64,info[key][3])) # omega_max
        end
    end

    return holstein, sim_params, fa
end

function file_to_dict(filename::String)
    
    d = Dict()
    open(filename, "r") do file
        for line in eachline(file)
            if !startswith(line,"#") && (line != "") && !startswith(line," ")
                atoms = rsplit(line)
                d[atoms[1]] = atoms[3:end]
            end
        end
    end
    return d
end

end