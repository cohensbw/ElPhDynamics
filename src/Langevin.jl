module Langevin

#######################
## INCLUDING MODULES ##
#######################

include("Utilities.jl")

include("Geometries.jl")

include("Lattices.jl")

include("Checkerboard.jl")

include("HolsteinModels.jl")

include("InitializePhonons.jl")

include("PhononAction.jl")

include("FourierAcceleration.jl")

include("LangevinDynamics.jl")

include("GreensFunctions.jl")

include("FourierTransforms.jl")

include("LatticeFFTs.jl")

include("TimeFreqFFTs.jl")

include("LangevinSimulationParameters.jl")

include("NonLocalMeasurements.jl")

include("LocalMeasurements.jl")

include("RunSimulation.jl")

include("ProcessInputFile.jl")

include("SimulationSummary.jl")

include("RestartedGMRES.jl")

include("BlockPreconditioners.jl")

include("Preconditioners.jl")

####################################
## DEFINING HIGHET LEVEL FUNCTION ##
##     TO RUN A SIMULATION        ##
####################################

using ..RunSimulation: run_simulation!
using ..ProcessInputFile: process_input_file
using ..SimulationSummary: write_simulation_summary

export simulate

"""
Highest level function used to run a langevin simulation of a Holstein model.
To run a simulation once this package has been installed run the following command:
`julia -O3 -e "using Langevin; simulate(ARGS)" -- input.toml`
"""
function simulate(args)

    ########################
    ## READING INPUT FILE ##
    ########################

    # getting iput filename
    input_file = args[1]

    # precoessing input file
    holstein, sim_params, fourier_accelerator, input = process_input_file(input_file)

    ########################
    ## RUNNING SIMULATION ##
    ########################

    simulation_time, measurement_time, write_time, iters = run_simulation!(holstein, sim_params, fourier_accelerator)

    ###################################
    ## SUMARIZING SIMULATION RESULTS ##
    ###################################

    write_simulation_summary(holstein, input, sim_params, simulation_time, measurement_time, write_time, iters)
end

end # module
