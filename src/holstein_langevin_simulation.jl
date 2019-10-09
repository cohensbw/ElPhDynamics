using Langevin.ProcessInputFile: process_input_file
using Langevin.RunSimulation: run_simulation!
using Langevin.SimulationSummary: write_simulation_summary

########################
## READING INPUT FILE ##
########################

# getting iput filename
input_file = ARGS[1]

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