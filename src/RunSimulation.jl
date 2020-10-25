module RunSimulation

using FFTW
using Random

using ..Models: AbstractModel
using ..SimulationParams: SimulationParameters
using ..GreensFunctions: EstimateGreensFunction, update!
using ..MuFinder: MuTuner, update_μ!
using ..FourierAcceleration: FourierAccelerator
using ..LangevinDynamics: evolve!, Dynamics, EulerDynamics, RungeKuttaDynamics, HeunsDynamics
using ..HMC: HybridMonteCarlo
import ..HMC

using ..Measurements: initialize_measurements_container, initialize_measurement_files!
using ..Measurements: make_measurements!, process_measurements!, write_measurements!, reset_measurements!

export run_simulation!

"""
Run Langevin simulation.
"""
function run_simulation!(model::AbstractModel, Gr::EstimateGreensFunction, μ_tuner::MuTuner, sim_params::SimulationParameters,
                         simulation_dynamics::Dynamics, burnin_dynamics::Dynamics, fa::FourierAccelerator,
                         measurement_info::Dict, preconditioner)

    ###############################################################
    ## PRE-ALLOCATING ARRAYS AND VARIABLES NEEDED FOR SIMULATION ##
    ###############################################################

    # initialize measurements container
    container = initialize_measurements_container(model,measurement_info)

    # initialize measurement files
    initialize_measurement_files!(container,sim_params)

    # keeps track of number of iterations needed to solve linear system
    iters = 0.0

    # time taken on langevin dynamics
    simulation_time = 0.0

    # time take on making measurements and write them to file
    measurement_time = 0.0

    # time taken writing data to file
    write_time = 0.0

    ##############################################
    ## RUNNING SIMULATION: THERMALIZATION STEPS ##
    ##############################################

    # frequency with which to update μ if tuning the denisty
    μ_update_freq = max(sim_params.meas_freq,1)

    # thermalizing system
    for interval in 1:div(sim_params.burnin,μ_update_freq)

        # evolve phonon fields according to dynamics
        for t in 1:μ_update_freq
            simulation_time += @elapsed iters += evolve!(model, burnin_dynamics, fa, preconditioner)
        end

        # update chemical potential
        if μ_tuner.active
            simulation_time += @elapsed update!(Gr,model,preconditioner)
            simulation_time += @elapsed update_μ!(model, μ_tuner, Gr)
        end
    end

    ###########################################
    ## RUNNING SIMULATION: MEASUREMENT STEPS ##
    ###########################################

    # iterate over bins
    for bin in 1:sim_params.num_bins

        # iterating over the size of each bin i.e. the number of measurements made per bin
        for n in 1:sim_params.bin_size

            # iterate over number of langevin steps per measurement
            for timestep in 1:sim_params.meas_freq

                simulation_time += @elapsed iters += evolve!(model, simulation_dynamics, fa, preconditioner)
            end

            # making measurements
            measurement_time += @elapsed make_measurements!(container,model,Gr,preconditioner)

            # update chemical potential
            if μ_tuner.active
                simulation_time += @elapsed update_μ!(model, μ_tuner, Gr)
            end
        end

        # process measurements
        measurement_time += @elapsed process_measurements!(container,sim_params,model)

        # write measurements to file
        write_time += @elapsed write_measurements!(container,sim_params,model,bin)

        # reset measurements container
        measurement_time += @elapsed reset_measurements!(container,model)
    end

    # calculating the average number of iterations needed to solve linear system
    iters /= (sim_params.nsteps+sim_params.burnin)

    # report timings in units of minutes
    simulation_time  /= 60.0
    measurement_time /= 60.0
    write_time       /= 60.0
    acceptance_rate   = 1.0

    return simulation_time, measurement_time, write_time, iters, acceptance_rate, container
end

"""
Run Hybrid Monte Carlo simulation.
"""
function run_simulation!(model::AbstractModel, Gr::EstimateGreensFunction, μ_tuner::MuTuner, sim_params::SimulationParameters,
                         simulation_hmc::HybridMonteCarlo, burnin_hmc::HybridMonteCarlo, fa::FourierAccelerator,
                         measurement_info::Dict, preconditioner)

    ###############################################################
    ## PRE-ALLOCATING ARRAYS AND VARIABLES NEEDED FOR SIMULATION ##
    ###############################################################

    # initialize measurements container
    container = initialize_measurements_container(model,measurement_info)

    # initialize measurement files
    initialize_measurement_files!(container,sim_params)

    # keeps track of number of iterations needed to solve linear system
    iters = 0.0

    # counts the number of accepted updates
    accepted_updates = 0

    # time taken on langevin dynamics
    simulation_time = 0.0

    # time take on making measurements and write them to file
    measurement_time = 0.0

    # time taken writing data to file
    write_time = 0.0

    ##############################################
    ## RUNNING SIMULATION: THERMALIZATION STEPS ##
    ##############################################

    # frequency with which to update μ if tuning the denisty
    μ_update_freq = max(sim_params.meas_freq,1)

    # thermalizing system
    for t in 1:sim_params.burnin

        simulation_time  += @elapsed accepted, niters = HMC.update!(model,burnin_hmc,fa,preconditioner)
        iters            += niters
        accepted_updates += accepted

        # update chemical potential
        if μ_tuner.active
            simulation_time += @elapsed update!(Gr,model,preconditioner)
            simulation_time += @elapsed update_μ!(model, μ_tuner, Gr)
        end
    end

    ###########################################
    ## RUNNING SIMULATION: MEASUREMENT STEPS ##
    ###########################################

    # iterate over bins
    for bin in 1:sim_params.num_bins

        # iterating over the size of each bin i.e. the number of measurements made per bin
        for n in 1:sim_params.bin_size

            # iterating over number of HMC updates between measurements
            for i in 1:sim_params.meas_freq

                # do hybrid monte carlo update
                simulation_time  += @elapsed accepted, niters = HMC.update!(model,simulation_hmc,fa,preconditioner)
                iters            += niters
                accepted_updates += accepted
            end

            # making measurements
            measurement_time += @elapsed make_measurements!(container,model,Gr,preconditioner)

            # update chemical potential
            if μ_tuner.active
                simulation_time += @elapsed update_μ!(model, μ_tuner, Gr)
            end
        end

        # process measurements
        measurement_time += @elapsed process_measurements!(container,sim_params,model)

        # write measurements to file
        write_time += @elapsed write_measurements!(container,sim_params,model,bin)

        # reset measurements container
        measurement_time += @elapsed reset_measurements!(container,model)
    end

    # calculating the average number of iterations needed to solve linear system
    iters /= (sim_params.nsteps+sim_params.burnin)

    # calculating the acceptance acceptance rate
    acceptance_rate = accepted_updates / (sim_params.nsteps+sim_params.burnin)

    # report timings in units of minutes
    simulation_time  /= 60.0
    measurement_time /= 60.0
    write_time       /= 60.0

    return simulation_time, measurement_time, write_time, iters, acceptance_rate, container
end

end