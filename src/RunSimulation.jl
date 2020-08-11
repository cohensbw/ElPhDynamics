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

using ..NonLocalMeasurements: make_nonlocal_measurements!, reset_nonlocal_measurements!
using ..NonLocalMeasurements: process_nonlocal_measurements!, construct_nonlocal_measurements_container
using ..NonLocalMeasurements: initialize_nonlocal_measurement_files
using ..NonLocalMeasurements: write_nonlocal_measurements

using ..LocalMeasurements: make_local_measurements!, reset_local_measurements!
using ..LocalMeasurements: process_local_measurements!, construct_local_measurements_container
using ..LocalMeasurements: initialize_local_measurements_file
using ..LocalMeasurements: write_local_measurements

export run_simulation!

"""
Run Langevin simulation.
"""
function run_simulation!(model::AbstractModel, Gr::EstimateGreensFunction, μ_tuner::MuTuner, sim_params::SimulationParameters, dynamics::Dynamics, fa::FourierAccelerator,
                         unequaltime_meas::AbstractVector{String}, equaltime_meas::AbstractVector{String}, preconditioner)

    ###############################################################
    ## PRE-ALLOCATING ARRAYS AND VARIABLES NEEDED FOR SIMULATION ##
    ###############################################################

    # declare container for storing non-local measurements in both
    # position-space and momentum-space
    container_rspace, container_kspace = construct_nonlocal_measurements_container(model, equaltime_meas, unequaltime_meas)

    # constructing container to hold local measurements
    local_meas_container = construct_local_measurements_container(model, unequaltime_meas)

    # Creating files that data will be written to.
    initialize_nonlocal_measurement_files(container_rspace, container_kspace, sim_params)
    initialize_local_measurements_file(local_meas_container, sim_params)

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
            simulation_time += @elapsed iters += evolve!(model, dynamics, fa, preconditioner)
        end

        # update chemical potential
        if μ_tuner.active
            simulation_time += @elapsed update!(Gr,model,preconditioner)
            simulation_time += @elapsed update_μ!(model.μ, μ_tuner, Gr.r₁, Gr.M⁻¹r₁, Gr.r₂, Gr.M⁻¹r₂)
        end
    end

    ###########################################
    ## RUNNING SIMULATION: MEASUREMENT STEPS ##
    ###########################################

    # iterate over bins
    for bin in 1:sim_params.num_bins

        # reset values in measurement containers
        reset_nonlocal_measurements!(container_rspace)
        reset_nonlocal_measurements!(container_kspace)
        reset_local_measurements!(local_meas_container)

        # iterating over the size of each bin i.e. the number of measurements made per bin
        for n in 1:sim_params.bin_size

            # iterate over number of langevin steps per measurement
            for timestep in 1:sim_params.meas_freq

                simulation_time += @elapsed iters += evolve!(model, dynamics, fa, preconditioner)
            end

            # update stochastic estimates of the Green's functions
            measurement_time += @elapsed update!(Gr,model,preconditioner)

            # making non-local measurements
            measurement_time += @elapsed make_nonlocal_measurements!(container_rspace, model, Gr)

            # make local measurements
            measurement_time += @elapsed make_local_measurements!(local_meas_container, model, Gr)

            # update chemical potential
            if μ_tuner.active
                simulation_time += @elapsed update_μ!(model.μ, μ_tuner, Gr.r₁, Gr.M⁻¹r₁, Gr.r₂, Gr.M⁻¹r₂)
            end
        end

        # process non-local measurements. This includes normalizing the real-space measurements
        # by the number of measurements made per bin, and also taking the Fourier Transform in order
        # to get the momentum-space measurements.
        measurement_time += @elapsed process_nonlocal_measurements!(container_rspace, container_kspace, sim_params)

        # process local measurements. This includes calculating certain derived quantities (like S-wave Susceptibility)
        measurement_time += @elapsed process_local_measurements!(local_meas_container, sim_params, model,
                                                                 container_rspace, container_kspace)

        # Write non-local measurements to file. Note that there is a little bit more averaging going on here as well.
        write_time += @elapsed write_nonlocal_measurements(container_rspace,sim_params,model,real_space=true)
        write_time += @elapsed write_nonlocal_measurements(container_kspace,sim_params,model,real_space=false)

        # write local measurements to file
        write_time += @elapsed write_local_measurements(local_meas_container,sim_params,model)
    end

    # calculating the average number of iterations needed to solve linear system
    iters /= (sim_params.nsteps+sim_params.burnin)

    # report timings in units of minutes
    simulation_time  /= 60.0
    measurement_time /= 60.0
    write_time       /= 60.0
    acceptance_rate   = 1.0

    return simulation_time, measurement_time, write_time, iters, acceptance_rate
end

"""
Run Hybrid Monte Carlo simulation.
"""
function run_simulation!(model::AbstractModel, Gr::EstimateGreensFunction, μ_tuner::MuTuner, sim_params::SimulationParameters, hmc::HybridMonteCarlo, fa::FourierAccelerator,
                         unequaltime_meas::AbstractVector{String}, equaltime_meas::AbstractVector{String}, preconditioner)

    ###############################################################
    ## PRE-ALLOCATING ARRAYS AND VARIABLES NEEDED FOR SIMULATION ##
    ###############################################################

    # declare container for storing non-local measurements in both
    # position-space and momentum-space
    container_rspace, container_kspace = construct_nonlocal_measurements_container(model, equaltime_meas, unequaltime_meas)

    # constructing container to hold local measurements
    local_meas_container = construct_local_measurements_container(model, unequaltime_meas)

    # Creating files that data will be written to.
    initialize_nonlocal_measurement_files(container_rspace, container_kspace, sim_params)
    initialize_local_measurements_file(local_meas_container, sim_params)

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
    for i in 1:div(sim_params.burnin,μ_update_freq)

        # update phonon fields with HMC udpates
        for j in 1:μ_update_freq
            simulation_time  += @elapsed accepted, niters = HMC.update!(model,hmc,fa,preconditioner)
            iters            += niters
            accepted_updates += accepted
        end

        # update chemical potential
        if μ_tuner.active
            simulation_time += @elapsed update!(Gr,model,preconditioner)
            simulation_time += @elapsed update_μ!(model.μ, μ_tuner, Gr.r₁, Gr.M⁻¹r₁, Gr.r₂, Gr.M⁻¹r₂)
        end
    end

    ###########################################
    ## RUNNING SIMULATION: MEASUREMENT STEPS ##
    ###########################################

    # iterate over bins
    for bin in 1:sim_params.num_bins

        # reset values in measurement containers
        reset_nonlocal_measurements!(container_rspace)
        reset_nonlocal_measurements!(container_kspace)
        reset_local_measurements!(local_meas_container)

        # iterating over the size of each bin i.e. the number of measurements made per bin
        for n in 1:sim_params.bin_size

            # iterating over number of HMC updates between measurements
            for i in 1:sim_params.meas_freq

                # do hybrid monte carlo update
                simulation_time  += @elapsed accepted, niters = HMC.update!(model,hmc,fa,preconditioner)
                iters            += niters
                accepted_updates += accepted
            end

            # update stochastic estimates of the Green's functions
            measurement_time += @elapsed update!(Gr,model,preconditioner)

            # making non-local measurements
            measurement_time += @elapsed make_nonlocal_measurements!(container_rspace, model, Gr)

            # make local measurements
            measurement_time += @elapsed make_local_measurements!(local_meas_container, model, Gr)

            # update chemical potential
            if μ_tuner.active
                simulation_time += @elapsed update_μ!(model.μ, μ_tuner, Gr.r₁, Gr.M⁻¹r₁, Gr.r₂, Gr.M⁻¹r₂)
            end
        end

        # process non-local measurements. This includes normalizing the real-space measurements
        # by the number of measurements made per bin, and also taking the Fourier Transform in order
        # to get the momentum-space measurements.
        measurement_time += @elapsed process_nonlocal_measurements!(container_rspace, container_kspace, sim_params)

        # process local measurements. This includes calculating certain derived quantities (like S-wave Susceptibility)
        measurement_time += @elapsed process_local_measurements!(local_meas_container, sim_params, model,
                                                                 container_rspace, container_kspace)

        # Write non-local measurements to file. Note that there is a little bit more averaging going on here as well.
        write_time += @elapsed write_nonlocal_measurements(container_rspace,sim_params,model,real_space=true)
        write_time += @elapsed write_nonlocal_measurements(container_kspace,sim_params,model,real_space=false)

        # write local measurements to file
        write_time += @elapsed write_local_measurements(local_meas_container,sim_params,model)
    end

    # calculating the average number of iterations needed to solve linear system
    iters /= (sim_params.nsteps+sim_params.burnin)

    # calculating the acceptance acceptance rate
    acceptance_rate = accepted_updates / (sim_params.nsteps+sim_params.burnin)

    # report timings in units of minutes
    simulation_time  /= 60.0
    measurement_time /= 60.0
    write_time       /= 60.0

    return simulation_time, measurement_time, write_time, iters, acceptance_rate
end

end