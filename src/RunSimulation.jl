module RunSimulation

using FFTW
using Random
using Serialization
using Parameters

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
                         container::NamedTuple, preconditioner, burnin_start::Int=1, sim_start::Int=1,
                         simulation_time::AbstractFloat=0.0, measurement_time::AbstractFloat=0.0, write_time::AbstractFloat=0.0,
                         iters::AbstractFloat=0.0, accepted_updates::Int=0)

    ###############################################################
    ## PRE-ALLOCATING ARRAYS AND VARIABLES NEEDED FOR SIMULATION ##
    ###############################################################

    # previous epoch time
    t_prev = 0.0

    # current epoch time
    t_new = time() 

    ##############################################
    ## RUNNING SIMULATION: THERMALIZATION STEPS ##
    ##############################################

    # frequency with which to update μ if tuning the denisty
    μ_update_freq = max(sim_params.meas_freq,1)

    # iterate over thermalization timesteps
    for t in burnin_start:sim_params.burnin

        # check if checkpoint need to be written
        t_new = time()
        if (t_new-t_prev) > sim_params.chckpnt_freq
            t_prev = t_new
            chkpnt = (model=model, μ_tuner=μ_tuner, container=container, burnin_start=t, sim_start=1,
                      simulation_time=simulation_time, measurement_time=measurement_time, write_time=write_time,
                      iters=iters, accepted_updates=accepted_updates)
            write_time += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)
        end

        # update phonon fields
        simulation_time += @elapsed iters += evolve!(model, burnin_dynamics, fa, preconditioner)

        # update chemical potential
        if μ_tuner.active && t%μ_update_freq==0
            simulation_time += @elapsed update!(Gr,model,preconditioner)
            simulation_time += @elapsed update_μ!(model, μ_tuner, Gr)
        end
    end

    ###########################################
    ## RUNNING SIMULATION: MEASUREMENT STEPS ##
    ###########################################

    # iterate over simulation time steps
    for t in sim_start:sim_params.nsteps

        # check if checkpoint needs to be written
        t_new = time()
        if (t_new-t_prev) > sim_params.chckpnt_freq
            t_prev = t_new
            chkpnt = (model=model, μ_tuner=μ_tuner, container=container, burnin_start=sim_params.burnin+1, sim_start=t,
                      simulation_time=simulation_time, measurement_time=measurement_time, write_time=write_time,
                      iters=iters, accepted_updates=accepted_updates)
            write_time += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)
        end

        # udpate phonon fields
        simulation_time += @elapsed iters += evolve!(model, simulation_dynamics, fa, preconditioner)

        # if time to make measurements
        if t%sim_params.meas_freq==0

            # make measurements
            measurement_time += @elapsed make_measurements!(container,model,Gr,preconditioner)

            # update chemical potential
            if μ_tuner.active
                simulation_time += @elapsed update_μ!(model, μ_tuner, Gr)
            end

            # get measurement number
            nmeas = div(t,sim_params.meas_freq)

            # if bin of measurements full
            if nmeas%sim_params.bin_size==0

                # get bin number
                bin = div(nmeas,sim_params.bin_size)

                # process measurements
                measurement_time += @elapsed process_measurements!(container,sim_params,model)

                # write measurements to file
                write_time += @elapsed write_measurements!(container,sim_params,model,bin)

                # reset measurements container
                measurement_time += @elapsed reset_measurements!(container,model)

                # write checkpoint
                t_new = time()
                t_prev = t_new
                chkpnt = (model=model, μ_tuner=μ_tuner, container=container, burnin_start=sim_params.burnin+1, sim_start=t+1,
                          simulation_time=simulation_time, measurement_time=measurement_time, write_time=write_time,
                          iters=iters, accepted_updates=accepted_updates)
                write_time += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)
            end
        end
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
function run_simulation!(model::AbstractModel, Gr::EstimateGreensFunction, μ_tuner::MuTuner, sim_params::SimulationParameters,
                         simulation_hmc::HybridMonteCarlo, burnin_hmc::HybridMonteCarlo, fa::FourierAccelerator,
                         container::NamedTuple, preconditioner, burnin_start::Int=1, sim_start::Int=1,
                         simulation_time::AbstractFloat=0.0, measurement_time::AbstractFloat=0.0, write_time::AbstractFloat=0.0,
                         iters::AbstractFloat=0.0, accepted_updates::Int=0)

    ###############################################################
    ## PRE-ALLOCATING ARRAYS AND VARIABLES NEEDED FOR SIMULATION ##
    ###############################################################

    # previous epoch time
    t_prev = 0.0

    # current epoch time
    t_new = time() 

    ##############################################
    ## RUNNING SIMULATION: THERMALIZATION STEPS ##
    ##############################################

    # thermalizing system
    for n in burnin_start:sim_params.burnin

        # check if checkpoint needs to be written
        t_new = time()
        if (t_new-t_prev) > sim_params.chckpnt_freq
            t_prev = t_new
            chkpnt = (model=model, μ_tuner=μ_tuner, container=container, burnin_start=n, sim_start=1,
                      simulation_time=simulation_time, measurement_time=measurement_time, write_time=write_time,
                      iters=iters, accepted_updates=accepted_updates)
            write_time += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)
        end

        simulation_time  += @elapsed accepted, niters = HMC.update!(model,burnin_hmc,fa,preconditioner)
        iters            += niters
        accepted_updates += accepted

        # update chemical potential
        if μ_tuner.active
            simulation_time += @elapsed update!(Gr,model,preconditioner)
            simulation_time += @elapsed update_μ!(model, μ_tuner, Gr)
        end
    end

    # close log file
    close(burnin_hmc.logfile)

    ###########################################
    ## RUNNING SIMULATION: MEASUREMENT STEPS ##
    ###########################################

    # iterate over hmc updates
    for n in sim_start:sim_params.nsteps

        # check if checkpoint needs to be written
        t_new = time()
        if (t_new-t_prev) > sim_params.chckpnt_freq
            t_prev = t_new
            chkpnt = (model=model, μ_tuner=μ_tuner, container=container, burnin_start=sim_params.burnin+1, sim_start=n,
                      simulation_time=simulation_time, measurement_time=measurement_time, write_time=write_time,
                      iters=iters, accepted_updates=accepted_updates)
            write_time += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)
        end

        # do hybrid monte carlo update
        simulation_time  += @elapsed accepted, niters = HMC.update!(model,simulation_hmc,fa,preconditioner)
        iters            += niters
        accepted_updates += accepted

        # if time to perform measurements (almost always yes)
        if n%sim_params.meas_freq==0

            # perform measurements
            measurement_time += @elapsed make_measurements!(container,model,Gr,preconditioner)

            # update chemical potential
            if μ_tuner.active
                simulation_time += @elapsed update_μ!(model, μ_tuner, Gr)
            end

            # get measurement number
            nmeas = div(n,sim_params.meas_freq)

            # if bin of measurements is full
            if nmeas%sim_params.bin_size==0

                # calculate bin number
                bin = div(nmeas,sim_params.bin_size)

                # process measurements
                measurement_time += @elapsed process_measurements!(container,sim_params,model)

                # write measurements to file
                write_time += @elapsed write_measurements!(container,sim_params,model,bin)

                # reset measurements container
                measurement_time += @elapsed reset_measurements!(container,model)

                # write checkpoint
                t_new = time()
                t_prev = t_new
                chkpnt = (model=model, μ_tuner=μ_tuner, container=container, burnin_start=sim_params.burnin+1, sim_start=n+1,
                          simulation_time=simulation_time, measurement_time=measurement_time, write_time=write_time,
                          iters=iters, accepted_updates=accepted_updates)
                write_time += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)
            end
        end
    end

    # calculating the average number of iterations needed to solve linear system
    iters /= (sim_params.nsteps+sim_params.burnin)

    # calculating the acceptance acceptance rate
    acceptance_rate = accepted_updates / (sim_params.nsteps+sim_params.burnin)

    # report timings in units of minutes
    simulation_time  /= 60.0
    measurement_time /= 60.0
    write_time       /= 60.0

    # close log files
    close(simulation_hmc.logfile)

    return simulation_time, measurement_time, write_time, iters, acceptance_rate
end

end