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
using ..SpecialUpdates: SpecialUpdate, NullUpdate, ReflectionUpdate, SwapUpdate, special_update!
using ..Measurements: initialize_measurements_container, initialize_measurement_folders!
using ..Measurements: make_measurements!, process_measurements!, write_measurements!, reset_measurements!

export run_simulation!

"""
Run Langevin simulation.
"""
function run_simulation!(model::AbstractModel, Gr::EstimateGreensFunction, μ_tuner::MuTuner, sim_params::SimulationParameters,
                         simulation_dynamics::Dynamics, burnin_dynamics::Dynamics,
                         sim_reflect_update::SpecialUpdate, burnin_reflect_update::SpecialUpdate,
                         sim_swap_update::SpecialUpdate, burnin_swap_update::SpecialUpdate,
                         fa::FourierAccelerator, container::NamedTuple, preconditioner, sim_stats::Dict,
                         burnin_start::Int=1, sim_start::Int=1,)::Dict

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
            chkpnt = (model=model, μ_tuner=μ_tuner, container=container,
                      burnin_start=t, sim_start=1, sim_stats=sim_stats)
            sim_stats["write_time"] += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)
        end

        # update phonon fields
        sim_stats["simulation_time"] += @elapsed sim_stats["iters"] += evolve!(model, burnin_dynamics, fa, preconditioner)

        # update chemical potential
        if μ_tuner.active && t%μ_update_freq==0
            sim_stats["simulation_time"] += @elapsed update!(Gr,model,preconditioner)
            sim_stats["simulation_time"] += @elapsed update_μ!(model, μ_tuner, Gr)
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
            chkpnt = (model=model, μ_tuner=μ_tuner, container=container,
                      burnin_start=sim_params.burnin+1, sim_start=t, sim_stats=sim_stats)
            sim_stats["write_time"] += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)
        end

        # udpate phonon fields
        sim_stats["simulation_time"] += @elapsed sim_stats["iters"] += evolve!(model, simulation_dynamics, fa, preconditioner)

        # if time to make measurements
        if t%sim_params.meas_freq==0

            # make measurements
            nmeas = div(t,sim_params.meas_freq)
            sim_stats["measurement_time"] += @elapsed make_measurements!(container,model,Gr,nmeas,preconditioner)

            # update chemical potential
            if μ_tuner.active
                sim_stats["simulation_time"] += @elapsed update_μ!(model, μ_tuner, Gr)
            end

            # get measurement number
            nmeas = div(t,sim_params.meas_freq)

            # if bin of measurements full
            if nmeas%sim_params.bin_size==0

                # get bin number
                bin = div(nmeas,sim_params.bin_size)

                # process measurements
                sim_stats["measurement_time"] += @elapsed process_measurements!(container,sim_params,model)

                # write measurements to file
                sim_stats["write_time"] += @elapsed write_measurements!(container,model,bin)

                # reset measurements container
                sim_stats["measurement_time"] += @elapsed reset_measurements!(container,model)

                # write checkpoint
                t_new = time()
                t_prev = t_new
                chkpnt = (model=model, μ_tuner=μ_tuner, container=container,
                          burnin_start=sim_params.burnin+1, sim_start=t+1, sim_stats=sim_stats)
                sim_stats["write_time"] += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)
            end
        end
    end

    # calculating the average number of iterations needed to solve linear system
    sim_stats["iters"] /= (sim_params.nsteps+sim_params.burnin)

    # report timings in units of minutes
    sim_stats["simulation_time"]  /= 60.0
    sim_stats["measurement_time"] /= 60.0
    sim_stats["write_time"]       /= 60.0
    sim_stats["acceptance_rate"]   = 1.0

    chkpnt = (model=model, μ_tuner=μ_tuner, burnin_start=sim_params.burnin+1,
              sim_start=sim_params.nsteps+1, sim_stats=sim_stats)
    sim_stats["write_time"] += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)

    return sim_stats
end

"""
Run Hybrid Monte Carlo simulation.
"""
function run_simulation!(model::AbstractModel, Gr::EstimateGreensFunction, μ_tuner::MuTuner, sim_params::SimulationParameters,
                         simulation_hmc::HybridMonteCarlo, burnin_hmc::HybridMonteCarlo,
                         sim_reflect_update::SpecialUpdate, burnin_reflect_update::SpecialUpdate,
                         sim_swap_update::SpecialUpdate, burnin_swap_update::SpecialUpdate,
                         fa::FourierAccelerator, container::NamedTuple, preconditioner, sim_stats::Dict,
                         burnin_start::Int=1, sim_start::Int=1)::Dict

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
            chkpnt = (model=model, μ_tuner=μ_tuner, container=container,
                      burnin_start=n, sim_start=1, sim_stats=sim_stats)
            sim_stats["write_time"] += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)
        end

        # perform hmc update
        sim_stats["simulation_time"] += @elapsed accepted, niters = HMC.update!(model,burnin_hmc,fa,preconditioner)
        sim_stats["iters"]           += niters
        sim_stats["acceptance_rate"] += accepted

        # do reflection update
        if burnin_reflect_update.active && mod(n,burnin_reflect_update.freq)==0
            sim_stats["simulation_time"] += @elapsed accepted = special_update!(model,burnin_hmc,burnin_reflect_update,preconditioner)
            sim_stats["reflect_acceptance_rate"] += accepted
        end

        # do swap update
        if burnin_swap_update.active && mod(n,burnin_swap_update.freq)==0
            sim_stats["simulation_time"] += @elapsed accepted = special_update!(model,burnin_hmc,burnin_swap_update,preconditioner)
            sim_stats["swap_acceptance_rate"] += accepted
        end

        # update chemical potential
        if μ_tuner.active
            sim_stats["simulation_time"] += @elapsed update!(Gr,model,preconditioner)
            sim_stats["simulation_time"] += @elapsed update_μ!(model, μ_tuner, Gr)
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
            chkpnt = (model=model, μ_tuner=μ_tuner, container=container,
                      burnin_start=sim_params.burnin+1, sim_start=n, sim_stats=sim_stats)
            sim_stats["write_time"] += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)
        end

        # perform hmc update
        sim_stats["simulation_time"] += @elapsed accepted, niters = HMC.update!(model,simulation_hmc,fa,preconditioner)
        sim_stats["iters"]           += niters
        sim_stats["acceptance_rate"] += accepted

        # do reflection update
        if burnin_reflect_update.active && mod(n,sim_reflect_update.freq)==0
            sim_stats["simulation_time"] += @elapsed accepted = special_update!(model,burnin_hmc,sim_reflect_update,preconditioner)
            sim_stats["reflect_acceptance_rate"] += accepted
        end

        # do swap update
        if burnin_swap_update.active && mod(n,sim_swap_update.freq)==0
            sim_stats["simulation_time"] += @elapsed accepted = special_update!(model,burnin_hmc,sim_swap_update,preconditioner)
            sim_stats["swap_acceptance_rate"] += accepted
        end

        # if time to perform measurements (almost always yes)
        if n%sim_params.meas_freq==0

            # perform measurements
            nmeas = div(n,sim_params.meas_freq)
            sim_stats["measurement_time"] += @elapsed make_measurements!(container,model,Gr,nmeas,preconditioner)

            # update chemical potential
            if μ_tuner.active
                sim_stats["simulation_time"] += @elapsed update_μ!(model, μ_tuner, Gr)
            end

            # get measurement number
            nmeas = div(n,sim_params.meas_freq)

            # if bin of measurements is full
            if nmeas%sim_params.bin_size==0

                # calculate bin number
                bin = div(nmeas,sim_params.bin_size)

                # process measurements
                sim_stats["measurement_time"] += @elapsed process_measurements!(container,sim_params,model)

                # write measurements to file
                sim_stats["write_time"] += @elapsed write_measurements!(container,model,bin)

                # reset measurements container
                sim_stats["measurement_time"] += @elapsed reset_measurements!(container,model)

                # write checkpoint
                t_new = time()
                t_prev = t_new
                chkpnt = (model=model, μ_tuner=μ_tuner, container=container,
                          burnin_start=sim_params.burnin+1, sim_start=n+1, sim_stats=sim_stats)
                sim_stats["write_time"] += @elapsed serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)
            end
        end
    end

    # calculating the average number of iterations needed to solve linear system
    sim_stats["iters"] /= (sim_params.nsteps+sim_params.burnin)

    # calculating the acceptance acceptance rate
    sim_stats["acceptance_rate"] /= (sim_params.nsteps + sim_params.burnin)

    # calculating special acceptance rate
    burnin_freq = burnin_reflect_update.freq
    sim_freq    = sim_reflect_update.freq
    sim_stats["reflect_acceptance_rate"] /= ( div(sim_params.nsteps,sim_freq) + div(sim_params.burnin,burnin_freq) )

    # calculating special acceptance rate
    burnin_freq = burnin_swap_update.freq
    sim_freq    = sim_swap_update.freq
    sim_stats["swap_acceptance_rate"] /= ( div(sim_params.nsteps,sim_freq) + div(sim_params.burnin,burnin_freq) )

    # report timings in units of minutes
    sim_stats["simulation_time"]  /= 60.0
    sim_stats["measurement_time"] /= 60.0
    sim_stats["write_time"]       /= 60.0

    # final checkpoint
    chkpnt = (model=model, μ_tuner=μ_tuner, burnin_start=sim_params.burnin+1,
              sim_start=sim_params.nsteps+1, sim_stats=sim_stats)
    serialize(joinpath(sim_params.datafolder,"checkpoint.jls"),chkpnt)

    # close log files
    close(simulation_hmc.logfile)

    return sim_stats
end

end