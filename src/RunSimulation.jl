module RunSimulation

using ..HolsteinModels: HolsteinModel
using ..LangevinSimulationParameters: SimulationParameters
using ..GreensFunctions: EstimateGreensFunction, update!
using ..LangevinDynamics: update_euler_fa!, update_rk_fa!
using ..FourierAcceleration: FourierAccelerator
using ..FourierTransforms: calc_fourier_transform_coefficients
using ..BlockPreconditioners: BlockPreconditioner, setup!

using ..NonLocalMeasurements: make_nonlocal_measurements!, reset_nonlocal_measurements!
using ..NonLocalMeasurements: process_nonlocal_measurements!, construct_nonlocal_measurements_container
using ..NonLocalMeasurements: initialize_nonlocal_measurement_files
using ..NonLocalMeasurements: write_nonlocal_measurements

using ..LocalMeasurements: make_local_measurements!, reset_local_measurements!
using ..LocalMeasurements: process_local_measurements!, construct_local_measurements_container
using ..LocalMeasurements: initialize_local_measurements_file
using ..LocalMeasurements: write_local_measurements

export run_simulation!

function run_simulation!(holstein::HolsteinModel{T1,T2}, sim_params::SimulationParameters{T1}, fa::FourierAccelerator{T1}, preconditioner) where {T1<:AbstractFloat, T2<:Number}

    ###############################################################
    ## PRE-ALLOCATING ARRAYS AND VARIABLES NEEDED FOR SIMULATION ##
    ###############################################################

    dϕ     = zeros(Float64, length(holstein))
    fft_dϕ = zeros(Complex{Float64}, length(holstein))

    dSdϕ     = zeros(Float64, length(holstein))
    fft_dSdϕ = zeros(Complex{Float64}, length(holstein))
    dSdϕ2    = zeros(Float64, length(holstein))

    g    = zeros(Float64, length(holstein))
    M⁻¹g = zeros(Float64, length(holstein))

    η     = zeros(Float64, length(holstein))
    fft_η = zeros(Complex{Float64}, length(holstein))

    # declare two electron greens function estimators
    Gr1 = EstimateGreensFunction(holstein)
    Gr2 = EstimateGreensFunction(holstein)

    # declare container for storing non-local measurements in both
    # position-space and momentum-space
    container_rspace, container_kspace = construct_nonlocal_measurements_container(holstein)

    # constructing container to hold local measurements
    local_meas_container = construct_local_measurements_container(holstein)

    # caluclating Fourier Transform coefficients
    ft_coeff = calc_fourier_transform_coefficients(holstein.lattice)

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

    # thermalizing system
    for timestep in 1:sim_params.burnin

        # set up block preconditioner if being used
        if holstein.use_gmres
            setup!(preconditioner)
        end

        if sim_params.euler
            # using Euler method with Fourier Acceleration
            simulation_time += @elapsed iters += update_euler_fa!(holstein, fa, dϕ, fft_dϕ, dSdϕ, fft_dSdϕ, g, M⁻¹g, η, fft_η, sim_params.Δt, preconditioner)
        else
            # using Runge-Kutta method with Fourier Acceleration
            simulation_time += @elapsed iters += update_rk_fa!(holstein, fa, dϕ, fft_dϕ, dSdϕ2, dSdϕ, fft_dSdϕ, g, M⁻¹g, η, fft_η, sim_params.Δt, preconditioner)
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

                # set up block preconditioner if being used
                if holstein.use_gmres
                    setup!(preconditioner)
                end

                if sim_params.euler
                    # using Euler method with Fourier Acceleration
                    simulation_time += @elapsed iters += update_euler_fa!(holstein, fa, dϕ, fft_dϕ, dSdϕ, fft_dSdϕ, g, M⁻¹g, η, fft_η, sim_params.Δt, preconditioner)
                else
                    # using Runge-Kutta method with Fourier Acceleration
                    simulation_time += @elapsed iters += update_rk_fa!(holstein, fa, dϕ, fft_dϕ, dSdϕ2, dSdϕ, fft_dSdϕ, g, M⁻¹g, η, fft_η, sim_params.Δt, preconditioner)
                end
            end

            # set up block preconditioner if being used
            if holstein.use_gmres
                setup!(preconditioner)
            end

            # update stochastic estimates of the Green's functions
            measurement_time += @elapsed update!(Gr1,holstein,preconditioner)
            measurement_time += @elapsed update!(Gr2,holstein,preconditioner)

            # making non-local measurements
            measurement_time += @elapsed make_nonlocal_measurements!(container_rspace, holstein, Gr1, Gr2, sim_params.downsample)

            # make local measurements
            measurement_time += @elapsed make_local_measurements!(local_meas_container, holstein, Gr1, Gr2)
        end

        # process non-local measurements. This include normalizing the real-space measurements
        # by the number of measurements made per bin, and also taking the Fourier Transform in order
        # to get the momentum-space measurements.
        measurement_time += @elapsed process_nonlocal_measurements!(container_rspace, container_kspace, sim_params, ft_coeff)

        # process local measurements
        measurement_time += @elapsed process_local_measurements!(local_meas_container, sim_params, holstein)

        # Write non-local measurements to file. Note that there is a little bit more averaging going on here as well.
        write_time += @elapsed write_nonlocal_measurements(container_rspace,sim_params,holstein,real_space=true)
        write_time += @elapsed write_nonlocal_measurements(container_kspace,sim_params,holstein,real_space=false)

        # write local measurements to file
        write_time += @elapsed write_local_measurements(local_meas_container,sim_params,holstein)
    end

    # calculating the average number of iterations needed to solve linear system
    iters /= (sim_params.nsteps+sim_params.burnin)

    # report timings in units of minutes
    simulation_time  /= 60.0
    measurement_time /= 60.0
    write_time       /= 60.0

    return simulation_time, measurement_time, write_time, iters
end

end