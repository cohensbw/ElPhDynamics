module SimulationParams

export SimulationParameters

struct SimulationParameters

    "Number of thermalization updates."
    burnin::Int

    "Number of simulation updates."
    nsteps::Int

    "Measurement frequncy."
    meas_freq::Int

    "Number of measurements made."
    num_meas::Int

    "Number of measurements averaged over in a bin"
    bin_size::Int

    "Total number of bins."
    num_bins::Int

    "Number of langevin steps per bin."
    bin_steps::Int

    "checkpointing frequency in seconds"
    chckpnt_freq::Int

    "path to where the data should be written"
    filepath::String

    "name of folder data will be dumped into"
    foldername::String

    "filepath + foldername"
    datafolder::String

    function SimulationParameters(burnin::Int, nsteps::Int, meas_freq::Int, num_bins::Int, chckpnt_freq::Real,
                                  filepath::String, foldername::String, datafolder::String)

        # sanity check
        @assert nsteps >= meas_freq * num_bins

        # calculating the number of measurements that will be made in the simulation
        @assert nsteps%max(1,meas_freq)==0
        @assert burnin%max(1,meas_freq)==0
        num_meas = div(nsteps, max(1,meas_freq) )

        # calculating the number of measurements that will be averaged over in each bin
        @assert num_meas%max(1,num_bins)==0
        bin_size = div(num_meas, max(1,num_bins) )
        
        # calculating the number of langevin time steps per bin
        bin_steps = meas_freq*bin_size

        # convert checkpointing frequency in minutes to seconds
        chckpnt_freq = round(Int,60*chckpnt_freq)

        new(burnin,nsteps,meas_freq,num_meas,bin_size,num_bins,bin_steps,chckpnt_freq,filepath,foldername,datafolder)
    end
end

end