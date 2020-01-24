module LangevinSimulationParameters

export SimulationParameters

struct SimulationParameters{T<:AbstractFloat}

    "Langevin time step"
    Δt::T

    "Whether to use Euler or Runge-Kutta update method."
    euler::Bool

    "Number of thermalization steps time steps."
    burnin::Int

    "Number of time steps in main simulation."
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

    "Down sampling in imaginary time direction when making measurement."
    downsample::Int

    "path to where the data should be written"
    filepath::String

    "name of folder data will be dumped into"
    foldername::String

    "filepath + foldername"
    datafolder::String

    function SimulationParameters(Δt::T, euler::Bool, burnin::Int, nsteps::Int, meas_freq::Int, num_bins::Int, downsample::Int, filepath::String, foldername::String) where {T<:AbstractFloat}

        # sanity check
        @assert nsteps>=meas_freq*num_bins

        # calculating the number of measurements that will be made in the simulation
        @assert nsteps%max(1,meas_freq)==0
        num_meas = div(nsteps, max(1,meas_freq) )

        # calculating the number of measurements that will be averaged over in each bin
        @assert num_meas%max(1,num_bins)==0
        bin_size = div(num_meas, max(1,num_bins) )
        
        # calculating the number of langevin time steps per bin
        bin_steps = meas_freq*bin_size

        @assert downsample > 0

        # formatting filepath
        if !endswith(filepath,"/")
            filepath *= "/"
        end

        # formatting foldername
        if !endswith(foldername,"/")
            foldername *= "/"
        end

        # data folder, including complete path to folder
        datafolder = filepath*foldername

        # making directory the data will be written into
        if !isdir(datafolder)
            mkdir(datafolder)
        end

        new{T}(Δt,euler,burnin,nsteps,meas_freq,num_meas,bin_size,num_bins,bin_steps,downsample,filepath,foldername,datafolder)
    end
end

end