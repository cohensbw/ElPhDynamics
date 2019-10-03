module LangevinSimulationParameters

export SimulationParameters

struct SimulationParameters{T<:AbstractFloat}

    "Langevin time step"
    Δt::T

    "Whether to use Euler or Runge-Kutta update method."
    euler::Bool

    "Tolerance used when solving linear systems"
    tol::T

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

    "path to where the data should be written"
    filepath::String

    "name of folder data will be dumped into"
    foldername::String

    "data folder"
    datafolder::String

    function SimulationParameters(Δt::T, euler::Bool, tol::T, burnin::Int, nsteps::Int, meas_freq::Int, num_bins::Int,
                                  filepath::String, foldername::String) where {T<:AbstractFloat}

        # calculating the number of measurements that will be made in the simulation
        @assert nsteps%meas_freq==0
        num_meas = div(nsteps,meas_freq)

        # calculating the number of measurements that will be averaged over in each bin
        @assert num_meas%num_bins==0
        bin_size = div(num_meas, num_bins)
        
        # calculating the number of langevin time steps per bin
        bin_steps = meas_freq*bin_size

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

        new{T}(Δt,euler,tol,burnin,nsteps,meas_freq,num_meas,bin_size,num_bins,bin_steps,filepath,foldername,datafolder)
    end
end

end