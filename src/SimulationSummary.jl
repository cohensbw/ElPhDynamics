module SimulationSummary

using Statistics
using Printf
using Pkg.TOML

using ..SimulationParams: SimulationParameters
using ..Models: AbstractBond, SSHBond, Bond
using ..Models: AbstractModel, SSHModel, HolsteinModel
using ..Models: write_phonons!, write_M_matrix

export initialize_simulation_summary!
export write_simulation_summary!

"""
Initialize simulation summary file.
"""
function initialize_simulation_summary!(model::AbstractModel{T1,T2,T3},sim_params::SimulationParameters,input::Dict) where {T1,T2,T3}

    # construct filename for summary filen
    filename = joinpath(sim_params.datafolder, sim_params.datafolder * "_summary.out")

    # open summary file
    open(filename,"w") do fout
        
        # copy contents of input file to output file
        write(fout,"#########################\n")
        write(fout,"## INPUT FILE CONTENTS ##\n")
        write(fout,"#########################\n\n")
        TOML.print(fout, input)
        write(fout,"\n")

        # write bond defintions to summary file
        write_bond_definitions!(fout,model)

        # write phonon definitions
        write_phonon_definitions!(fout,model)
    end

    return nothing
end

"""
Write results of simulation to simulation summary file.
"""
function write_simulation_summary!(model::AbstractModel{T1,T2,T3},sim_params::SimulationParameters,container::NamedTuple,
                                  simulation_time::T1, measurement_time::T1, write_time::T1,
                                  iters::T1, acceptance_rate::T1,Nbins::Int=10) where {T1,T2,T3}

    # data folder (including path)
    datafolder = sim_params.datafolder

    # data foldername
    foldername = sim_params.foldername

    # construct filename for summary filen
    filename = joinpath(datafolder, foldername * "_summary.out")

    # open summary file
    open(filename,"a") do fout
        
        # write simulation statistics to file
        write(fout, "#####################","\n")
        write(fout, "## SIMULATION INFO ##","\n")
        write(fout, "#####################","\n","\n")

        total_time = simulation_time + measurement_time + write_time
        write(fout, "Total Time (min) = ",       @sprintf("%.4f",total_time), "\n")
        write(fout, "Simulation Time (min) = ",  @sprintf("%.4f",simulation_time), "\n")
        write(fout, "Measurement Time (min) = ", @sprintf("%.4f",measurement_time), "\n")
        write(fout, "Write Time (min) = ",       @sprintf("%.4f",write_time), "\n")
        write(fout, "Iterative Solver Steps = ", @sprintf("%.4f",iters), "\n")
        write(fout, "Acceptance Rate = ",        @sprintf("%.4f",acceptance_rate), "\n", "\n")

        # write global measurements to file
        write(fout, "#########################","\n")
        write(fout, "## GLOBAL MEASUREMENTS ##","\n")
        write(fout, "#########################","\n","\n")

        write_global_measurements!(fout,model,container.global_meas,sim_params,Nbins)

    end

    return nothing
end


######################
## PRIVATE FUNCTION ##
######################

"""
Write bond defintions to file.
"""
function write_bond_definitions!(fout,model::AbstractModel{T1,T2,T3}) where {T1,T2,T3}

    bonds = model.bond_definitions

    write(fout,"######################\n")
    write(fout,"## BOND DEFINITIONS ##\n")
    write(fout,"######################\n\n")

    for (id, bond) in enumerate(bonds)

        write(fout,"Bond ID = ", string(id),      "\n")
        write(fout,"t_avg = ", string(bond.t),  "\n")
        write(fout,"t_std = ", string(bond.σt), "\n")
        write(fout,"Initial Orbit = ", string(bond.o₁), "\n")
        write(fout,"Final Orbit   = ", string(bond.o₁), "\n")
        write(fout,"Displacement  = ", string(bond.v),  "\n", "\n")
    end

    return nothing
end

"""
Write SSH Phonon definitions to file.
"""
function write_phonon_definitions!(fout,model::SSHModel{T1,T2,T3}) where {T1,T2,T3}

    bonds = model.bond_definitions

    write(fout,"############################\n")
    write(fout,"## SSH PHONON DEFINITIONS ##\n")
    write(fout,"############################\n\n")

    for (id, bond) in enumerate(bonds)

        write(fout,"SSH Phonon ID = ", string(id),      "\n")
        write(fout,"t_avg = ", string(bond.t),  "\n")
        write(fout,"t_std = ", string(bond.σt), "\n")
        write(fout,"alpha_avg = ", string(bond.α),  "\n")
        write(fout,"alpha_std = ", string(bond.σα), "\n")
        write(fout,"omega_avg = ", string(bond.ω),  "\n")
        write(fout,"omega_std = ", string(bond.σω), "\n")
        write(fout,"Initial Orbit = ", string(bond.o₁), "\n")
        write(fout,"Final Orbit   = ", string(bond.o₁), "\n")
        write(fout,"Displacement  = ", string(bond.v),  "\n", "\n")
    end

    return nothing
end

function write_phonon_definitions!(fout,model::HolsteinModel{T1,T2,T3}) where {T1,T2,T3}

    return nothing
end

"""
Write global measurements to file.
"""
function write_global_measurements!(fout,model::AbstractModel{T1,T2,T3},container::Dict,sim_params::SimulationParameters,Nbins::Int) where {T1,T2,T3}

    # filename containing global measurements
    datafn = joinpath(sim_params.datafolder, "global_measurements.out")

    # file statistics will be written to
    statsfn = joinpath(sim_params.datafolder, "global_measurements_stats.out")

    # number of measurements in data file
    Nmeas = sim_params.num_bins

    # calculate number of measurements per bin
    Nbins = min(Nmeas,Nbins)
    N     = div(Nmeas,Nbins)
    @assert Nmeas%Nbins==0

    # container to hold binned data
    binned_data = Dict(k=>zeros(T2,Nbins) for k in keys(container))

    # open file with global measurements data
    open(datafn,"r") do fin

        # read in header
        header = readline(fin)

        # get column names
        columns = split(header,",")

        # iterate over lines in file
        for line in eachline(fin)

            # split line apart
            atoms = split(line,",")

            # get the measurement number
            nmeas = parse(Int,atoms[1])

            # get bin
            bin = div(nmeas-1,N) + 1

            # iterate over measurements/columns
            for i in 2:length(columns)

                # get measurement
                measurement = columns[i]

                # record measurement
                binned_data[measurement][bin] += parse(T2,atoms[i]) / N
            end
        end
    end

    # write averaged statistics to file
    open(statsfn,"w") do statsout

        # header
        header = "global_meas avg error\n"
        write(fout,    header)
        write(statsout,header)

        # iterate over measurements
        for measurement in keys(binned_data)

            # calculate mean and standard deviation of mean
            avg, err = mean_and_error(binned_data[measurement])

            # write statistics to file
            line = measurement * @sprintf(" %.6f %.6f\n",avg,err)
            write(fout,    line)
            write(statsout,line)
        end

        write(fout,"\n")
    end

    return nothing
end

"""
Calculate mean and standard deviation of the mean of a set of data.
"""
function mean_and_error(v::AbstractArray{T})::Tuple{T,T} where {T<:Number}
    
    N   = length(v)
    avg = mean(v)
    err = std(v,corrected=true,mean=avg)/sqrt(N)
    return avg, err
end

end