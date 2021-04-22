module SimulationSummary

using Statistics
using Printf
using TOML

using ..SimulationParams: SimulationParameters
using ..Models: AbstractBond, SSHBond, Bond
using ..Models: AbstractModel, SSHModel, HolsteinModel, write_phonons!
using ..Models: write_phonons!, write_M_matrix!
using ..Measurements: measure_κ
using ..MuFinder: MuTuner, update_μ!, estimate_μ
using ..Utilities: reshaped

export initialize_simulation_summary!
export write_simulation_summary!

"""
Initialize simulation summary file.
"""
function initialize_simulation_summary!(model::AbstractModel{T1,T2,T3},sim_params::SimulationParameters,input::Dict) where {T1,T2,T3}

    # construct filename for summary filen
    filename = joinpath(sim_params.datafolder, "$(splitpath(sim_params.datafolder)[end])_summary.out")

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

        # write chemical potential defintions
        write_chemical_potential_definitions!(fout,model)
    end

    return nothing
end

"""
Write results of simulation to simulation summary file.
"""
function write_simulation_summary!(model::AbstractModel{T1,T2,T3}, sim_params::SimulationParameters,
                                   μ_tuner::MuTuner{T1}, container::NamedTuple, input::Dict,
                                   simulation_time::T1, measurement_time::T1, write_time::T1,
                                   iters::T1, acceptance_rate::T1,Nbins::Int=10) where {T1,T2,T3}

    # data folder (including path)
    datafolder = sim_params.datafolder

    # data foldername
    foldername = sim_params.foldername

    # write phonon fields to file
    write_phonons!(model,joinpath(datafolder,"$(foldername)_config.out"))

    # write M matrix to file
    if haskey(input["simulation"],"write_M_matrix")
        if input["simulation"]["write_M_matrix"]
            write_M_matrix!(model, joinpath(datafolder,"$(foldername)_matrix.out"))
        end
    end

    # construct filename for summary file
    filename = joinpath(sim_params.datafolder, "$(splitpath(sim_params.datafolder)[end])_summary.out")

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
        write(fout, "Acceptance Rate = ",        @sprintf("%.4f",acceptance_rate), "\n")

        # write global measurements to file
        write(fout,"\n")
        write(fout, "#########################","\n")
        write(fout, "## GLOBAL MEASUREMENTS ##","\n")
        write(fout, "#########################","\n","\n")

        write_global_measurements!(fout,model,μ_tuner,container.global_meas,sim_params,Nbins)

        # write global measurements to file
        write(fout,"\n")
        write(fout, "##########################","\n")
        write(fout, "## ON-SITE MEASUREMENTS ##","\n")
        write(fout, "##########################","\n","\n")

        write_onsite_measurements!(fout,model,μ_tuner,container.onsite_meas,sim_params,Nbins)

        # write global measurements to file
        write(fout,"\n")
        write(fout, "#############################","\n")
        write(fout, "## INTER-SITE MEASUREMENTS ##","\n")
        write(fout, "#############################","\n","\n")

        write_intersite_measurements!(fout,model,container.intersite_meas,sim_params,Nbins)

        # write on-site correlations
        write(fout,"\n")
        write(fout, "##########################","\n")
        write(fout, "## ON-SITE CORRELATIONS ##","\n")
        write(fout, "##########################","\n","\n")

        write_correlations!(fout,model,container.onsite_corr,sim_params,Nbins)

        # write inter-site correlations
        write(fout,"\n")
        write(fout, "#############################","\n")
        write(fout, "## INTER-SITE CORRELATIONS ##","\n")
        write(fout, "#############################","\n","\n")

        write_correlations!(fout,model,container.intersite_corr,sim_params,Nbins)

        # write susceptibilities
        write(fout,"\n")
        write(fout, "######################","\n")
        write(fout, "## SUSCEPTIBILITIES ##","\n")
        write(fout, "######################","\n","\n")

        write_susceptibilities!(fout,model,container.susceptibility,sim_params,Nbins)

    end

    return nothing
end


######################
## PRIVATE FUNCTION ##
######################

"""
Write bond defintions to file.
"""
function write_bond_definitions!(fout,model::SSHModel{T1,T2,T3}) where {T1,T2,T3}

    bonds = model.bond_definitions

    write(fout,"######################\n")
    write(fout,"## BOND DEFINITIONS ##\n")
    write(fout,"######################\n\n")

    for (id, bond) in enumerate(bonds)

        write(fout,"Bond ID = ", string(id), "\n")
        write(fout,"name  = ",bond.name, "\n")
        write(fout,"t_avg = ", string(bond.t),  "\n")
        write(fout,"t_std = ", string(bond.σt), "\n")
        write(fout,"Initial Orbit = ", string(bond.o₁), "\n")
        write(fout,"Final Orbit   = ", string(bond.o₂), "\n")
        write(fout,"Displacement  = ", string(bond.v),  "\n", "\n")
    end

    return nothing
end

function write_bond_definitions!(fout,model::HolsteinModel{T1,T2,T3}) where {T1,T2,T3}

    bonds = model.bond_definitions

    write(fout,"######################\n")
    write(fout,"## BOND DEFINITIONS ##\n")
    write(fout,"######################\n\n")

    for (id, bond) in enumerate(bonds)

        write(fout,"Bond ID = ", string(id), "\n")
        write(fout,"t_avg = ", string(bond.t),  "\n")
        write(fout,"t_std = ", string(bond.σt), "\n")
        write(fout,"Initial Orbit = ", string(bond.o₁), "\n")
        write(fout,"Final Orbit   = ", string(bond.o₂), "\n")
        write(fout,"Displacement  = ", string(bond.v),  "\n", "\n")
    end

    return nothing
end

"""
Write Phonon definitions to file.
"""
function write_phonon_definitions!(fout,model::SSHModel{T1,T2,T3}) where {T1,T2,T3}

    bonds = model.bond_definitions

    write(fout,"############################\n")
    write(fout,"## SSH PHONON DEFINITIONS ##\n")
    write(fout,"############################\n\n")

    id = 0
    for bond in bonds

        if bond.has_phonon
            id += 1
            write(fout,"SSH Phonon ID = ", string(id), "\n")
            write(fout,"name = ",bond.name, "\n")
            write(fout,"t_avg = ", string(bond.t),  "\n")
            write(fout,"t_std = ", string(bond.σt), "\n")
            write(fout,"alpha_avg = ", string(bond.α),  "\n")
            write(fout,"alpha_std = ", string(bond.σα), "\n")
            write(fout,"alpha2_avg = ", string(bond.α₂),  "\n")
            write(fout,"alpha2_std = ", string(bond.σα₂), "\n")
            write(fout,"omega_avg = ", string(bond.ω),  "\n")
            write(fout,"omega_std = ", string(bond.σω), "\n")
            write(fout,"omega4_avg = ", string(bond.ω₄),  "\n")
            write(fout,"omega4_std = ", string(bond.σω₄), "\n")
            write(fout,"Initial Orbit = ", string(bond.o₁), "\n")
            write(fout,"Final Orbit   = ", string(bond.o₂), "\n")
            write(fout,"Displacement  = ", string(bond.v),  "\n", "\n")
        end
    end

    return nothing
end

function write_phonon_definitions!(fout,model::HolsteinModel{T1,T2,T3}) where {T1,T2,T3}

    write(fout,"#################################\n")
    write(fout,"## HOLSTEIN PHONON DEFINITIONS ##\n")
    write(fout,"#################################\n\n")

    L₁ = model.lattice.L1
    L₂ = model.lattice.L2
    L₃ = model.lattice.L3
    n  = model.lattice.unit_cell.norbits

    ω  = reshaped(model.ω,(n,L₁,L₂,L₃))
    ω₄ = reshaped(model.ω₄,(n,L₁,L₂,L₃))
    λ  = reshaped(model.λ,(n,L₁,L₂,L₃))
    λ₂ = reshaped(model.λ₂,(n,L₁,L₂,L₃))

    for o in 1:n

        write(fout,"Orbit = ", string(o), "\n")

        vals  = @view ω[o,:,:,:]
        avg   = mean(vals)
        if length(vals)>1
            stdev = std(vals)
        else
            stdev = 0.0
        end
        write(fout,"Omega_avg   = ", string(avg), "\n")
        write(fout,"Omega_std   = ", string(stdev), "\n")

        vals  = @view ω₄[o,:,:,:]
        avg   = mean(vals)
        if length(vals)>1
            stdev = std(vals)
        else
            stdev = 0.0
        end
        write(fout,"Omega4_avg  = ", string(avg), "\n")
        write(fout,"Omega4_std  = ", string(stdev), "\n")

        vals  = @view λ[o,:,:,:]
        avg   = mean(vals)
        if length(vals)>1
            stdev = std(vals)
        else
            stdev = 0.0
        end
        write(fout,"Lambda_avg  = ", string(avg), "\n")
        write(fout,"Lambda_std  = ", string(stdev), "\n")

        vals  = @view λ₂[o,:,:,:]
        avg   = mean(vals)
        if length(vals)>1
            stdev = std(vals)
        else
            stdev = 0.0
        end
        write(fout,"Lambda2_avg = ", string(avg), "\n")
        write(fout,"Lambda2_std = ", string(stdev), "\n","\n")
    end

    return nothing
end

"""
Write chemical potential defintions to file.
"""
function write_chemical_potential_definitions!(fout,model::AbstractModel{T1,T2,T3}) where {T1,T2,T3}

    bonds = model.bond_definitions

    write(fout,"#########################\n")
    write(fout,"## CHEMICAL POTENTIALS ##\n")
    write(fout,"#########################\n\n")

    L₁ = model.lattice.L1
    L₂ = model.lattice.L2
    L₃ = model.lattice.L3
    n  = model.lattice.unit_cell.norbits

    μ  = reshaped(model.μ,(n,L₁,L₂,L₃))

    for o in 1:n

        vals  = @view μ[o,:,:,:]
        avg   = mean(vals)
        if length(vals)>1
            stdev = std(vals)
        else
            stdev = 0.0
        end
        write(fout,"Orbit  = ", string(o), "\n")
        write(fout,"Mu_avg = ", string(avg), "\n")
        write(fout,"Mu_std = ", string(stdev), "\n", "\n")
    end

    return nothing
end

"""
Write global measurements to file.
"""
function write_global_measurements!(fout, model::AbstractModel{T1,T2,T3}, μ_tuner::MuTuner{T1}, container::Dict, sim_params::SimulationParameters, Nbins::Int) where {T1,T2,T3}

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
    binned_data = Dict(string(k)=>zeros(T2,Nbins) for k in keys(container))

    # container to contain statistics
    stats = Dict(string(k)=>zeros(T2,2) for k in keys(container))

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

    # calculate average and error bar for each measurement
    for measurement in keys(binned_data)

        # deal with chemical potential seperately due to μ-tuning algorithm
        if measurement=="mu"
            estimate_μ(μ_tuner)
            stats[measurement][1] = μ_tuner.μ_avg
            stats[measurement][2] = μ_tuner.μ_err
        else
            avg, err = mean_and_error(binned_data[measurement])
            stats[measurement][1] = avg
            stats[measurement][2] = err
        end
    end

    # calculate the compressibility
    β   = model.β
    N   = model.Nsites
    n   = stats["density"][1]
    Δn  = stats["density"][2]
    N²  = stats["Nsqr"][1]
    ΔN² = stats["Nsqr"][2]
    κ, Δκ = measure_κ(β,N,N²,ΔN²,n,Δn)
    stats["compressibility"] = [κ,Δκ]

    # write averaged statistics to file
    open(statsfn,"w") do statsout

        # header
        header = "global_meas avg error\n"
        write(fout,    header)
        write(statsout,header)

        # iterate over measurements
        for measurement in keys(stats)

            # calculate mean and standard deviation of mean
            avg = stats[measurement][1]
            err = stats[measurement][2]

            # write statistics to file
            line = measurement * @sprintf(" %.6f %.6f\n",avg,err)
            write(fout,    line)
            write(statsout,line)
        end
    end

    return nothing
end

"""
Write on-site measreuments to file.
"""
function write_onsite_measurements!(fout,model::AbstractModel{T1,T2,T3},μ_tuner::MuTuner{T1},container::NamedTuple,sim_params::SimulationParameters,Nbins::Int) where {T1,T2,T3}

    # number of site per unit cell
    nₛ = model.lattice.unit_cell.norbits::Int

    # filename containing global measurements
    datafn = joinpath(sim_params.datafolder, "onsite_measurements.out")

    # file statistics will be written to
    statsfn = joinpath(sim_params.datafolder, "onsite_measurements_stats.out")

    # number of measurements in data file
    Nmeas = sim_params.num_bins

    # calculate number of measurements per bin
    Nbins = min(Nmeas,Nbins)
    N     = div(Nmeas,Nbins)
    @assert Nmeas%Nbins==0

    # container to hold binned data
    binned_data = Dict(string(k)=>zeros(T2,Nbins,nₛ) for k in keys(container))

    # container to contain statistics
    stats = Dict(string(k)=>zeros(T2,2,nₛ) for k in keys(container))

    # open file with on-site measurement data
    open(datafn,"r") do fin

        # read in header
        header = readline(fin)

        # get column names
        columns = split(header,",")

        # iterate over lines of data
        for line in eachline(fin)

            # split line apart
            atoms = split(line,",")

            # get the measurement number
            nmeas = parse(Int,atoms[1])

            # get bin
            bin = div(nmeas-1,N) + 1

            # get the current orbital/site in unit cell
            orbit = parse(Int,atoms[2])

            # iterate over remain columns
            for i in 3:length(columns)

                # get the measurement associated with the column
                measurement = columns[i]

                # record measurement
                binned_data[measurement][bin,orbit] += parse(T2,atoms[i]) / N
            end
        end
    end

    # iterating over measurements
    for measurement in keys(stats)

        # deal with chemical potential seperately due to μ-tuning algorithm
        if measurement=="mu"
            μ  = reshaped(model.μ,(nₛ,div(model.Nsites,nₛ)))
            μ″ = mean(μ)
            μ′ = μ_tuner.μ_avg
            Δμ = μ_tuner.μ_err
            dμ = μ′-μ″
            for orbit in 1:nₛ
                μₒ = @view μ[orbit,:]
                stats[measurement][1,orbit] = mean(μₒ) + dμ
                stats[measurement][2,orbit] = Δμ
            end
        else
            # iterate over orbits
            for orbit in 1:nₛ

                # calculating average and error of measurement
                data     = @view binned_data[measurement][:,orbit]
                avg, err = mean_and_error(data)
                stats[measurement][1,orbit] = avg
                stats[measurement][2,orbit] = err
            end
        end
    end

    # write averaged statistics to file
    open(statsfn,"w") do statsout

        # construct header
        header = "measurement orbit avg error\n"
        write(fout,    header)
        write(statsout,header)

        # iterate over measurements
        for measurement in keys(stats)

            # iterate over orbits
            for orbit in 1:nₛ

                # write out measurement
                avg  = stats[measurement][1,orbit]
                err  = stats[measurement][2,orbit]
                line = measurement * @sprintf(" %d %.6f %.6f\n",orbit,avg,err)
                write(fout,    line)
                write(statsout,line)
            end
        end
    end

    return nothing
end

"""
Write on-site measreuments to file.
"""
function write_intersite_measurements!(fout,model::AbstractModel{T1,T2,T3},container::NamedTuple,sim_params::SimulationParameters,Nbins::Int) where {T1,T2,T3}

    # number of site per unit cell
    nᵥ = model.nbonds::Int

    # filename containing global measurements
    datafn = joinpath(sim_params.datafolder, "intersite_measurements.out")

    # file statistics will be written to
    statsfn = joinpath(sim_params.datafolder, "intersite_measurements_stats.out")

    # number of measurements in data file
    Nmeas = sim_params.num_bins

    # calculate number of measurements per bin
    Nbins = min(Nmeas,Nbins)
    N     = div(Nmeas,Nbins)
    @assert Nmeas%Nbins==0

    # container to hold binned data
    binned_data = Dict(string(k)=>zeros(T2,Nbins,nᵥ) for k in keys(container))

    # container to contain statistics
    stats = Dict(string(k)=>zeros(T2,2,nᵥ) for k in keys(container))

    # open file with on-site measurement data
    open(datafn,"r") do fin

        # read in header
        header = readline(fin)

        # get column names
        columns = split(header,",")

        # iterate over lines of data
        for line in eachline(fin)

            # split line apart
            atoms = split(line,",")

            # get the measurement number
            nmeas = parse(Int,atoms[1])

            # get bin
            bin = div(nmeas-1,N) + 1

            # get the current orbital/site in unit cell
            orbit = parse(Int,atoms[2])

            # iterate over remain columns
            for i in 3:length(columns)

                # get the measurement associated with the column
                measurement = columns[i]

                # record measurement
                binned_data[measurement][bin,orbit] += parse(T2,atoms[i]) / N
            end
        end
    end

    # iterating over measurements
    for measurement in keys(stats)

        # iterate over orbits
        for vector in 1:nᵥ

            # calculating average and error of measurement
            data     = @view binned_data[measurement][:,vector]
            avg, err = mean_and_error(data)
            stats[measurement][1,vector] = avg
            stats[measurement][2,vector] = err
        end
    end

    # write averaged statistics to file
    open(statsfn,"w") do statsout

        # construct header
        header = "measurement bond avg error\n"
        write(fout,    header)
        write(statsout,header)

        # iterate over measurements
        for measurement in keys(stats)

            # iterate over orbits
            for vector in 1:nᵥ

                # write out measurement
                avg  = stats[measurement][1,vector]
                err  = stats[measurement][2,vector]
                line = measurement * @sprintf(" %d %.6f %.6f\n",vector,avg,err)
                write(fout,    line)
                write(statsout,line)
            end
        end
    end

    return nothing
end

"""
Write correlations to file.
"""
function write_correlations!(fout,model::AbstractModel{T1,T2,T3},container::NamedTuple,sim_params::SimulationParameters,Nbins::Int) where {T1,T2,T3}

    # iterate over correlation functions
    for key in keys(container)
        measurement = string(key)
        position    = container[key].position
        momentum    = container[key].momentum
        write_correlation!(fout,measurement,model,position,sim_params,Nbins,true)
        write_correlation!(fout,measurement,model,momentum,sim_params,Nbins,false)
    end

    return nothing
end

"""
Write a correlation to file.
"""
function write_correlation!(fout,measurement::String,model::AbstractModel{T1,T2,T3},container::Array{Complex{T1},6},sim_params::SimulationParameters,Nbins::Int,is_position::Bool) where {T1,T2,T3}
    
    # get size of conatiner
    Lₜ, L₁, L₂, L₃, n, extra = size(container)

    # declare array to contained binned data
    binned_data = zeros(T1,Nbins,Lₜ,L₁,L₂,L₃,n,n)

    # number of measurements in data file
    Nmeas = sim_params.num_bins

    # calculate number of measurements per bin
    Nbins = min(Nmeas,Nbins)
    N     = div(Nmeas,Nbins)
    @assert Nmeas%Nbins==0

    # if position space data
    if is_position

        # filename containing global measurements
        datafn = joinpath(sim_params.datafolder, "$(measurement)_position.out")

        # file statistics will be written to
        statsfn = joinpath(sim_params.datafolder, "$(measurement)_position_stats.out")

        # header line
        header = "n1 n2 r3 r2 r1 tau $(measurement) error\n"
    
    # if momentum space data
    else

        # filename containing global measurements
        datafn = joinpath(sim_params.datafolder, "$(measurement)_momentum.out")

        # file statistics will be written to
        statsfn = joinpath(sim_params.datafolder, "$(measurement)_momentum_stats.out")

        # header line
        header = "n1 n2 k3 k2 k1 tau $(measurement) error\n"
    end

    # read in correlation data
    read_correlation!(binned_data, datafn, N)

    # open stats file
    open(statsfn,"w") do statsout

        # write header to file
        write(fout,header)
        write(statsout,header)

        # iterate over all measurements
        for n₁ in 1:n
            for n₂ in 1:n
                for l₃ in 1:L₃
                    for l₂ in 1:L₂
                        for l₁ in 1:L₁
                            for τ in 1:Lₜ
                                data = @view binned_data[:,τ,l₁,l₂,l₃,n₂,n₁]
                                # calculate average and error of measurement
                                avg, err = mean_and_error(data)
                                # write to file
                                line = @sprintf("%d %d %d %d %d %d %.6f %.6f\n",n₁,n₂,l₃-1,l₂-1,l₁-1,τ-1,avg,err)
                                write(fout,line)
                                write(statsout,line)
                            end
                        end
                    end
                end
            end
        end
    end

    write(fout,"\n")

    return nothing
end

"""
Read correlation data.
"""
function read_correlation!(binned_data::AbstractArray{T,7}, datafn::String, N::Int) where {T}

    # open data file
    open(datafn,"r") do fin

        # read in header
        datafile_header = readline(fin)

        # iterate over lines in data file
        for line in eachline(fin)

            # split line apart
            atoms = split(line,",")

            # get the measurement number
            nmeas = parse(Int,atoms[1])

            # get bin
            bin = div(nmeas-1,N) + 1
            
            # parse the data
            n₁ = parse(Int, atoms[2])
            n₂ = parse(Int, atoms[3])
            l₃ = parse(Int, atoms[4]) + 1
            l₂ = parse(Int, atoms[5]) + 1
            l₁ = parse(Int, atoms[6]) + 1
            τ  = parse(Int, atoms[7]) + 1
            v  = parse(T,   atoms[8])

            # record measurement
            binned_data[bin,τ,l₁,l₂,l₃,n₂,n₁] += v/N
        end
    end

    return nothing
end

"""
Write susceptibilities to file.
"""
function write_susceptibilities!(fout,model::AbstractModel{T1,T2,T3},container::NamedTuple,sim_params::SimulationParameters,Nbins::Int) where {T1,T2,T3}

    # iterate over correlation functions
    for key in keys(container)
        measurement = string(key)
        position    = container[key].position
        momentum    = container[key].momentum
        write_susceptibility!(fout,measurement,model,position,sim_params,Nbins,true)
        write_susceptibility!(fout,measurement,model,momentum,sim_params,Nbins,false)
    end

    return nothing
end

"""
Write a susceptibility to file.
"""
function write_susceptibility!(fout,measurement::String,model::AbstractModel{T1,T2,T3},container::Array{Complex{T1},5},sim_params::SimulationParameters,Nbins::Int,is_position::Bool) where {T1,T2,T3}
    
    # get size of conatiner
    L₁, L₂, L₃, n, extra = size(container)

    # declare array to contained binned data
    binned_data = zeros(T1,Nbins,L₁,L₂,L₃,n,n)

    # number of measurements in data file
    Nmeas = sim_params.num_bins

    # calculate number of measurements per bin
    Nbins = min(Nmeas,Nbins)
    N     = div(Nmeas,Nbins)
    @assert Nmeas%Nbins==0

    # if position space data
    if is_position

        # filename containing global measurements
        datafn = joinpath(sim_params.datafolder, "$(measurement)_position.out")

        # file statistics will be written to
        statsfn = joinpath(sim_params.datafolder, "$(measurement)_position_stats.out")

        # header line
        header = "n1 n2 r3 r2 r1 $(measurement) error\n"
    
    # if momentum space data
    else

        # filename containing global measurements
        datafn = joinpath(sim_params.datafolder, "$(measurement)_momentum.out")

        # file statistics will be written to
        statsfn = joinpath(sim_params.datafolder, "$(measurement)_momentum_stats.out")

        # header line
        header = "n1 n2 k3 k2 k1 $(measurement) error\n"
    end

    # read in susceptibility data
    read_susceptibility!(binned_data, datafn, N)

    # open stats file
    open(statsfn,"w") do statsout

        # write header to file
        write(fout,header)
        write(statsout,header)

        # iterate over all measurements
        for n₁ in 1:n
            for n₂ in 1:n
                for l₃ in 1:L₃
                    for l₂ in 1:L₂
                        for l₁ in 1:L₁
                            data = @view binned_data[:,l₁,l₂,l₃,n₂,n₁]
                            # calculate average and error of measurement
                            avg, err = mean_and_error(data)
                            # write to file
                            line = @sprintf("%d %d %d %d %d %.6f %.6f\n",n₁,n₂,l₃-1,l₂-1,l₁-1,avg,err)
                            write(fout,line)
                            write(statsout,line)
                        end
                    end
                end
            end
        end
    end

    write(fout,"\n")

    return nothing
end

"""
Read susceptibility data.
"""
function read_susceptibility!(binned_data::AbstractArray{T,6}, datafn::String, N::Int) where {T}

    # open data file
    open(datafn,"r") do fin

        # read in header
        datafile_header = readline(fin)

        # iterate over lines in data file
        for line in eachline(fin)

            # split line apart
            atoms = split(line,",")

            # get the measurement number
            nmeas = parse(Int,atoms[1])

            # get bin
            bin = div(nmeas-1,N) + 1
            
            # parse the data
            n₁ = parse(Int, atoms[2])
            n₂ = parse(Int, atoms[3])
            l₃ = parse(Int, atoms[4]) + 1
            l₂ = parse(Int, atoms[5]) + 1
            l₁ = parse(Int, atoms[6]) + 1
            v  = parse(T,   atoms[7])

            # record measurement
            binned_data[bin,l₁,l₂,l₃,n₂,n₁] += v/N
        end
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