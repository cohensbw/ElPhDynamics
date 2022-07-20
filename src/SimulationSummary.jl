module SimulationSummary

using Statistics
using Serialization
using Printf
using Parameters
using TOML
using CSV

using ..SimulationParams: SimulationParameters
using ..Models: AbstractBond, SSHBond, Bond
using ..Models: AbstractModel, SSHModel, HolsteinModel, write_phonons!
using ..Models: write_phonons!, write_M_matrix!
using ..Measurements: measure_κ
using ..MuFinder: MuTuner, update_μ!, estimate_μ
using ..Utilities: reshaped

export write_simulation_summary!

"""
Write results of simulation to simulation summary file.
"""
function write_simulation_summary!(datafolder::String,Nbins::Int=10)

    # unpack last checkpoint from simulation
    chkpnt = deserialize(joinpath(datafolder,"checkpoint.jls"))
    @unpack model, μ_tuner, sim_stats = chkpnt

    # data foldername
    foldername = splitdir(datafolder)[end]

    # read in config file
    files    = readdir(datafolder)
    tomls    = findall(f -> endswith(f, r"\.toml|\.TOML"), files)
    tomlfile = joinpath(datafolder, files[tomls[1]])
    input    = TOML.parsefile(tomlfile)

    # write final phonon config to file
    write_phonons!(model,joinpath(datafolder,"$(foldername)_config.out"))

    # write M matrix to file
    if haskey(input["simulation"],"write_M_matrix")
        if input["simulation"]["write_M_matrix"]
            write_M_matrix!(model, joinpath(datafolder,"$(foldername)_matrix.out"))
        end
    end

    # construct filename for summary file
    filename = joinpath(datafolder, "$(foldername)_summary.out")

    # open summary file
    open(filename,"w") do fout

        # copy contents of input file to output file
        write(fout,"#########################\n")
        write(fout,"## INPUT FILE CONTENTS ##\n")
        write(fout,"#########################\n\n")

        TOML.print(fout, input)
        write(fout,"\n")

        # write bond defintions to summary file
        write(fout,"######################\n")
        write(fout,"## BOND DEFINITIONS ##\n")
        write(fout,"######################\n\n")

        write_bond_definitions!(fout,model)

        # write phonon definitions
        write(fout,"########################\n")
        write(fout,"## PHONON DEFINITIONS ##\n")
        write(fout,"########################\n\n")

        write_phonon_definitions!(fout,model)

        # write chemical potential defintions
        write(fout,"#########################\n")
        write(fout,"## CHEMICAL POTENTIALS ##\n")
        write(fout,"#########################\n\n")

        write_chemical_potential_definitions!(fout,model)
        
        # write simulation statistics to file
        write(fout, "#####################","\n")
        write(fout, "## SIMULATION INFO ##","\n")
        write(fout, "#####################","\n","\n")

        total_time = sim_stats["simulation_time"] + sim_stats["measurement_time"] + sim_stats["write_time"]
        @printf fout "Total Time (min)        = %.8f\n" total_time
        @printf fout "Simulation Time (min)   = %.8f\n" sim_stats["simulation_time"]
        @printf fout "Measurement Time (min)  = %.8f\n" sim_stats["measurement_time"]
        @printf fout "Write Time (min)        = %.8f\n" sim_stats["write_time"]
        @printf fout "Iterative Solver Steps  = %.8f\n" sim_stats["iters"]
        @printf fout "Acceptance Rate         = %.8f\n" sim_stats["acceptance_rate"]
        @printf fout "Reflect Acceptance Rate = %.8f\n" sim_stats["reflect_acceptance_rate"]
        @printf fout "Swap Acceptance Rate    = %.8f\n" sim_stats["swap_acceptance_rate"]

        # write global measurements to file
        write(fout,"\n")
        write(fout, "#########################","\n")
        write(fout, "## GLOBAL MEASUREMENTS ##","\n")
        write(fout, "#########################","\n","\n")

        write_global_measurements!(fout,model,μ_tuner,datafolder,Nbins)

        # write global measurements to file
        write(fout, "\n")
        write(fout, "##########################","\n")
        write(fout, "## ON-SITE MEASUREMENTS ##","\n")
        write(fout, "##########################","\n","\n")

        write_onsite_measurements!(fout,model,μ_tuner,datafolder,Nbins)

        # write global measurements to file
        write(fout,"\n")
        write(fout, "#############################","\n")
        write(fout, "## INTER-SITE MEASUREMENTS ##","\n")
        write(fout, "#############################","\n","\n")

        write_intersite_measurements!(fout,model,datafolder,Nbins)

        # write susceptibilities
        write(fout,"\n")
        write(fout, "######################","\n")
        write(fout, "## SUSCEPTIBILITIES ##","\n")
        write(fout, "######################","\n","\n")

        write_susceptibilities!(fout,model,datafolder,Nbins)

        # write on-site correlations
        write(fout,"\n")
        write(fout, "##################","\n")
        write(fout, "## CORRELATIONS ##","\n")
        write(fout, "##################","\n","\n")

        write_correlations!(fout,model,datafolder,Nbins)
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

    for (id, bond) in enumerate(bonds)

        write(fout,"Bond ID       = ", string(id), "\n")
        write(fout,"name          = ",bond.name, "\n")
        write(fout,"t_avg         = ", string(bond.t),  "\n")
        write(fout,"t_std         = ", string(bond.σt), "\n")
        write(fout,"Initial Orbit = ", string(bond.o₁), "\n")
        write(fout,"Final Orbit   = ", string(bond.o₂), "\n")
        write(fout,"Displacement  = ", string(bond.v),  "\n", "\n")
    end

    return nothing
end

function write_bond_definitions!(fout,model::HolsteinModel{T1,T2,T3}) where {T1,T2,T3}

    bonds = model.bond_definitions

    for (id, bond) in enumerate(bonds)

        write(fout,"Bond ID       = ", string(id), "\n")
        write(fout,"t_avg         = ", string(bond.t),  "\n")
        write(fout,"t_std         = ", string(bond.σt), "\n")
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
function write_global_measurements!(fout, model::AbstractModel{T1,T2,T3}, μ_tuner::MuTuner{T1}, datafolder::String, Nbins::Int) where {T1,T2,T3}

    # file statistics will be written to
    statsfn = joinpath(datafolder, "global_measurements_stats.out")

    # data folder
    datafolder = joinpath(datafolder,"global_measurements_f")

    # get files in data folder
    files = readdir(datafolder,join=true)

    # number of data files
    Nmeas = length(files)

    # calculate number of measurements per bin
    Nbins = min(Nmeas,Nbins)
    N     = div(Nmeas,Nbins)
    @assert Nmeas%Nbins==0

    # get measurements
    measurements = String[]
    open(files[1],"r") do fin
        for line in readlines(fin)
            atoms = split(line)
            push!( measurements , atoms[1] )
        end
    end

    # container to hold binned data
    binned_data = Dict(string(k)=>zeros(T1,Nbins) for k in measurements)

    # container to contain statistics
    stats = Dict(string(k)=>zeros(T1,2) for k in measurements)

    # iterate over files in data folder
    for (i,file) in enumerate(files)
        # open file
        open(file,"r") do fin
            # iterate over each line in data file
            for line in eachline(fin)
                # split the line at white space
                atoms = split(line)
                # get the bin
                bin   = div(i-1,N)+1
                # record measurement
                binned_data[atoms[1]][bin] += parse(T1,atoms[2]) / N
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
            @printf fout     "%s %.8f %.8f\n" measurement avg err
            @printf statsout "%s %.8f %.8f\n" measurement avg err
        end
    end

    return nothing
end

"""
Write on-site measreuments to file.
"""
function write_onsite_measurements!(fout,model::AbstractModel{T1,T2,T3},μ_tuner::MuTuner{T1},datafolder::String,Nbins::Int) where {T1,T2,T3}

    # number of site per unit cell
    nₛ = model.lattice.unit_cell.norbits::Int

    # data folder
    folder = joinpath(datafolder, "onsite_measurements_f")

    # get files containing data
    files = readdir(folder,join=true)

    # number of data files
    Nmeas = length(files)

    # file statistics will be written to
    statsfn = joinpath(datafolder, "onsite_measurements_stats.out")

    # calculate number of measurements per bin
    Nbins = min(Nmeas,Nbins)
    N     = div(Nmeas,Nbins)
    @assert Nmeas%Nbins==0

    # get measurements
    measurements = String[]
    open(files[1],"r") do fin
        header = readline(fin)
        for line in readlines(fin)
            push!( measurements , split(line)[1] )
        end
    end
    measurements = unique(measurements)

    # container to hold binned data
    binned_data = Dict(string(k)=>zeros(T1,Nbins,nₛ) for k in measurements)

    # container to contain statistics
    stats = Dict(string(k)=>zeros(T1,2,nₛ) for k in measurements)

    # iterate over data files
    for (i,file) in enumerate(files)
        bin = div(i-1,N) + 1
        open(file,"r") do fin
            header = readline(fin)
            for line in eachline(fin)
                atoms = split(line)
                meas  = atoms[1]
                orbit = parse(Int,atoms[2])
                val   = parse(T1,atoms[3])
                binned_data[meas][bin,orbit] += val / N
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
            # calculating average and error for on-site energy of each orbital
            for orbit in 1:nₛ
                μ₀ = @view μ[orbit,:]
                stats[measurement][1,orbit] = mean(μ₀) + dμ
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
                @printf fout     "%s %d %.6f %.6f\n" measurement orbit avg err
                @printf statsout "%s %d %.6f %.6f\n" measurement orbit avg err
            end
        end
    end

    return nothing
end

"""
Write on-site measreuments to file.
"""
function write_intersite_measurements!(fout,model::AbstractModel{T1,T2,T3},datafolder::String,Nbins::Int) where {T1,T2,T3}

    # number 
    nᵥ = model.nbonds

    # data folder
    folder = joinpath(datafolder, "intersite_measurements_f")

    # get files containing data
    files = readdir(folder,join=true)

    # number of data files
    Nmeas = length(files)

    # file statistics will be written to
    statsfn = joinpath(datafolder, "intersite_measurements_stats.out")

    # calculate number of measurements per bin
    Nbins = min(Nmeas,Nbins)
    N     = div(Nmeas,Nbins)
    @assert Nmeas%Nbins==0

    # get measurements
    measurements = String[]
    open(files[1],"r") do fin
        header = readline(fin)
        for line in readlines(fin)
            push!( measurements , split(line)[1] )
        end
    end
    measurements = unique(measurements)

    # container to hold binned data
    binned_data = Dict(string(k)=>zeros(T1,Nbins,nᵥ) for k in measurements)

    # container to contain statistics
    stats = Dict(string(k)=>zeros(T1,2,nᵥ) for k in measurements)

    # iterate over data files
    for (i,file) in enumerate(files)
        bin = div(i-1,N) + 1
        open(file,"r") do fin
            header = readline(fin)
            for line in eachline(fin)
                atoms = split(line)
                meas  = atoms[1]
                bond  = parse(Int,atoms[2])
                val   = parse(T1,atoms[3])
                binned_data[meas][bin,bond] += val / N
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
            for b in 1:nᵥ

                # write out measurement
                avg  = stats[measurement][1,b]
                err  = stats[measurement][2,b]
                @printf fout     "%s %d %.6f %.6f\n" measurement b avg err
                @printf statsout "%s %d %.6f %.6f\n" measurement b avg err
            end
        end
    end

    return nothing
end

"""
Write correlations to file.
"""
function write_correlations!(fout,model::AbstractModel,datafolder::String,Nbins::Int)

    # get contents of data folder
    contents = readdir(datafolder)
    # iterate of names in data folder
    for name in contents
        # if name corresponds to folder
        if endswith(name,"_f")
            # if name of directory starts with capital letter
            if isuppercase(name[1])
                # split name by underscore
                atoms = split(name,"_")
                # if a correlation function
                if !endswith(atoms[1],"Susc")
                    # name of measurement
                    measurement = string(atoms[1])
                    # position or momentum space correlation function
                    space = string(atoms[2])
                    # process data
                    write_correlation!(fout,model,measurement,space,Nbins,datafolder)
                end
            end
        end
    end

    return nothing
end

"""
Write a correlation to file.
"""
function write_correlation!(fout,model::AbstractModel{T},measurement::String,space::String,Nbins::Int,datafolder::String) where {T<:AbstractFloat}    

    # data folder
    folder = joinpath(datafolder,"$(measurement)_$(space)_f")
    files  = readdir(folder,join=true)

    # read info in from key file
    Lₜ      = 1
    pairs   = Vector{Tuple{Int,Int}}()
    header  = ""
    keyfile = joinpath(folder,"$(measurement)_$(space)_key.out")
    open(keyfile,"r") do fin
        # construct header for stats file, determining whether to label first two colmns
        # "orbit1 orbit2" or "bond1 bond2" depending on contents of key file
        keyheader = readline(fin)
        atoms     = split(keyheader)
        if space == "position"
            header = "$(atoms[2]) $(atoms[3]) r3 r2 r1 tau $(measurement)_real $(measurement)_imag error_real error_imag\n"
        else
            header = "$(atoms[2]) $(atoms[3]) k3 k2 k1 tau $(measurement)_real $(measurement)_imag error_real error_imag\n"
        end
        # get unique pairs of orbitals/bonds the correlation function was measured for
        # and determine if measurement was time dependent
        for line in eachline(fin)
            atoms = split(line)
            p     = ( parse(Int,atoms[2]) , parse(Int,atoms[3]) )
            if !(p in pairs)
                push!(pairs,p)
            end
            τ  = parse(Int,atoms[7])
            Lₜ = max(τ+1,Lₜ)
        end
    end

    # get number of pairs of orbitals/bonds
    nₚ = length(pairs)

    # get lattice size
    L₁ = model.lattice.L1::Int
    L₂ = model.lattice.L2::Int
    L₃ = model.lattice.L3::Int
    V  = L₁*L₂*L₃*nₚ*Lₜ

    # number of measurements
    Nmeas = length(files)-1

    # calculate number of measurements per bin
    Nbins = min(Nmeas,Nbins)
    N     = div(Nmeas,Nbins)
    @assert Nmeas % Nbins == 0

    # file statistics will be written to
    statsfn = joinpath(datafolder, "$(measurement)_$(space)_stats.out")

    # declare container for binned data
    container  = zeros(Complex{T},Nbins,Lₜ,L₁,L₂,L₃,nₚ)
    container′ = reshaped(container,Nbins,V)

    # iterate over number of files
    for m in 1:Nmeas
        # get bin number
        bin = div(m-1,N) + 1
        # iterate of rows in current data file
        for row in CSV.Rows(files[m], header=1, types=[Int,T,T], delim=" ", reusebuffer=true)
            # calculate binned averages
            index = row[1]
            val   = row[2] + im*row[3]
            container′[bin,index] += val / N
        end
    end
    
    # write stats file
    open(statsfn,"w") do statsout
        # write header
        write(fout,     header)
        write(statsout, header)
        # counter to indicate what chunk of data file to read
        n = 0
        # iterate of all position or momentum space variable
        for p in 1:nₚ
            n₁ = pairs[p][1]
            n₂ = pairs[p][2]
            for l₃ in 0:L₃-1
                for l₂ in 0:L₂-1
                    for l₁ in 0:L₁-1
                        # write averaged stats to file
                        for τ in 0:Lₜ-1
                            data     = @view container[:,τ+1,l₁+1,l₂+1,l₃+1,p]
                            avg, err = mean_and_error(data)
                            @printf fout     "%d %d %d %d %d %d %.8f %.8f %.8f %.8f\n" n₁ n₂ l₃ l₂ l₁ τ real(avg) imag(avg) real(err) imag(err)
                            @printf statsout "%d %d %d %d %d %d %.8f %.8f %.8f %.8f\n" n₁ n₂ l₃ l₂ l₁ τ real(avg) imag(avg) real(err) imag(err)
                        end
                        # increment to next chunk of file
                        n += 1
                    end
                end
            end
        end
    end

    write(fout,"\n")

    return nothing
end

"""
Write susceptibilities to file.
"""
function write_susceptibilities!(fout,model::AbstractModel, datafolder::String,Nbins::Int)

    # get contents of data folder
    contents = readdir(datafolder)
    # iterate of names in data folder
    for name in contents
        # if name corresponds to folder
        if endswith(name,"_f")
            # if name of directory starts with capital letter
            if isuppercase(name[1])
                # split name by underscore
                atoms = split(name,"_")
                # if a correlation function
                if endswith(atoms[1],"Susc")
                    # name of measurement
                    measurement = string(atoms[1])
                    # position or momentum space correlation function
                    space = string(atoms[2])
                    # process data
                    write_susceptibility!(fout,model,measurement,space,Nbins,datafolder)
                end
            end
        end
    end

    return nothing
end

"""
Write a susceptibility to file.
"""
function write_susceptibility!(fout,model::AbstractModel{T},measurement::String,space::String,Nbins::Int,datafolder::String) where {T<:AbstractFloat} 
    
    # data folder
    folder = joinpath(datafolder,"$(measurement)_$(space)_f")
    files  = readdir(folder,join=true)

    # read info in from key file
    pairs   = Vector{Tuple{Int,Int}}()
    header  = ""
    keyfile = joinpath(folder,"$(measurement)_$(space)_key.out")
    open(keyfile,"r") do fin
        # construct header for stats file, determining whether to label first two colmns
        # "orbit1 orbit2" or "bond1 bond2" depending on contents of key file
        keyheader = readline(fin)
        atoms     = split(keyheader)
        if space == "position"
            header = "$(atoms[2]) $(atoms[3]) r3 r2 r1 $(measurement)_real $(measurement)_imag error_real error_imag\n"
        else
            header = "$(atoms[2]) $(atoms[3]) k3 k2 k1 $(measurement)_real $(measurement)_imag error_real error_imag\n"
        end
        # get unique pairs of orbitals/bonds the correlation function was measured for
        for line in eachline(fin)
            atoms = split(line)
            p     = ( parse(Int,atoms[2]) , parse(Int,atoms[3]) )
            if !(p in pairs)
                push!(pairs,p)
            end
        end
    end

    # get number of pairs of orbitals/bonds
    nₚ = length(pairs)

    # get lattice size
    L₁ = model.lattice.L1::Int
    L₂ = model.lattice.L2::Int
    L₃ = model.lattice.L3::Int
    V  = L₁*L₂*L₃*nₚ

    # number of measurements
    Nmeas = length(files)-1

    # calculate number of measurements per bin
    Nbins = min(Nmeas,Nbins)
    N     = div(Nmeas,Nbins)
    @assert Nmeas % Nbins == 0

    # file statistics will be written to
    statsfn = joinpath(datafolder, "$(measurement)_$(space)_stats.out")

    # container to hold binned data
    container  = zeros(Complex{T},Nbins,L₁,L₂,L₃,nₚ)
    container′ = reshaped(container,Nbins,V)

    # iterate over files
    for m in 1:Nmeas
        # calculate bin
        bin = div(m-1,N) + 1
        # iterate over rows of csv file
        for row in CSV.Rows(files[m], types=[Int,T,T], reusebuffer=true, delim=' ')
            index = row[1]
            val   = row[2] + im*row[3]
            container′[bin,index] += val / N
        end
    end

    # write avg and error of measurements to file
    open(statsfn,"w") do statsout
        write(statsout,header)
        write(fout,    header)
        for p in 1:nₚ
            n₁ = pairs[p][1]
            n₂ = pairs[p][2]
            for l₃ in 0:L₃-1
                for l₂ in 0:L₂-1
                    for l₁ in 0:L₁-1
                        vals     = @view container[:,l₁+1,l₂+1,l₃+1,p]
                        avg, err = mean_and_error(vals)
                        @printf fout     "%d %d %d %d %d %.8f %.8f %.8f %.8f\n" n₁ n₂ l₃ l₂ l₁ real(avg) imag(avg) real(err) imag(err)
                        @printf statsout "%d %d %d %d %d %.8f %.8f %.8f %.8f\n" n₁ n₂ l₃ l₂ l₁ real(avg) imag(avg) real(err) imag(err)
                    end
                end
            end
        end
    end

    write(fout,"\n")

    return nothing
end

"""
Calculate mean and standard deviation of the mean of a set of data.
"""
function mean_and_error(v::AbstractArray{T})::Tuple{T,T} where {T<:AbstractFloat}
    
    N   = length(v)
    avg = mean(v)
    err = std(v,corrected=true,mean=avg)/sqrt(N)

    return avg, err
end

function mean_and_error(v::AbstractArray{T})::Tuple{T,T} where {T<:Complex}
    
    N        = length(v)
    avg      = mean(v)
    err_real = std(real.(v), corrected=true, mean=real(avg)) / sqrt(N)
    err_imag = std(imag.(v), corrected=true, mean=imag(avg)) / sqrt(N)

    return avg, err_real + im*err_imag
end

end