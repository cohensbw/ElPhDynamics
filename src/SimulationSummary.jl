module SimulationSummary

using DataFrames
using CSV
using Statistics
using Printf

using ..LangevinSimulationParameters: SimulationParameters

export write_simulation_summary


"""
Writes a simulation summary file after a simulation completes.
"""
function write_simulation_summary(inputfn::String, sim_params::SimulationParameters{T}, simulation_time::T, measurement_time::T, write_time::T, iters::T, nbins::Int=10) where {T<:Number}
    
    # array to contain binned data
    bins = zeros(T,nbins)

    # getting name of output file
    outputfn = sim_params.foldername[1:end-1]*".out"

    # get list of all output files from simulation
    files = readdir(sim_params.datafolder)
    
    #############################
    ## READ DATA IN DATAFRAMES ##
    #############################
    
    # read non-local measurement data into dataframes
    df_rspace = Dict()
    df_kspace = Dict()
    for file in files
        if endswith(file,"_r.out")
            df_rspace[file[1:end-4]] = CSV.read(sim_params.datafolder*file;
                                       delim=",", header=1,
                                       types=[Int,Int,Int,Int,Int,Int,Float64])
        elseif endswith(file,"_k.out")
            df_kspace[file[1:end-4]] = CSV.read(sim_params.datafolder*file;
                                       delim=",", header=1,
                                       types=[Int,Int,Int,Int,Int,Int,Float64])
        end
    end
    
    # read local measurements data into dataframe
    df_local = CSV.read(sim_params.datafolder*"local_measurements.out",delim=",")
    
    # getting lattice size info
    norbits = maximum(df_rspace["Greens_r"].orbit1)
    L1 = maximum(df_rspace["Greens_r"].dL1)+1
    L2 = maximum(df_rspace["Greens_r"].dL2)+1
    L3 = maximum(df_rspace["Greens_r"].dL3)+1
    Lτ = maximum(df_rspace["Greens_r"].tau)+1
    
    ########################
    ## WRITE SUMMARY FILE ##
    ########################
    
    # writing summary file
    open(sim_params.datafolder*outputfn,"w") do outfile
        
        #################################
        ## COPY CONTENTS OF INPUT FILE ##
        #################################
        
        # copy contents of input file to output file
        write(outfile,"#########################\n")
        write(outfile,"## INPUT FILE CONTENTS ##\n")
        write(outfile,"#########################\n\n")
        
        open(inputfn,"r") do inputfn
            for line in eachline(inputfn)
                if !startswith(line,"#") && (line!="")
                    write(outfile,line,"\n")
                end
            end
        end
        
        ###########################
        ## WRITE SIMULATION INFO ##
        ###########################
        
        write(outfile,"\n#####################\n")
        write(outfile,  "## SIMULATION INFO ##\n")
        write(outfile,  "#####################\n\n")
        
        write(outfile, "Simulation Time (min) = ",  @sprintf("%.4f",simulation_time), "\n")
        write(outfile, "Measurement Time (min) = ", @sprintf("%.4f",measurement_time), "\n")
        write(outfile, "Write Time (min) = ",       @sprintf("%.4f",write_time), "\n")
        write(outfile, "Iterative Solver Steps = ", @sprintf("%.4f",iters), "\n")
        
        ##################################
        ## WRITE LOCAL MEASUREMENT DATA ##
        ##################################
        
        write(outfile,"\n########################\n")
        write(outfile,"## LOCAL MEASUREMENTS ##\n")
        write(outfile,"########################\n\n")
        
        # write header associated with local data
        write(outfile,"measurement","  ","orbit","  ","avg","  ","std","\n")
        
        # getting names of measurements
        cols = names(df_local)
        
        # iterate of orbitals
        for orbit in 1:norbits
            
            # select portion of dataframe corresponding to current orbital
            df_sel = df_local[df_local.orbit.==orbit,:]
            
            # iterate over measurements
            for col in cols[2:end]
                
                # calculating average and standard deviation of measurement
                avg, sd = binned_statistics(df_sel[:,col],bins)
                
                # writing measurement to file
                write( outfile, string(col), "  ", @sprintf("%d  %.6f  %.6f",orbit,avg,sd), "\n" )
            end
        end
        
        #################################################
        ## WRITE REAL-SPACE NON-LOCAL MEASUREMENT DATA ##
        #################################################
        
        write(outfile, "\n#############################################\n")
        write(outfile,   "## WRITE REAL-SPACE NON-LOCAL MEASUREMENTS ##\n")
        write(outfile,   "#############################################\n\n")
        
        write_nonlocal_data(outfile, df_rspace, norbits, L1, L2, L3, Lτ, bins)
        
        #################################################
        ## WRITE REAL-SPACE NON-LOCAL MEASUREMENT DATA ##
        #################################################
        
        write(outfile, "\n#################################################\n")
        write(outfile,   "## WRITE MOMENTUM-SPACE NON-LOCAL MEASUREMENTS ##\n")
        write(outfile,   "#################################################\n\n")
        
        write_nonlocal_data(outfile, df_kspace, norbits, L1, L2, L3, Lτ, bins)

    end
end

####################
## HELPER METHODS ##
####################

function write_nonlocal_data(outfile, dataframes, norbits, L1, L2, L3, Lτ, bins)
    
    # write header
    write(outfile, "orbit1  orbit2  dL1  dL2  dL3  tau")
    for key in keys(dataframes)
        write(outfile, "  ", key*"_avg", "  ", key*"_std")
    end
    write(outfile,"\n")

    # iterate over possible dispacements in time and space
    for orbit1 in 1:norbits
        for orbit2 in orbit1:norbits
            for dL3 in 0:L3-1
                for dL2 in 0:L2-1
                    for dL1 in 0:L1-1
                        for τ in 0:Lτ-1

                            # write displacement info to file
                            write(outfile,@sprintf("%d  %d  %d  %d  %d  %d",
                                                   orbit1,orbit2,dL1,dL2,dL3,τ))

                            # iteate over different measurements
                            for key in keys(dataframes)

                                # get pointer to relevant df
                                df = dataframes[key]

                                # retrieve relevant data
                                data = @view df[(df.orbit1.==orbit1).&
                                                (df.orbit2.==orbit2).&
                                                (df.dL3.==dL3).&
                                                (df.dL2.==dL2).&
                                                (df.dL1.==dL1).&
                                                (df.tau.==τ), end]

                                # calculate average and standard devation
                                avg, sd = binned_statistics(data,bins)

                                # write averaged info to file
                                write( outfile, @sprintf("  %.6f  %.6f", avg, sd) )
                            end
                            # end the line
                            write(outfile,"\n")
                        end
                    end
                end
            end
        end
    end
end

"""
Calculates the average and binned standard deviation of a set of data.
The number of bins used is equal to the length of the preallocated `bins` vector
passed to the function.
"""
function binned_statistics(data::AbstractVector{T},bins::Vector{T})::Tuple{T,T} where {T<:Number}
    
    N = length(data)
    n = length(bins)
    @assert length(data)%length(bins)==0
    binsize = div(N,n)
    bins .= 0
    for bin in 1:n
        for i in 1:binsize
            bins[bin] += data[i+(bin-1)*binsize]
        end
        bins[bin] /= binsize
    end
    avg = mean(bins)
    return avg, std(bins,corrected=true,mean=avg)/sqrt(n)
end

end