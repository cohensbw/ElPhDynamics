module NonLocalMeasurements

using Printf
using FFTW
using LinearAlgebra

using ..Utilities: get_index, get_site, get_τ, θ, δ
using ..HolsteinModels: HolsteinModel
using ..SimulationParams: SimulationParameters
using ..GreensFunctions: EstimateGreensFunction, update!, measure_GΔ0, measure_GΔ0_GΔ0, measure_GΔΔ_G00, measure_GΔ0_G0Δ

export make_nonlocal_measurements!
export reset_nonlocal_measurements!
export process_nonlocal_measurements!
export construct_nonlocal_measurements_container
export initialize_nonlocal_measurement_files
export write_nonlocal_measurements


#########################################
## METHODS DEFINING MEASUREMENTS BELOW ##
#########################################

"""
Measure time-ordered single-particle electron Green's function ⟨cᵢ₊ᵣ(τ₂)⋅c⁺ᵢ(0)⟩ for 0≤τ<β
"""
function measure_Greens(estimator,l₁,l₂,l₃,o₁,o₂,τ)

    Gᵣ₀τ0 = measure_GΔ0(estimator,l₁,l₂,l₃,o₁,o₂,τ)
    return Gᵣ₀τ0
end


"""
Measure the density-density correlation function ⟨nᵢ₊ᵣ(τ)⋅nᵢ(0)⟩ for 0≤τ<β
"""
function measure_DenDen(estimator,l₁,l₂,l₃,o₁,o₂,τ)

    Gᵣ₀τ0       = measure_GΔ0(estimator,l₁,l₂,l₃,o₁,o₂,τ)    
    G₀₀00       = measure_GΔ0(estimator,0,0,0,o₁,o₁,0)
    Gᵣᵣττ       = measure_GΔ0(estimator,0,0,0,o₂,o₂,0)
    Gᵣᵣττ_G₀₀00 = measure_GΔΔ_G00(estimator,l₁,l₂,l₃,o₁,o₂,τ)
    Gᵣ₀τ0_G₀ᵣ0τ = measure_GΔ0_G0Δ(estimator,l₁,l₂,l₃,o₁,o₂,τ)
    δᵣ          = δ(l₁)*δ(l₂)*δ(l₃)*δ(o₁,o₂)
    nᵣτ_n₀0     = 4.0 * ( 1.0 - Gᵣᵣττ - G₀₀00 + Gᵣᵣττ_G₀₀00 + 0.5 * ( δᵣ*δ(τ)*Gᵣ₀τ0 - Gᵣ₀τ0_G₀ᵣ0τ ) )
    return nᵣτ_n₀0
end


"""
Measure pair Green's function ⟨Δᵢ₊ᵣ(τ) Δ⁺ᵢ(0)⟩ where Δᵢ(τ) = cᵢ₊(τ)cᵢ₋(τ) for 0≤τ<β
"""
function measure_PairGreens(estimator,l₁,l₂,l₃,o₁,o₂,τ)

    Gᵣ₀τ0_Gᵣ₀τ0 = measure_GΔ0_GΔ0(estimator,l₁,l₂,l₃,o₁,o₂,τ)
    return Gᵣ₀τ0_Gᵣ₀τ0
end


####################################################
## GENERATE FUNCTIONS TO MAKE MEASUREMENTS INSIDE ##
## LOOPS OVER SPACE AND TIME DISPLACEMENT VECTORE ##
####################################################

for measurement in [ :Greens , :DenDen , :PairGreens ]

    # constructing symbol for function name
    op = Symbol(:measure_,measurement,:!)

    # function to make measurement
    measure = Symbol(:measure_,measurement)

    @eval begin
        function $op(container::Array{Complex{T1},6}, Gr::EstimateGreensFunction{T1}) where {T1<:AbstractFloat,T2<:Number}

            # getting size of system
            Lτ      = size(container,1)
            norbits = size(container,2)
            L1      = size(container,4)
            L2      = size(container,5)
            L3      = size(container,6)

            # iterate over all relevant space-time displacement vectors
            for ΔL3 in 0:L3-1
                for ΔL2 in 0:L2-1
                    for ΔL1 in 0:L1-1
                        for orbit1 in 1:norbits
                            for orbit2 in 1:norbits
                                for τ in 0:Lτ-1
                                    container[ τ+1, orbit2, orbit1, ΔL1+1, ΔL2+1, ΔL3+1 ] += $measure(Gr,ΔL1,ΔL2,ΔL3,orbit1,orbit2,τ)
                                end
                            end
                        end
                    end
                end
            end
            return nothing
        end
    end
end


#######################################
## FUNCTIONS TO BE CALLED OUTSIDE OF ##
## THIS SCRIPT TO MAKE MEASUREMENTS  ##
#######################################

"""
Makes non-local measurements in real-space.
Each type of measurement is made for all possible imaginary time seperations and displacement vectors.
Measurements will be stored in arrays with 6 indices where the indices correspond to
`measurement[ΔL1+1, ΔL2+1, ΔL3+1, orbit2, orbit1, τ+1]`, where ΔLi is a displacement
in unit cells in the direction of the i'th lattice vector.
"""
function make_nonlocal_measurements!(container::NamedTuple, holstein::HolsteinModel{T1,T2}, Gr::EstimateGreensFunction{T1}) where {T1<:AbstractFloat,T2<:Number}

    # iterate over measurements
    for key in keys(container)
        if key==:Greens
            # Measure Electron Greens Function
            measure_Greens!(container.Greens,Gr)
        elseif key==:DenDen
            # Measure Density-Density Correlation Function
            measure_DenDen!(container.DenDen,Gr)
        elseif key==:PairGreens
            # Measure Pair Greens Function
            measure_PairGreens!(container.PairGreens,Gr)
        else
            error("The following key is not a valid measurment: ",String(key))
        end
    end

    return nothing
end


"""
Construct a dictionary for the real space and momentum space measurements,
where each measurement is a key in the dictionary and points to an array
where the measured values will be stored.
"""
function construct_nonlocal_measurements_container(holstein::HolsteinModel{T1,T2}, equaltime_meas::AbstractVector{String}, unequaltime_meas::AbstractVector{String})::Tuple{NamedTuple, NamedTuple} where {T1<:AbstractFloat,T2<:Number}

    # get size of lattice
    Lτ = holstein.Lτ
    L1 = holstein.lattice.L1
    L2 = holstein.lattice.L2
    L3 = holstein.lattice.L3
    norbits = holstein.lattice.unit_cell.norbits::Int

    container_rspace = Dict()
    container_kspace = Dict()

    # iterate over all unequal time measurements
    for measurement in unequaltime_meas
        # declare containers
        container_rspace[measurement] = zeros(Complex{T1},(Lτ,norbits,norbits,L1,L2,L3))
        container_kspace[measurement] = zeros(Complex{T1},(Lτ,norbits,norbits,L1,L2,L3))
    end

    # iterate over all equal time measurements
    for measurement in equaltime_meas
        # declare containers
        container_rspace[measurement] = zeros(Complex{T1},(1,norbits,norbits,L1,L2,L3))
        container_kspace[measurement] = zeros(Complex{T1},(1,norbits,norbits,L1,L2,L3))
    end

    # converting dictionary to named tuple
    container_rspace = NamedTuple{Tuple(Symbol.(keys(container_rspace)))}(values(container_rspace))
    container_kspace = NamedTuple{Tuple(Symbol.(keys(container_kspace)))}(values(container_kspace))

    return container_rspace, container_kspace
end


"""
Reset the arrays that contain the measurements to all zeros.
"""
function reset_nonlocal_measurements!(container::NamedTuple)

    for key in keys(container)
        fill!(container[key],0.0)
    end

    return nothing
end


"""
Process the real-space and momentum-space measurements.
This includes first performing the fourier transform to get the momentum space
values and then normalzing the measured values by the number of measurement made per bin.
"""
function process_nonlocal_measurements!(container_rspace::NamedTuple, container_kspace::NamedTuple, sim_params::SimulationParameters)

    # iterate over measurements
    for key in keys(container_kspace)

        vals_r = container_rspace[key]
        vals_k = container_kspace[key]

        # do fft from r to k space
        copyto!(vals_k,vals_r)
        fft!(vals_k, (4,5,6))

        # normalize measurements
        vals_r ./= sim_params.bin_size
        vals_k ./= sim_params.bin_size
    end

    return nothing
end


"""
Initializes files (including header) that each measurement will be written to.
"""
function initialize_nonlocal_measurement_files(container_rspace::NamedTuple, container_kspace::NamedTuple, sim_params::SimulationParameters)

    # iterating over real space measurements
    for key in keys(container_rspace)
        # get measurement string
        measurement = String(key)
        # Intializing data file
        open(sim_params.datafolder * measurement * "_r.out", "w") do file
            # writing file header
            write(file, "orbit1", ",", "orbit2", ",", "dL1",  ",", "dL2",  ",", "dL3",  ",", "tau", ",", measurement*"_r", "\n")
        end
    end

    # iterating over momentum-space measurements
    for key in keys(container_kspace)
        # get measurement string
        measurement = String(key)
        # Intializing data file
        open(sim_params.datafolder * measurement * "_k.out", "w") do file
            # writing file header
            write(file, "orbit1", ",", "orbit2", ",", "dL1",  ",", "dL2",  ",", "dL3",  ",", "tau", ",", measurement*"_k", "\n")
        end
    end

    return nothing
end


"""
Write non-local measurements to file. Each measurement gets its own file.
"""
function write_nonlocal_measurements(container::NamedTuple, sim_params::SimulationParameters, holstein::HolsteinModel{T1,T2}; real_space::Bool)  where {T1<:AbstractFloat,T2<:Number}

    # string to hold filename
    filename = ""

    # iterate over measurements
    for key in keys(container)

        # getting measurement string
        measurement = String(key)

        # getting a pointer to the array containing the measurements
        vals = container[key]

        # getting size of system
        Lτ      = size(vals,1)
        norbits = size(vals,2)
        L1      = size(vals,4)
        L2      = size(vals,5)
        L3      = size(vals,6)

        # constructing filename that measurements should be written to.
        # filename is adjusted according to whether the measurement is being
        # given in real-space or momentum-space.
        if real_space
            filename = sim_params.datafolder * measurement * "_r.out"
        else
            filename = sim_params.datafolder * measurement * "_k.out"
        end
        # opening file correspond to current measurement
        open( filename , "a" ) do file
            # iterating over possible pairs of orbitals
            for orbit1 in 1:norbits
                for orbit2 in 1:norbits
                    # iterating all displacement vectors in unit cells
                    for ΔL3 in 0:L3-1
                        for ΔL2 in 0:L2-1
                            for ΔL1 in 0:L1-1
                                # iterating over time slice
                                for τ in 0:Lτ-1
                                    # Getting value of measurement. 
                                    # Note that this averages over the two possible orderings for the orbitals
                                    meas  = real( vals[τ+1,orbit2,orbit1,ΔL1+1,ΔL2+1,ΔL3+1] )
                                    write(file, @sprintf("%d,%d,%d,%d,%d,%d,%.6f\n", orbit1, orbit2, ΔL1, ΔL2, ΔL3, τ, meas))
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    return nothing
end

end