module NonLocalMeasurements

using Printf
using Langevin.HolsteinModels: HolsteinModel, get_index, get_site, get_τ
using Langevin.LangevinSimulationParameters: SimulationParameters
using Langevin.GreensFunctions: EstimateGreensFunction, update!, estimate, estimate_time_ordered
using Langevin.FourierTransforms: fourier_transform!

export make_nonlocal_measurements!
export reset_nonlocal_measurements!
export process_nonlocal_measurements!
export construct_nonlocal_measurements_container
export initialize_nonlocal_measurements_file
export write_nonlocal_measurements

"""
Makes non-local measurements. Each type of measurement is made for all possible imaginary time seperations
and in terms of unit cells, but for a single fixed pairing of orbital types.
Measurements will be stored in rank 4 tensors where the indices correspond to
`measurement[ΔL1+1, ΔL2+1, ΔL3+1, orbit2, orbit1, τ+1]`, where ΔLi is a displacement
 in unit cells in the direction of the i'th lattice vector.
"""
function make_nonlocal_measurements!(container::Dict{String,Array{T1,6}}, holstein::HolsteinModel{T1,T2}, Gr1::EstimateGreensFunction{T1}, Gr2::EstimateGreensFunction{T2}) where {T1<:AbstractFloat,T2<:Number}
    
    # update the stochastic estimates of the green's functions
    update!(Gr1,holstein)
    update!(Gr2,holstein)

    # lattice object
    lattice = holstein.lattice

    # array containing sets of translationally equivalent pairs of sites in lattice
    sets = holstein.trans_equiv_sets

    # number of measurements associated with given displacement vector
    # and time seperation 
    normalization = div(lattice.nsites,lattice.norbits) * holstein.Lτ

    # initialize all quantities being measured to zero to zero
    greens .= 0.0

    # iterate over time seperations
    for τ in 0:holstein.Lτ-1
        # iterating over all possible parings of orbitals
        for orbit1 in 1:lattice.norbits
            for orbit2 in 1:lattice.norbits
                # iterating over all displacements vectors defined in terms
                # of unit cells in the direction of each lattice vector.
                for ΔL3 in 0:lattice.L3-1
                    for ΔL2 in 0:lattice.L2-1
                        for ΔL1 in 0:lattice.L1-1
                            # iterating over pairs of sites corresponding to current displacement vector
                            for pair in 1:numorbits

                                # getting current pair of sites associated with specified
                                # displacement vector 
                                j = sets[1,pair,ΔL1+1,ΔL2+1,ΔL3+1,orbit2,orbit1]
                                i = sets[2,pair,ΔL1+1,ΔL2+1,ΔL3+1,orbit2,orbit1]

                                # iterating over time slices
                                for τ₁ in 1:holstein.Lτ

                                    # NOTE: We are calculating measurements for a displacement vector
                                    # defined as r=i-j (j==>i) and imaginary time 0⩽τ<β.

                                    # getting τ₂=τ₁+τ accounting for boundary conditions
                                    τ₂ = (τ₁+τ-1)%Gr.Lτ+1

                                    # getting β-τ
                                    βmτ = Gr.Lτ-τ

                                    # PRIMARY GREENS FUNCTION!
                                    # Gᵢⱼ(τ) = ⟨cᵢ(τ)⋅c⁺ⱼ(0)⟩ = ⟨T⋅cᵢ(τ+τ₁)⋅c⁺ⱼ(τ₁)⟩ for 0⩽τ<β.
                                    # Getting two stochastic estimates.
                                    Grτᵢⱼ1 = estimate_time_ordered(Gr1,i,j,τ,τ₁) 
                                    Grτᵢⱼ2 = estimate_time_ordered(Gr2,i,j,τ,τ₁)

                                    # Gⱼᵢ(τ) = ⟨cⱼ(τ)⋅c⁺ᵢ(0)⟩ = ⟨T⋅cⱼ(τ+τ₁)⋅c⁺ᵢ(τ₁)⟩ for 0⩽τ<β.
                                    # Getting two stochastic estimates.
                                    Grτⱼᵢ1 = estimate_time_ordered(Gr1,j,i,τ,τ₁) 
                                    Grτⱼᵢ2 = estimate_time_ordered(Gr2,j,i,τ,τ₁)

                                    # Gᵢⱼ(β-τ) = ⟨cᵢ(β-τ)⋅c⁺ⱼ(0)⟩ = ⟨T⋅cᵢ(β-τ+τ₁)⋅c⁺ⱼ(τ₁)⟩ for 0⩽τ<β.
                                    # Getting two stochastic estimates.
                                    Grβmτᵢⱼ1 = estimate_time_ordered(Gr1,i,j,βmτ,τ₁) 
                                    Grβmτᵢⱼ2 = estimate_time_ordered(Gr2,i,j,βmτ,τ₁)

                                    # Gⱼᵢ(β-τ) = ⟨cⱼ(β-τ)⋅c⁺ᵢ(0)⟩ = ⟨T⋅cⱼ(β-τ+τ₁)⋅c⁺ᵢ(τ₁)⟩ for 0⩽τ<β.
                                    # Getting two stochastic estimates.
                                    Grβmτⱼᵢ1 = estimate_time_ordered(Gr1,j,i,βmτ,τ₁) 
                                    Grβmτⱼᵢ2 = estimate_time_ordered(Gr2,j,i,βmτ,τ₁)

                                    # Gᵢᵢ(0) = ⟨cᵢ(0)⋅c⁺ᵢ(0)⟩ = ⟨cᵢ(τ+τ₁)⋅c⁺ᵢ(τ+τ₁)⟩ = ⟨cᵢ(τ₂)⋅c⁺ᵢ(τ₂)⟩
                                    # Getting two stochastic estimates.
                                    Gr0ᵢᵢ1 = estimate(Gr1,i,i,τ₂,τ₂)
                                    Gr0ᵢᵢ2 = estimate(Gr2,i,i,τ₂,τ₂)
                                    
                                    # Gⱼⱼ(0) = ⟨cⱼ(0)⋅c⁺ⱼ(0)⟩ = ⟨cⱼ(τ₁)⋅c⁺ⱼ(τ₁)⟩.
                                    # Getting two stochastic estimates.
                                    Gr0ⱼⱼ1 = estimate(Gr1,j,j,τ₁,τ₁)
                                    Gr0ⱼⱼ2 = estimate(Gr2,j,j,τ₁,τ₁)

                                    # measuring electron green's function
                                    container["greens"][ ΔL1+1, ΔL2+1, ΔL3+1, orbit2, orbit1, τ+1 ] += measure_greens( Grτᵢⱼ1 , Grτᵢⱼ2 )/normalization
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


function construct_nonlocal_measurements_container(holstein::HolsteinModel{T1,T2})::Tuple{Dict{String,Array{T1,6}},Dict{String,Array{T1,6}}} where {T1<:AbstractFloat,T2<:Number}

    lattice = holstein.lattice
    container_rspace = Dict()
    container_kspace = Dict()
    # ierate over all measurements to be made
    for meas in ("greens",)
        container_rspace[meas*"_r"] = zeros(T1,(lattice.L1,lattice.L2,lattice.L3,lattice.norbits,lattice.norbits,holstein.Lτ))
        container_rspace[meas*"_k"] = zeros(T1,(lattice.L1,lattice.L2,lattice.L3,lattice.norbits,lattice.norbits,holstein.Lτ))
    end
    return container_rspace, container_kspace
end


function reset_nonlocal_measurements!(container::Dict{String,Array{T,6}}) where {T<:AbstractFloat}

    for key in keys(container)
        container[key] .= 0.0
    end
end


function process_nonlocal_measurements!(container_R::Dict{String,Array{T,6}}, container_K::Dict{String,Array{T,6}}, sim_params::SimulationParameters{T}, ft_coeff::Array{Complex{T},6}) where {T<:AbstractFloat}

    # first, normalize the position-space measurements by the number of measurements per bin
    for key in keys(container_R)
        container_R[key] ./= sim_params.bin_size
    end

    # compute the fourier transform of the position-space measurements
    for key in keys(container_R)
        fourier_transform!(container_K[key], container_R[key], ft_coef)
    end
end


function initialize_nonlocal_measurement_files(container_rspace::Dict{String,Array{T,6}}, container_kspace::Dict{String,Array{T,6}}, sim_params::SimulationParameters{T})  where {T<:AbstractFloat}

    # data filename
    filename = "" 

    # constructing full filepath
    filepath = sim_params.filepath * sim_params.foldername

    # making directory the data will be written into
    if !isdir(filepath)
        mkdir(filepath)
    end

    # iterating over real space measurements
    for key in keys(container_rspace)
        # Intializing data file
        open(filepath * key * ".out", "w") do file
            # writing file header
            write(file, "tau", "\t", "orbit1", "\t", "orbit2", "\t", "dL1",  "\t", "dL2",  "\t", "dL3",  "\t", key, "\n")
        end
    end

    # iterating over momentum-space measurements
    for key in keys(container_kspace)
        # Intializing data file
        open(filepath * key* ".out", "w") do file
            # writing file header
            write(file, "tau", "\t", "orbit1", "\t", "orbit2", "\t", "dL1",  "\t", "dL2",  "\t", "dL3",  "\t", key, "\n")
        end
    end
end


function write_nonlocal_measurements(container::Dict{String,Array{T,6}}, sim_params::SimulationParameters{T})  where {T<:AbstractFloat}

    # getting array of measurements
    measurements::Vector{String} = collect(keys(container))

    # getting size of lattice
    L1, L2, L3, norbits, ignore, Lτ = size(container[ measurements[1] ])

    # measurement value to be written to file
    meas::T = 0.0

    # iterate over measurements
    for measurement in measurements
        # opening file correspond to current measurement
        open( sim_params.filepath * sim_params.foldername * measurement * ".out" , "a" ) do file
            # iterating over time slice
            for τ in 0:Lτ-1
                # iterating over unique orbital pairs
                for orbit1 in 1:norbits
                    for orbit2 in orbit1:norbits
                        # iterating all displacement vectors in unit cells
                        for ΔL3 in 0:L3-1
                            for ΔL2 in 0:L2-1
                                for ΔL1 in 0:L1-1
                                    # Getting value of measurement. 
                                    # Note that this averages over the two possible ordering for the orbitals
                                    meas  = container[measurement][ΔL1+1,ΔL2+1,ΔL3+1,orbit2,orbit1,τ+1]
                                    meas += container[measurement][ΔL1+1,ΔL2+1,ΔL3+1,orbit1,orbit2,τ+1]
                                    meas /= 2.0
                                    write(file, @sprintf( "%d\t%d\t%d\t%d\t%d\t%d\t%.6f\n" , τ, orbit1, orbit2, ΔL1, ΔL2, ΔL3, meas ) )
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

#########################################
## METHODS DEFINING MEASUREMENTS BELOW ##
#########################################

function measure_greens(Grτᵢⱼ₊, Grτᵢⱼ₋)

    return ( Grτᵢⱼ₊ + Grτᵢⱼ₋ ) / 2.0
end

end