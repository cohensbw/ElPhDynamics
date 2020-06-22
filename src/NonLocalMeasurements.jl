module NonLocalMeasurements

using Printf
using FFTW
using LinearAlgebra

using ..Utilities: get_index, get_site, get_τ, θ, δ
using ..HolsteinModels: HolsteinModel
using ..SimulationParams: SimulationParameters
using ..GreensFunctions: EstimateGreensFunction, update!, estimate

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
Measure time-ordered single-particle electron Green's function ⟨T⋅cᵢ(τ₂)c⁺ⱼ(τ₁)⟩
"""
@inline function measure_Greens(i,j,τ₂,τ₁,Gr1,Gr2)

    # estimate ⟨cᵢ(τ₂)c⁺ⱼ(τ₁)⟩
    Gᵢⱼτ₂τ₁1 = estimate(Gr1,i,j,τ₂,τ₁)
    Gᵢⱼτ₂τ₁2 = estimate(Gr2,i,j,τ₂,τ₁)

    # takes care of time-ordering with Heaviside step function
    return (1-2*θ(τ₁-τ₂))*(Gᵢⱼτ₂τ₁1+Gᵢⱼτ₂τ₁2)/2
end


"""
Measure the density-density correlation function ⟨nᵢ(τ₂)nⱼ(τ₁)⟩ where
nᵢ(τ₂) = nᵢ₊(τ₂) + nᵢ₋(τ₂).
"""
function measure_DenDen(i,j,τ₂,τ₁,Gr1,Gr2)

    # estimate ⟨cᵢ(τ₂)c⁺ⱼ(τ₁)⟩
    Gᵢⱼτ₂τ₁1 = estimate(Gr1,i,j,τ₂,τ₁)
    Gᵢⱼτ₂τ₁2 = estimate(Gr2,i,j,τ₂,τ₁)

    # estimate ⟨cⱼ(τ₁)c⁺ᵢ(τ₂)⟩
    Gⱼᵢτ₁τ₂1 = estimate(Gr1,j,i,τ₁,τ₂)
    Gⱼᵢτ₁τ₂2 = estimate(Gr2,j,i,τ₁,τ₂)

    # estimate ⟨cⱼ(τ₁)c⁺ⱼ(τ₁)⟩
    Gⱼⱼτ₁τ₁1 = estimate(Gr1,j,j,τ₁,τ₁)
    Gⱼⱼτ₁τ₁2 = estimate(Gr2,j,j,τ₁,τ₁)

    # estimate ⟨cᵢ(τ₂)c⁺ᵢ(τ₂⟩
    Gᵢᵢτ₂τ₂1 = estimate(Gr1,i,i,τ₂,τ₂)
    Gᵢᵢτ₂τ₂2 = estimate(Gr2,i,i,τ₂,τ₂)

    # ⟨nᵢ₊(τ₂)nⱼ₊(τ₁)⟩ = [1-⟨cᵢ₊(τ₂)c⁺ᵢ₊(τ₂)⟩]⋅[1-⟨cⱼ₊(τ₁)c⁺ⱼ₊(τ₁)⟩]
    #                  + ⟨cᵢ₊(τ₂)c⁺ⱼ₊(τ₁)⟩⋅[δ(τ₂,τ₁)δ(i,j)-⟨cⱼ₊(τ₁)c⁺ᵢ₊(τ₂)⟩]
    nᵢ₊τ₂_nⱼ₊τ₁ = (1-Gᵢᵢτ₂τ₂1)*(1-Gⱼⱼτ₁τ₁2) + Gᵢⱼτ₂τ₁1*(δ(τ₂,τ₁)*δ(i,j)-Gⱼᵢτ₁τ₂2)

    # ⟨nᵢ₋(τ₂)nⱼ₋(τ₁)⟩ = (same as above but with the spin flipped)
    nᵢ₋τ₂_nⱼ₋τ₁ = (1-Gᵢᵢτ₂τ₂2)*(1-Gⱼⱼτ₁τ₁1) + Gᵢⱼτ₂τ₁2*(δ(τ₂,τ₁)*δ(i,j)-Gⱼᵢτ₁τ₂1)

    # ⟨nᵢ₊(τ₂)nⱼ₋(τ₁)⟩ = [1-⟨cᵢ₊(τ₂)c⁺ᵢ₊(τ₂)⟩]⋅[1-⟨cⱼ₋(τ₁)c⁺ⱼ₋(τ₁)⟩]
    nᵢ₊τ₂_nⱼ₋τ₁ = (1-Gᵢᵢτ₂τ₂1)*(1-Gⱼⱼτ₁τ₁2)

    # ⟨nᵢ₋(τ₂)nⱼ₊(τ₁)⟩ = (same as above but with the spin flipped)
    nᵢ₋τ₂_nⱼ₊τ₁ = (1-Gᵢᵢτ₂τ₂2)*(1-Gⱼⱼτ₁τ₁1)

    # ⟨nᵢ(τ₂)nⱼ(τ₁)⟩ = ⟨[nᵢ₊(τ₂)+nᵢ₋(τ₂)]⋅[nⱼ₊(τ₁)+nⱼ₋(τ₁)]⟩
    # ⟨nᵢ(τ₂)nⱼ(τ₁)⟩ = ⟨nᵢ₊(τ₂)nⱼ₊(τ₁)⟩ + ⟨nᵢ₋(τ₂)nⱼ₋(τ₁)⟩ + ⟨nᵢ₊(τ₂)nⱼ₋(τ₁)⟩ + ⟨nᵢ₋(τ₂)nⱼ₊(τ₁)⟩
    nᵢτ₂_nⱼτ₁ = nᵢ₊τ₂_nⱼ₊τ₁ + nᵢ₋τ₂_nⱼ₋τ₁ + nᵢ₊τ₂_nⱼ₋τ₁ + nᵢ₋τ₂_nⱼ₊τ₁

    return nᵢτ₂_nⱼτ₁
end


"""
Measure pair Green's function ⟨Δᵢ(τ₂) Δ⁺ⱼ(τ₁)⟩ where Δᵢ(τ₂) = cᵢ₊(τ₂)cᵢ₋(τ₂).
"""
function measure_PairGreens(i,j,τ₂,τ₁,Gr1,Gr2)

    # estimate ⟨cᵢ(τ₂)c⁺ⱼ(τ₁)⟩
    Gᵢⱼτ₂τ₁1 = estimate(Gr1,i,j,τ₂,τ₁)
    Gᵢⱼτ₂τ₁2 = estimate(Gr2,i,j,τ₂,τ₁)

    # ⟨Δᵢ(τ₂) Δ⁺ⱼ(τ₁)⟩ = ⟨cᵢ₊(τ₂)cᵢ₋(τ₂)   c⁺ⱼ₊(τ₁)c⁺ⱼ₋(τ₁)⟩
    # ⟨Δᵢ(τ₂) Δ⁺ⱼ(τ₁)⟩ = ⟨cᵢ₊(τ₂)c⁺ⱼ₊(τ₁)⟩⋅⟨cᵢ₋(τ₂)c⁺ⱼ₋(τ₁)⟩
    return Gᵢⱼτ₂τ₁1*Gᵢⱼτ₂τ₁2
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
        function $op(container::Array{Complex{T1},6}, trans_equiv_sets::Array{Int,7},
                     Gr1::EstimateGreensFunction{T1}, Gr2::EstimateGreensFunction{T2},
                     downsample::Int=1) where {T1<:AbstractFloat,T2<:Number}

            # getting size of system
            Lτ, L1, L2, L3, norbits, ignore = size(container)

            # number of unit cells in lattice
            ncells = L1*L2*L3

            # normalization factor
            normalization = ncells * length(1:downsample:Gr1.β)

            # iterating over all possible parings of orbitals
            @fastmath @inbounds for orbit1 in 1:norbits
                for orbit2 in 1:norbits
                    # iterating over all displacements vectors defined in terms
                    # of unit cells in the direction of each lattice vector.
                    for ΔL3 in 0:L3-1
                        for ΔL2 in 0:L2-1
                            for ΔL1 in 0:L1-1
                                
                                # iterating over pairs of sites corresponding to current displacement vector
                                for pair in 1:ncells

                                    # getting current pair of sites associated with specified
                                    # displacement vector r=i-j
                                    j = trans_equiv_sets[1,pair,ΔL1+1,ΔL2+1,ΔL3+1,orbit2,orbit1]
                                    i = trans_equiv_sets[2,pair,ΔL1+1,ΔL2+1,ΔL3+1,orbit2,orbit1]

                                    # iterate over possible time seperations
                                    for τ in 0:Lτ-1

                                        # iterating over time slices
                                        for τ₁ in 1:downsample:Lτ
                                            
                                            # getting second time slice τ₂=τ₁+τ accounting for boundary conditions
                                            τ₂ = mod1(τ₁+τ,Lτ)

                                            # making measurement
                                            container[ τ+1, ΔL1+1, ΔL2+1, ΔL3+1, orbit2, orbit1 ] += $measure(i,j,τ₂,τ₁,Gr1,Gr2) / normalization
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
function make_nonlocal_measurements!(container::Dict{String,Array{Complex{T1},6}}, holstein::HolsteinModel{T1,T2}, Gr1::EstimateGreensFunction{T1}, Gr2::EstimateGreensFunction{T1}, downsample::Int=1) where {T1<:AbstractFloat,T2<:Number}

    # get translationally equivalent sets of points
    trans_equiv_sets = holstein.trans_equiv_sets

    # iterate over measurements
    for key in keys(container)
        if key=="Greens"
            # Measure Electron Greens Function
            measure_Greens!(container["Greens"],trans_equiv_sets,Gr1,Gr2,downsample)
        elseif key=="DenDen"
            # Measure Density-Density Correlation Function
            measure_DenDen!(container["DenDen"],trans_equiv_sets,Gr1,Gr2,downsample)
        elseif key=="PairGreens"
            # Measure Pair Greens Function
            measure_PairGreens!(container["PairGreens"],trans_equiv_sets,Gr1,Gr2,downsample)
        else
            error("The following key is not a valid measurment: ",key)
        end
    end

    return nothing
end


"""
Construct a dictionary for the real space and momentum space measurements,
where each measurement is a key in the dictionary and points to an array
where the measured values will be stored.
"""
function construct_nonlocal_measurements_container(holstein::HolsteinModel{T1,T2}, equaltime_meas::AbstractVector{String}, unequaltime_meas::AbstractVector{String})::Tuple{ Dict{String,Array{Complex{T1},6}} , Dict{String,Array{Complex{T1},6}} } where {T1<:AbstractFloat,T2<:Number}

    # get size of lattice
    Lτ = holstein.Lτ
    L1 = holstein.lattice.L1
    L2 = holstein.lattice.L2
    L3 = holstein.lattice.L3
    norbits = holstein.lattice.norbits

    container_rspace = Dict()
    container_kspace = Dict()

    # iterate over all unequal time measurements
    for measurement in unequaltime_meas
        # declare containers
        container_rspace[measurement] = zeros(Complex{T1},(Lτ,L1,L2,L3,norbits,norbits))
        container_kspace[measurement] = zeros(Complex{T1},(Lτ,L1,L2,L3,norbits,norbits))
    end

    # iterate over all equal time measurements
    for measurement in equaltime_meas
        # declare containers
        container_rspace[measurement] = zeros(Complex{T1},(1,L1,L2,L3,norbits,norbits))
        container_kspace[measurement] = zeros(Complex{T1},(1,L1,L2,L3,norbits,norbits))
    end

    return container_rspace, container_kspace
end


"""
Reset the arrays that contain the measurements to all zeros.
"""
function reset_nonlocal_measurements!(container::Dict{String,Array{T,6}}) where {T<:Number}

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
function process_nonlocal_measurements!(container_rspace::Dict{String,Array{Complex{T},6}}, container_kspace::Dict{String,Array{Complex{T},6}}, sim_params::SimulationParameters) where {T<:AbstractFloat}

    # iterate over measurements
    for key in keys(container_kspace)

        vals_r = container_rspace[key]
        vals_k = container_kspace[key]

        # do fft from r to k space
        copyto!(vals_k,vals_r)
        fft!(vals_k, (2,3,4))

        # normalize measurements
        container_rspace[key] ./= sim_params.bin_size
        container_kspace[key] ./= sim_params.bin_size
    end

    return nothing
end


"""
Initializes files (including header) that each measurement will be written to.
"""
function initialize_nonlocal_measurement_files(container_rspace::Dict{String,Array{Complex{T},6}}, container_kspace::Dict{String,Array{Complex{T},6}}, sim_params::SimulationParameters)  where {T<:AbstractFloat}

    # iterating over real space measurements
    for key in keys(container_rspace)
        # Intializing data file
        open(sim_params.datafolder * key * "_r.out", "w") do file
            # writing file header
            write(file, "orbit1", ",", "orbit2", ",", "dL1",  ",", "dL2",  ",", "dL3",  ",", "tau", ",", key*"_r", "\n")
        end
    end

    # iterating over momentum-space measurements
    for key in keys(container_kspace)
        # Intializing data file
        open(sim_params.datafolder * key * "_k.out", "w") do file
            # writing file header
            write(file, "orbit1", ",", "orbit2", ",", "dL1",  ",", "dL2",  ",", "dL3",  ",", "tau", ",", key*"_k", "\n")
        end
    end

    return nothing
end


"""
Write non-local measurements to file. Each measurement gets its own file.
"""
function write_nonlocal_measurements(container::Dict{String,Array{T,6}}, sim_params::SimulationParameters, holstein::HolsteinModel; real_space::Bool)  where {T<:Number}

    # string to hold filename
    filename = ""

    # iterate over measurements
    for measurement in keys(container)

        # getting a pointer to the array containing the measurements
        vals = container[measurement]

        # getting size of system
        Lτ, L1, L2, L3, norbits, ignore = size(vals)

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
            # iterating over unique orbital pairs
            for orbit1 in 1:norbits
                for orbit2 in orbit1:norbits
                    # iterating all displacement vectors in unit cells
                    for ΔL3 in 0:L3-1
                        for ΔL2 in 0:L2-1
                            for ΔL1 in 0:L1-1
                                # iterating over time slice
                                for τ in 0:Lτ-1
                                    # Getting value of measurement. 
                                    # Note that this averages over the two possible orderings for the orbitals
                                    meas  = real( vals[τ+1,ΔL1+1,ΔL2+1,ΔL3+1,orbit2,orbit1] )
                                    meas += real( vals[τ+1,ΔL1+1,ΔL2+1,ΔL3+1,orbit1,orbit2] )
                                    meas /= 2.0
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