module NonLocalMeasurements

using Printf
using ..HolsteinModels: HolsteinModel, get_index, get_site, get_τ
using ..LangevinSimulationParameters: SimulationParameters
using ..GreensFunctions: EstimateGreensFunction, update!, estimate
using ..FourierTransforms: fourier_transform!

export make_nonlocal_measurements!
export reset_nonlocal_measurements!
export process_nonlocal_measurements!
export construct_nonlocal_measurements_container
export initialize_nonlocal_measurement_files
export write_nonlocal_measurements

"""
Makes non-local measurements in real-space.
Each type of measurement is made for all possible imaginary time seperations and displacement vectors.
Measurements will be stored in arrays with 6 indices where the indices correspond to
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

    # getting the number of paired sites assoicated with each displacement vector
    npairs = div(lattice.nsites,lattice.norbits)

    # normalization factor
    normalization = npairs * holstein.Lτ

    # getting pointers to arrays containing measurements
    greens = container["Greens"]
    denden = container["DenDen"]
    pairgreens = container["PairGreens"]

    # iterating over all possible parings of orbitals
    @fastmath @inbounds for orbit1 in 1:lattice.norbits
        for orbit2 in 1:lattice.norbits
            # iterating over all displacements vectors defined in terms
            # of unit cells in the direction of each lattice vector.
            for ΔL3 in 0:lattice.L3-1
                for ΔL2 in 0:lattice.L2-1
                    for ΔL1 in 0:lattice.L1-1
                        # iterate over possible time seperations
                        for τ in 0:holstein.Lτ-1
                            # iterating over pairs of sites corresponding to current displacement vector
                            for pair in 1:npairs
                                # getting current pair of sites associated with specified
                                # displacement vector r=i-j
                                j = sets[1,pair,ΔL1+1,ΔL2+1,ΔL3+1,orbit2,orbit1]
                                i = sets[2,pair,ΔL1+1,ΔL2+1,ΔL3+1,orbit2,orbit1]
                                # iterating over time slices
                                for τ₁ in 1:holstein.Lτ
                                    
                                    # getting second time slice τ₂=τ₁+τ accounting for boundary conditions
                                    τ₂ = (τ₁+τ-1)%Gr1.β+1

                                    # estimate ⟨cᵢ(τ₂)c⁺ⱼ(τ₁)⟩
                                    Gᵢⱼτ₂τ₁1 = estimate(Gr1,i,j,τ₂,τ₁)
                                    Gᵢⱼτ₂τ₁2 = estimate(Gr2,i,j,τ₂,τ₁)

                                    # estimate ⟨cⱼ(τ₁)c⁺ᵢ(τ₂)⟩
                                    Gⱼᵢτ₁τ₂1 = estimate(Gr1,j,i,τ₁,τ₂)
                                    Gⱼᵢτ₁τ₂2 = estimate(Gr2,j,i,τ₁,τ₂)

                                    # estimate ⟨cⱼ(τ₁)c⁺ⱼ(τ₁)⟩
                                    Gⱼⱼτ₁τ₁1 = estimate(Gr1,j,j,τ₁,τ₁)
                                    Gⱼⱼτ₁τ₁2 = estimate(Gr2,j,j,τ₁,τ₁)

                                    # estimate ⟨cⱼ(τ₁)c⁺ⱼ(τ₁)⟩
                                    Gᵢᵢτ₂τ₂1 = estimate(Gr1,i,i,τ₂,τ₂)
                                    Gᵢᵢτ₂τ₂2 = estimate(Gr2,i,i,τ₂,τ₂)

                                    # measuring electron green's function ⟨cᵢ(τ)c⁺ⱼ(0)⟩ where β>τ≥0.
                                    greens[ τ+1, ΔL1+1, ΔL2+1, ΔL3+1, orbit2, orbit1 ] +=
                                        measure_Greens(τ₂,τ₁,Gᵢⱼτ₂τ₁1,Gᵢⱼτ₂τ₁2) / normalization

                                    # measuring density-density correlation ⟨nᵢ(τ)nⱼ(0)⟩
                                    denden[ τ+1, ΔL1+1, ΔL2+1, ΔL3+1, orbit2, orbit1 ] +=
                                        measure_DenDen(i,j,τ₂,τ₁,Gᵢⱼτ₂τ₁1,Gᵢⱼτ₂τ₁2,Gⱼᵢτ₁τ₂1,Gⱼᵢτ₁τ₂2,Gⱼⱼτ₁τ₁1,Gⱼⱼτ₁τ₁2,Gᵢᵢτ₂τ₂1,Gᵢᵢτ₂τ₂2) / normalization

                                    # measuring Pair Green's Function ⟨Δᵢ(τ)Δ⁺ⱼ(0)+Δⱼ(0)Δ⁺ᵢ(τ)⟩
                                    pairgreens[ τ+1, ΔL1+1, ΔL2+1, ΔL3+1, orbit2, orbit1 ] +=
                                        measure_PairGreens(Gᵢⱼτ₂τ₁1,Gᵢⱼτ₂τ₁2,Gⱼᵢτ₁τ₂1,Gⱼᵢτ₁τ₂2) / normalization
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end


"""
Construct a dictionary for the real space and momentum space measurements,
where each measurement is a key in the dictionary and points to an array
where the measured values will be stored.
"""
function construct_nonlocal_measurements_container(holstein::HolsteinModel{T1,T2})::Tuple{ Dict{String,Array{T1,6}} , Dict{String,Array{Complex{T1},6}} } where {T1<:AbstractFloat,T2<:Number}

    lattice = holstein.lattice
    container_rspace = Dict()
    container_kspace = Dict()
    # ierate over all measurements to be made
    for meas in ("Greens","DenDen","PairGreens")
        container_rspace[meas] = zeros(T1,(holstein.Lτ,lattice.L1,lattice.L2,lattice.L3,lattice.norbits,lattice.norbits))
        container_kspace[meas] = zeros(Complex{T1},(holstein.Lτ,lattice.L1,lattice.L2,lattice.L3,lattice.norbits,lattice.norbits))
    end
    return container_rspace, container_kspace
end


"""
Reset the arrays that contain the measurements to all zeros.
"""
function reset_nonlocal_measurements!(container::Dict{String,Array{T,6}}) where {T<:Number}

    for key in keys(container)
        container[key] .= 0.0
    end
end


"""
Process the real-space and momentum-space measurements.
This includes first performing the fourier transform and
the normalzing by the number of measurement per bin.
"""
function process_nonlocal_measurements!(container_rspace::Dict{String,Array{T,6}}, container_kspace::Dict{String,Array{Complex{T},6}}, sim_params::SimulationParameters{T}, ft_coeff::Array{Complex{T},6}) where {T<:AbstractFloat}

    # compute the fourier transform of the position-space measurements
    for key in keys(container_kspace)
        fourier_transform!(container_kspace[key], container_rspace[key], ft_coeff)
    end

    # normalize the values by the number of measurements per bin
    for key in keys(container_rspace)
        container_rspace[key] ./= sim_params.bin_size
        container_kspace[key] ./= sim_params.bin_size
    end
end


"""
Initializes files (including header) that each measurement will be written to.
"""
function initialize_nonlocal_measurement_files(container_rspace::Dict{String,Array{T,6}}, container_kspace::Dict{String,Array{Complex{T},6}}, sim_params::SimulationParameters{T})  where {T<:AbstractFloat}

    # data filename
    filename = "" 

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
end


"""
Write non-local measurements to file. Each measurement gets its own file.
"""
function write_nonlocal_measurements(container::Dict{String,Array{T,6}}, sim_params::SimulationParameters, holstein::HolsteinModel; real_space::Bool)  where {T<:Number}

    # getting size of lattice
    Lτ = holstein.Lτ::Int
    L1 = holstein.lattice.L1::Int
    L2 = holstein.lattice.L2::Int
    L3 = holstein.lattice.L3::Int
    norbits = holstein.lattice.norbits::Int

    # measurement value to be written to file
    meas = 0.0

    # string to hold filename
    filename = ""

    # iterate over measurements
    for measurement in keys(container)

        # getting a pointer to the array containing the measurements
        vals = container[measurement]

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
                                    # Note that this averages over the two possible ordering for the orbitals
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
end

#########################################
## METHODS DEFINING MEASUREMENTS BELOW ##
#########################################

"""
Measure time-ordered single-particle electron Green's function ⟨T⋅cᵢ(τ₂)c⁺ⱼ(τ₁)⟩
"""
function measure_Greens(τ₂,τ₁,Gᵢⱼτ₂τ₁1,Gᵢⱼτ₂τ₁2)

    G = (Gᵢⱼτ₂τ₁1+Gᵢⱼτ₂τ₁2)/2
    # time ordering
    if τ₂<τ₁
        G *= -1.0
    end
    return G
end


"""
Measure the density-density correlation function ⟨nᵢ(τ₂)nⱼ(τ₁)⟩ where
nᵢ(τ₂) = nᵢ₊(τ₂) + nᵢ₋(τ₂).
"""
function measure_DenDen(i, j, τ₂, τ₁, Gᵢⱼτ₂τ₁1, Gᵢⱼτ₂τ₁2, Gⱼᵢτ₁τ₂1, Gⱼᵢτ₁τ₂2, Gⱼⱼτ₁τ₁1, Gⱼⱼτ₁τ₁2, Gᵢᵢτ₂τ₂1, Gᵢᵢτ₂τ₂2)

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
Measure pair Green's function ⟨Δᵢ(τ₂)Δ⁺ⱼ(τ₁)+h.c.⟩=⟨Δᵢ(τ₂)Δ⁺ⱼ(τ₁)+Δⱼ(τ₁)Δ⁺ᵢ(τ₂)⟩
where Δᵢ(τ₂) = cᵢ₊(τ₂)cᵢ₋(τ₂).
"""
function measure_PairGreens(Gᵢⱼτ₂τ₁1, Gᵢⱼτ₂τ₁2, Gⱼᵢτ₁τ₂1, Gⱼᵢτ₁τ₂2)

    # ⟨Δᵢ(τ₂)Δ⁺ⱼ(τ₁)+Δⱼ(τ₁)Δ⁺ᵢ(τ₂)⟩ = ⟨cᵢ₊(τ₂)cᵢ₋(τ₂)c⁺ⱼ₊(τ₁)c⁺ⱼ₋(τ₁) + cⱼ₊(τ₁)cⱼ₋(τ₁)c⁺ᵢ₊(τ₂)c⁺ᵢ₋(τ₂)
    # ⟨Δᵢ(τ₂)Δ⁺ⱼ(τ₁)+Δⱼ(τ₁)Δ⁺ᵢ(τ₂)⟩ = ⟨cᵢ₊(τ₂)c⁺ⱼ₊(τ₁)⟩⋅⟨cᵢ₋(τ₂)c⁺ⱼ₋(τ₁)⟩ + ⟨cⱼ₊(τ₁)c⁺ᵢ₊(τ₂)⟩⋅⟨cⱼ₋(τ₁)c⁺ᵢ₋(τ₂)⟩
    return Gᵢⱼτ₂τ₁1*Gᵢⱼτ₂τ₁2 + Gⱼᵢτ₁τ₂1*Gⱼᵢτ₁τ₂2
end


"""
Delta function.
"""
@inline function δ(i::T,j::T)::T where {T<:Number}

    return i==j
end

end