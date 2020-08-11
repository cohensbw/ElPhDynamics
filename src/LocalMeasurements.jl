module LocalMeasurements

using Printf
using UnsafeArrays

using ..Utilities: get_index, get_site, get_τ, δ
using ..Models: HolsteinModel
using ..SimulationParams: SimulationParameters
using ..GreensFunctions: EstimateGreensFunction, estimate
using ..NonLocalMeasurements: measure_Greens, measure_DenDen

export make_local_measurements!
export construct_local_measurements_container
export process_local_measurements!
export reset_local_measurements!
export initialize_local_measurements_file
export write_local_measurements


function make_local_measurements!(container::NamedTuple, holstein::HolsteinModel, Gr::EstimateGreensFunction)
    
    # phonon fields
    x = holstein.x

    # phonon frequncies
    ω = holstein.ω

    # anhmarnic phonon frequency
    ω4 = holstein.ω4

    # electron-phonon coupling
    λ = holstein.λ

    # getting number of orbitals
    norbits = holstein.lattice.unit_cell.norbits::Int

    # number of physical sites in lattice
    nsites = holstein.nsites::Int

    # length of imaginary time axis
    Lτ = holstein.Lτ::Int

    # for measuring phonon kinetic energy
    Δτ  = holstein.Δτ
    Δτ² = Δτ * Δτ

    # normalization
    normalization = div(nsites,norbits)*Lτ

    # iterating over orbital types
    for orbit in 1:norbits
        # measure density
        container.density[orbit] += 2.0 * (1.0-real(measure_Greens(Gr,0,0,0,orbit,orbit,0)))
        # measure double occupancy
        container.double_occ[orbit] += real(measure_DenDen(Gr,0,0,0,orbit,orbit,0))
        # iterating over orbits of the current type
        for site in orbit:norbits:nsites
            # iterating over time slices
            for τ in 1:Lτ
                # getting current index
                index = get_index(τ,site,Lτ)
                # estimate ⟨cᵢ(τ)c⁺ᵢ(τ)⟩
                G1 = estimate(Gr,site,site,τ,τ,1)
                G2 = estimate(Gr,site,site,τ,τ,2)
                # measuring phonon kinetic energy such that
                # ⟨KE⟩ = 1/(2Δτ) - ⟨(1/2)[xᵢ(τ+1)-xᵢ(τ)]²/Δτ²⟩
                Δx = x[get_index(τ%Lτ+1,site,Lτ)]-x[index]
                container.phonon_kin[orbit] += (0.5/Δτ-(Δx*Δx)/Δτ²/2) / normalization
                # measuring phonon potential energy
                container.phonon_pot[orbit] += (ω[site]^2*x[index]^2/2 + ω4[site]*x[index]^4) / normalization
                # measuring the electron phonon energy λ⟨x⋅(n₊+n₋)⟩
                container.elph_energy[orbit] += λ[site]*x[index]*(2.0-G1-G2) / normalization
                # measure ⟨x⟩
                container.phi[orbit] += x[index] / normalization
                # measure ⟨x²⟩
                container.phi_squared[orbit] += x[index]*x[index] / normalization
                # measure chemical potential
                container.mu[orbit] += holstein.μ[site] / normalization
            end
        end
    end

    return nothing
end


"""
Construct a dictionary to hold local measurement data.
"""
function construct_local_measurements_container(holstein::HolsteinModel{T1,T2}, unequaltime_meas::AbstractVector{String})::NamedTuple where {T1<:AbstractFloat,T2<:Number}

    local_meas_container = Dict()
    for meas in ("density", "double_occ", "phonon_kin", "phonon_pot", "elph_energy", "phi_squared", "phi","mu")
        local_meas_container[meas] = zeros(T1,holstein.lattice.unit_cell.norbits)
    end

    # only measure S-Wave Susceptibility if Unequal Time Pair Green's Function is being measured
    if "PairGreens" in unequaltime_meas
        local_meas_container["swave_susc"] = zeros(T1,holstein.lattice.unit_cell.norbits)
    end

    # converting dictionary to named tuple
    local_meas_container = NamedTuple{Tuple(Symbol.(keys(local_meas_container)))}(values(local_meas_container))

    return local_meas_container
end


"""
Process Local Measurements.
"""
function process_local_measurements!(container::NamedTuple, sim_params::SimulationParameters, holstein::HolsteinModel,
                                     container_rspace::NamedTuple, container_kspace::NamedTuple)
    
    # normalizing measurements
    for key in keys(container)
        container[key] ./= sim_params.bin_size
    end

    # number of orbits/sites per unit cell
    norbits = holstein.lattice.unit_cell.norbits::Int

    # if measuring s-wave susceptibility
    if :swave_susc in keys(container)
        # iterating over orbitals
        for orbit in 1:norbits
            # pair green's function
            Pᵣ = container_rspace.PairGreens
            # green's function
            Gᵣ = container_rspace.Greens
            # calculating s-wave susceptibility
            container.swave_susc[orbit] = calc_swave_susc(Pᵣ, Gᵣ, holstein.Δτ, orbit)
        end
    end

    return nothing
end


"""
Reset the arrays that contain the measurements to all zeros.
"""
function reset_local_measurements!(container::NamedTuple) where {T<:Number}

    for key in keys(container)
        container[key] .= 0.0
    end

    return nothing
end


"""
Initializes file that will contain local measurement data, with header included.
"""
function initialize_local_measurements_file(container::NamedTuple, sim_params::SimulationParameters) where {T<:Number}

    open(sim_params.datafolder*"local_measurements.out", "w") do file
        write(file, "orbit")
        for key in keys(container)
            measurement = String(key)
            write(file, ",", measurement)
        end
        write(file, "\n")
    end
    return nothing
end


"""
Write non-local measurements to file.
"""
function write_local_measurements(container::NamedTuple, sim_params::SimulationParameters, holstein::HolsteinModel) where {T<:Number}

    norbits = holstein.lattice.unit_cell.norbits::Int
    open(sim_params.datafolder*"local_measurements.out", "a") do file
        for orbit in 1:norbits
            write(file, string(orbit))
            for key in keys(container)
                write(file, @sprintf(",%.6f", container[key][orbit]))
            end
            write(file, "\n")
        end
    end

    return nothing
end

#################################
## MEASUREMENT IMPLEMENTATIONS ##
#################################

"""
Measure S-wave Pair Susceptibility using Simpson's rule for evaluating the numerical integral.
Pᵣ is Pair Green's Functions and Gᵣ is the real space Green's Functions
"""
function calc_swave_susc(Pᵣ::Array{Complex{T},6},Gᵣ::Array{Complex{T},6},Δτ,orbit)::T where {T<:AbstractFloat}
    
    L  = size(Pᵣ,1)
    L₁ = size(Pᵣ,4)
    L₂ = size(Pᵣ,5)
    L₃ = size(Pᵣ,6)
    Pₛ = 0.0im
    @uviews Pᵣ begin
        # iterate over unit cell displacements
        for l₃ in 0:L₃-1
            for l₂ in 0:L₂-1
                for l₁ in 0:L₁-1
                    # calculate Pair Green's Function Pᵣ(β)
                    pᵣ  = @view Pᵣ[:,orbit,orbit,l₁+1,l₂+1,l₃+1]
                    Gᵣ0 = Gᵣ[1,orbit,orbit,l₁+1,l₂+1,l₃+1]
                    pᵣβ = δ(l₁)*δ(l₂)*δ(l₃)*(1.0-2*Gᵣ0) + pᵣ[1]
                    # iterate over τ to do Simpson integration
                    for τ in 2:2:L
                        Pₛ +=     pᵣ[τ-1] * 1.0/3.0 * Δτ
                        Pₛ +=     pᵣ[τ]   * 4.0/3.0 * Δτ
                        if τ==L
                            Pₛ += pᵣβ     * 1.0/3.0 * Δτ
                        else
                            Pₛ += pᵣ[τ+1] * 1.0/3.0 * Δτ
                        end
                    end
                    # deal with boundary condition for Simpson integration
                    if isodd(L)
                        Pₛ -= pᵣ[L-1] * 1.0/12.0 * Δτ
                        Pₛ += pᵣ[L]   * 2.0/3.0  * Δτ
                        Pₛ += pᵣβ     * 5.0/12.0 * Δτ
                    end
                end
            end
        end
    end
    return real(Pₛ)
end

end