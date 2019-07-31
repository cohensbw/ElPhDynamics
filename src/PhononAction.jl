module PhononAction

using Langevin.HolsteinModels: HolsteinModel
using Langevin.QuantumLattices: view_by_site, view_by_τ

export calc_Sbose, calc_dSbose!

"""
Calculates the phonon action.
"""
function calc_Sbose(holstein::HolsteinModel{T})::T where {T<:AbstractFloat}

    ϕ  = holstein.ϕ
    nsites = holstein.lattice.nsites
    Δτ = holstein.qlattice.Δτ
    ω  = holstein.ω
    Δϕ::Complex{T} = 0.0
    Δτω²over2::Complex{T} = 0.0

    ################################################
    ## Calculating Phonon Action Associated With  ##
    ## Local Phonon Frequency and Phonon Momentum ##
    ################################################

    Sbose = 0.0 # phonon action

    for i in 1:holstein.lattice.nsites
        # get phonon fields associated with current site
        ϕi = view_by_site(ϕ,i,nsites)
        # getting local phonon frequency
        Δτω²over2 = Δτ*ω[i]*ω[i]/1.0
        # iterating over time slices
        for τ in 1:Lτ
            # updating based on local phonon freq
            Sbose += Δτω²over2 * ϕi[τ] * ϕi[τ]
            # updating based on phonon momentum
            Δϕ = ϕi[τ%Lτ+1] - ϕi[τ]
            Sbose += (Δϕ*Δϕ)/(2*Δτ)
        end
    end

    #################################################################
    ## Calculating Phonon Action Associated with Phonon Dispersion ##
    #################################################################

    # checking if any dispersive phonon modes are defined
    if length(holstein.ωij)>0
        ωij                = holstein.ωij
        neighbor_table_ωij = holstein.neighbor_table_ωij
        site1 = 0
        site2 = 0
        Δτωij²over2::Complex{T} = 0.0
        # iterating over dispersive phonon modes
        for m in 1:length(holstein.ωij)
            Δτωij²over2 = Δτ*holstein.ωij[m]*holstein.ωij[m]/2.0
            site1 = neighbor_table_ωij[1,m]
            site2 = neighbor_table_ωij[2,m]
            # getting phonon fields associated with each site
            ϕ1 = view_by_site(ϕ,site1,nsites)
            ϕ2 = view_by_site(ϕ,site2,nsites)
            # iterating over time slices
            for τ in 1:Lτ
                Δϕ = ϕ2[tau] - ϕ1[tau]
                Sbose += Δτωij²over2 * Δϕ * Δϕ
            end
        end
    end

    return Sbose
end


"""
Calculates the dervative phonon action with respect to each phonon field.
"""
function calc_dSbose!(dSbose::Vector{Complex{T}},holstein::HolsteinModel{T}) where {T<:AbstractFloat}

    @assert length(dSbose)==holstein.qlattice.nindices

    nsites   = holstein.lattice.nsites::Int
    Lτ   = holstein.qlattice.Lτ::Int
    Δτ   = holstein.qlattice.Δτ::T
    ω    = holstein.ω::Vector{Complex{T}}
    ϕ    = holstein.ϕ::Vector{Complex{T}}
    τp1  = 0
    Δτω²::Complex{T} = 0.0
    Δ::Complex{T}    = 0.0

    # Intialize derivatves of phonon action to zero
    dSbose .= 0.0

    #####################################################
    ## Calculating Derivative Phonon Action Associated ##
    ## With Local Phonon Frequency And Phonon Momentum ##
    #####################################################

    # iterating over sites in lattice
    for i in 1:holstein.lattice.nsites
        Δτω² = Δτ * ω[i] * ω[i]
        # getting the phonon fields associated with current site
        ϕi = view_by_site(ϕ,i,nsites)
        # getting view into derviative for current sites
        dSbi = view_by_site(dSbose,i,nsites)
        # iterating over imaginary time axis
        for τ in 1:Lτ
            # updating action based on phonon potential energy
            dSbi[τ] += Δτω² * ϕi[τ]
            # updating action based on phonon momentum by taking the derivative
            # of [ϕᵢ(τ+1)-ϕᵢ(τ)]²/(2⋅Δτ) with respect to both ϕᵢ(τ+1) and ϕᵢ(τ)
            τp1 = τ%Lτ+1 #  get τ+1 accounting for periodic boundary conditions
            Δ = ( ϕi[τp1] - ϕi[τ] ) / Δτ # Δ = [ϕᵢ(τ+1)-ϕᵢ(τ)]/Δτ with pbc
            dSbi[τp1] +=  Δ # ∂Smomentum/∂ϕᵢ(τ+1) +=  [ϕᵢ(τ+1)-ϕᵢ(τ)]/Δτ with pbc
            dSbi[τ]   += -Δ # ∂Smomentum/∂ϕᵢ(τ)   += -[ϕᵢ(τ+1)-ϕᵢ(τ)]/Δτ
        end
    end

    ############################################################################
    ## Calculating Derivative Phonon Action Associated With Phonon Dispersion ##
    ############################################################################

    # checking if there are any dispersive phonon modes
    if length(holstein.ωij)>0
        ωij = holstein.ωij
        sign_ωij = holstein.sign_ωij
        neighbor_table_ωij = holstein.neighbor_table_ωij
        Δτωij²::Complex{T} = 0.0
        i = 0
        j = 0
        sgn = 1
        # iterate over dispersive phonon modes
        for m in 1:length(ωij)
            Δτωij² = Δτ * ωij[m] * ωij[m]
            sgn = sign_ωij[m]
            # getting pair of neighboring sites
            i = neighbor_table_ωij[1,m]
            j = neighbor_table_ωij[2,m]
            # getting phonon fields associated with pair of neighboring sites
            ϕi = view_by_site(ϕ,i,nsites)
            ϕj = view_by_site(ϕ,j,nsites)
            # get view into action associated with each site
            dSbi = view_by_site(dSbose,i,nsites)
            dSbj = view_by_site(dSbose,j,nsites)
            # iterating over imaginary time axis
            for τ in 1:Lτ
                # udpating derviative of phonon action
                Δ = Δτωij² * (ϕi[τ] + sgn*ϕj[τ])
                dSbi[τ] += Δ
                dSbj[τ] += sgn*Δ
            end
        end
    end

    return nothing
end

end