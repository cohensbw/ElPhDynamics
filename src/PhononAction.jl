module PhononAction

using Langevin.HolsteinModels: HolsteinModel
using Langevin.HolsteinModels: view_by_site, view_by_τ

export calc_Sbose, calc_dSbosedϕ!

"""
Calculates the phonon action.
"""
function calc_Sbose(holstein::HolsteinModel{T1,T2})::T2  where {T1<:AbstractFloat,T2<:Number}

    ϕ             = holstein.ϕ::Vector{T1}
    nsites        = holstein.nsites::Int
    Δτ            = holstein.Δτ::T
    ω             = holstein.ω::Vector{T}
    Δϕ::T1        = 0.0
    Δτω²over2::T1 = 0.0

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
        ωij                = holstein.ωij::Vector{T2}
        sign_ωij           = holstein.sign_ωij::Vector{Int}
        neighbor_table_ωij = holstein.neighbor_table_ωij::Matrix{Int}
        site1              = 0
        site2              = 0
        sgn                = 1
        Δτωij²over2::T2    = 0.0
        # iterating over dispersive phonon modes
        for m in 1:length(holstein.ωij)
            Δτωij²over2 = Δτ*holstein.ωij[m]*holstein.ωij[m]/2.0
            site1 = neighbor_table_ωij[1,m]
            site2 = neighbor_table_ωij[2,m]
            sgn   = sign_ωij[m]
            # getting phonon fields associated with each site
            ϕ1 = view_by_site(ϕ,site1,nsites)
            ϕ2 = view_by_site(ϕ,site2,nsites)
            # iterating over time slices
            for τ in 1:Lτ
                Δϕ = ϕ2[tau] + sgn*ϕ1[tau]
                Sbose += Δτωij²over2 * Δϕ * Δϕ
            end
        end
    end

    return Sbose
end


"""
Calculates the dervative phonon action with respect to each phonon field and adds that value in place
to the vector dSbose.
"""
function calc_dSbosedϕ!(dSbose::Vector{T2}, holstein::HolsteinModel{T1,T2})  where {T1<:AbstractFloat,T2<:Number}

    @assert length(dSbose)==holstein.nindices

    ϕ        = holstein.ϕ::Vector{T1}
    nsites   = holstein.nsites::Int
    Lτ       = holstein.Lτ::Int
    Δτ       = holstein.Δτ::T1
    ω        = holstein.ω::Vector{T1}
    τp1      = 0
    τm1      = 0
    Δτω²::T1 = 0.0

    #####################################################
    ## Calculating Derivative Phonon Action Associated ##
    ## With Local Phonon Frequency And Phonon Momentum ##
    #####################################################

    # iterating over sites in lattice
    for i in 1:holstein.lattice.nsites
        Δτω² = Δτ * ω[i] * ω[i]
        # getting the phonon fields associated with current site
        ϕi = view_by_site(ϕ,i,nsites)
        # getting view into array containing partial derivatives of phonon action for current sites
        dSbi = view_by_site(dSbose,i,nsites)
        # iterating over imaginary time axis
        for τ in 1:Lτ
            # get τ+1 accounting for periodic boundary conditions
            τp1 = τ%Lτ+1
            # get τ-1 accounting for periodic boundary conditions
            τm1 = (τ+Lτ-2)%Lτ+1
            # update the action
            dSbi[τ] += (2.0*ϕi[τ] - ϕi[τp1] - ϕi[τm1])/Δτ + Δτω²*ϕi[τ]
        end
    end

    ############################################################################
    ## Calculating Derivative Phonon Action Associated With Phonon Dispersion ##
    ############################################################################

    # checking if there are any dispersive phonon modes
    if length(holstein.ωij)>0
        ωij                = holstein.ωij::Vector{T2}
        sign_ωij           = holstein.sign_ωij::Vector{Int}
        neighbor_table_ωij = holstein.neighbor_table_ωij::Matrix{Int}
        Δτωij²::T2         = 0.0
        i                  = 0
        j                  = 0
        sgn                = 1
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