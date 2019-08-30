module PhononAction

using Langevin.HolsteinModels: HolsteinModel, get_index

export calc_dSbosedϕ!

"""
Calculates the dervative phonon action with respect to each phonon field and adds that value in place
to the vector dSbose.
"""
function calc_dSbosedϕ!(dSbose::Vector{T2}, holstein::HolsteinModel{T1,T2})  where {T1<:AbstractFloat,T2<:Number}

    @assert length(dSbose)==holstein.nindices

    ϕ          = holstein.ϕ::Vector{T1}
    nsites     = holstein.nsites::Int
    Lτ         = holstein.Lτ::Int
    Δτ         = holstein.Δτ::T1
    ω          = holstein.ω::Vector{T1}
    τp1        = 0
    indx_τ     = 0
    indx_τp1   = 0
    Δτω²::T1   = 0.0
    Δ::T2      = 0.0

    #####################################################
    ## Calculating Derivative Phonon Action Associated ##
    ## With Local Phonon Frequency And Phonon Momentum ##
    #####################################################

    # iterating over sites
    for site in 1:nsites
        Δτω² = Δτ * ω[site] * ω[site]
        # iterating over time slices
        for τ in 1:Lτ
            # get τ+1 accounting for periodic boundary conditions
            τp1 = τ%Lτ+1
            # indexing offset into vectors associated with τ time slice
            indx_τ = get_index(τ,site,Lτ)
            # indexing offset into vectors associated with τ+1 time slice
            indx_τp1 = get_index(τp1,site,Lτ)
            # Δ = ( ϕᵢ(τ+1) - ϕᵢ(τ) ) / Δτ
            Δ = ( ϕ[indx_τp1] - ϕ[indx_τ] ) / Δτ
            # updating partial derivative ∂Sb∂ϕᵢ(τ)
            dSbose[indx_τ]   += -Δ + Δτω² * ϕ[indx_τ]
            # updating partial derivative ∂Sb∂ϕᵢ(τ+1)
            dSbose[indx_τp1] +=  Δ
        end
    end

    #############################################################
    ## Calculating Derivative Of Phonon Action Associated With ##
    ##               Dispersive Phonon Modes                   ##
    #############################################################

    if length(holstein.ωij)>0
        ωij                = holstein.ωij::Vector{T2}
        sign_ωij           = holstein.sign_ωij::Vector{Int}
        neighbor_table_ωij = holstein.neighbor_table_ωij::Matrix{Int}
        Δτωij²::T2         = 0.0
        i                  = 0
        j                  = 0
        sgn                = 1
        indx_i             = 0
        indx_j             = 0
        # iterate over dispersive phonon modes
        for m in 1:length(ωij)
            Δτωij² = Δτ * ωij[m] * ωij[m]
            sgn = sign_ωij[m]
            # getting pair of neighboring sites
            i = neighbor_table_ωij[1,m]
            j = neighbor_table_ωij[2,m]
            # iterating over time slices
            for τ in 1:Lτ
                # indexing offset into vectors associated with τ time slice
                indx_i = get_index(τ,i,Lτ)
                indx_j = get_index(τ,j,Lτ)
                # updating partial derivative
                Δ = Δτωij² * ( ϕ[indx_i] + sgn*ϕ[indx_j] )
                dSbose[offset_τ+i] += Δ
                dSbose[offset_τ+j] += sgn*Δ
            end
        end
    end

    return nothing
end

end