module PhononAction

using ..Models: HolsteinModel, SSHModel
using ..Utilities: get_index

export calc_Sbose, calc_dSbosedx!

"""
Calculates the pure phonon action Sbose such that exp{-Sbose}.
"""
function calc_Sbose(holstein::HolsteinModel{T1,T2}) where {T1,T2}

    x   = holstein.x::Vector{T1}
    N   = holstein.Nsites::Int
    Lτ  = holstein.Lτ::Int
    Δτ  = holstein.Δτ::T1
    ω   = holstein.ω::Vector{T1}
    ω4  = holstein.ω4::Vector{T1}
    Sbose = 0.0::T1

    # iterate over sites in lattice
    for i in 1:N
        # iterate over time slice in lattice
        for τ in 1:Lτ
            # get τ-1 accounting for periodic boundary conditions
            τm1 = (τ-2+Lτ)%Lτ+1
            # xᵢ(τ)
            xᵢτ = x[get_index(τ,i,Lτ)]
            # xᵢ(τ-1)
            xᵢτm1 = x[get_index(τm1,i,Lτ)]
            # calculate potential energy
            Sbose += ω[i]^2*xᵢτ^2/2 + ω4[i]*xᵢτ^4
            # calculate kintetic energy
            Sbose += (xᵢτ-xᵢτm1)^2/Δτ^2/2
        end
    end

    # calculate phonon potential energy associated with phonon dispersion
    if length(holstein.ωij)>0
        ωij                = holstein.ωij::Vector{T2}
        sign_ωij           = holstein.sign_ωij::Vector{Int}
        neighbor_table_ωij = holstein.neighbor_table_ωij::Matrix{Int}
        # iterate over dispersive phonon modes
        for m in 1:length(ωij)
            ωij² = ωij[m] * ωij[m]
            sgn = sign_ωij[m]
            # getting pair of neighboring sites
            i = neighbor_table_ωij[1,m]
            j = neighbor_table_ωij[2,m]
            # iterating over time slices
            for τ in 1:L
                # indexing offset into vectors associated with τ time slice
                indx_i = get_index(τ,i,Lτ)
                indx_j = get_index(τ,j,Lτ)
                # updating partial derivative
                Sbose += ωij² * ( x[indx_i] + sgn*x[indx_j] )^2/2
            end
        end
    end

    # necessary scaling factor as defintiion of Sbose includes a Δτ out front
    Sbose *= Δτ
    
    return Sbose
end

function calc_Sbose(ssh::SSHModel{T1,T2,T3}) where {T1,T2,T3}

    x   = ssh.x::Vector{T1}
    N   = ssh.Nph::Int
    Lτ  = ssh.Lτ::Int
    Δτ  = ssh.Δτ::T1
    ω   = ssh.ω::Vector{T1}
    Sbose = 0.0::T1

    # iterate over sites in lattice
    for i in 1:N
        # iterate over time slice in lattice
        for τ in 1:Lτ
            # get τ-1 accounting for periodic boundary conditions
            τm1 = (τ-2+Lτ)%Lτ+1
            # xᵢ(τ)
            xᵢτ = x[get_index(τ,i,Lτ)]
            # xᵢ(τ-1)
            xᵢτm1 = x[get_index(τm1,i,Lτ)]
            # calculate potential energy
            Sbose += ω[i]^2*xᵢτ^2/2
            # calculate kintetic energy
            Sbose += (xᵢτ-xᵢτm1)^2/Δτ^2/2
        end
    end

    # necessary scaling factor as defintiion of Sbose includes a Δτ out front
    Sbose *= Δτ
    
    return Sbose
end


"""
Calculates the dervative phonon action with respect to each phonon field and adds that value in place
to the vector dSbose.
"""
function calc_dSbosedx!(dSbose::Vector{T2}, holstein::HolsteinModel{T1,T2})  where {T1<:AbstractFloat,T2<:Number}

    @assert length(dSbose)==holstein.Ndof

    x          = holstein.x::Vector{T1}
    nsites     = holstein.Nsites::Int
    Lτ         = holstein.Lτ::Int
    Δτ         = holstein.Δτ::T1
    ω          = holstein.ω::Vector{T1}
    ω4         = holstein.ω4::Vector{T1}

    #####################################################
    ## Calculating Derivative Phonon Action Associated ##
    ## With Local Phonon Frequency And Phonon Momentum ##
    #####################################################]

    # iterating over site
    @fastmath @inbounds for site in 1:nsites
        Δτω² = Δτ * ω[site] * ω[site]
        Δτ4ω4 = Δτ * 4 * ω4[site]
        # iterating over time slices
        for τ in 1:Lτ
            # get τ+1 accounting for periodic boundary conditions
            τp1 = τ%Lτ+1
            # get τ-1 accounting for periodic boundary conditions
            τm1 = (τ-2+Lτ)%Lτ+1
            # indexing offset into vectors associated with τ time slice
            indx_τ = get_index(τ,site,Lτ)
            # indexing offset into vectors associated with τ+1 time slice
            indx_τp1 = get_index(τp1,site,Lτ)
            # indexing offset into vectors associated with τ-1 time slice
            indx_τm1 = get_index(τm1,site,Lτ)
            # phonon field at current time slice
            xτ = x[indx_τ]
            # updating partial derivative
            dSbose[indx_τ] += Δτω² * xτ # derivative of Δτ⋅ω²/2⋅x² term
            dSbose[indx_τ] += Δτ4ω4 * xτ * xτ * xτ # derivative of Δτ⋅ω₄⋅x⁴ term.
            dSbose[indx_τ] -= ( x[indx_τp1] + x[indx_τm1] - 2.0*xτ )/Δτ
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
                Δ = Δτωij² * ( x[indx_i] + sgn*x[indx_j] )
                dSbose[indx_i] += Δ
                dSbose[indx_j] += sgn*Δ
            end
        end
    end

    return nothing
end

function calc_dSbosedx!(dSbose::Vector{T2}, ssh::SSHModel{T1,T2,T3})  where {T1,T2,T3}

    @assert length(dSbose)==ssh.Ndof

    x   = ssh.x::Vector{T1}
    Nph = ssh.Nph::Int
    Lτ  = ssh.Lτ::Int
    Δτ  = ssh.Δτ::T1
    ω   = ssh.ω::Vector{T1}

    #####################################################
    ## Calculating Derivative Phonon Action Associated ##
    ## With Local Phonon Frequency And Phonon Momentum ##
    #####################################################]

    # iterating over site
    @fastmath @inbounds for i in 1:Nph
        Δτω² = Δτ * ω[i] * ω[i]
        # iterating over time slices
        for τ in 1:Lτ
            # get τ+1 accounting for periodic boundary conditions
            τp1      = mod1(τ+1,Lτ)
            # get τ-1 accounting for periodic boundary conditions
            τm1      = mod1(τ-1,Lτ)
            # indexing offset into vectors associated with τ time slice
            indx_τ   = get_index(τ,i,Lτ)
            # indexing offset into vectors associated with τ+1 time slice
            indx_τp1 = get_index(τp1,i,Lτ)
            # indexing offset into vectors associated with τ-1 time slice
            indx_τm1 = get_index(τm1,i,Lτ)
            # phonon field at current time slice
            xτ       = x[indx_τ]
            # updating partial derivative
            dSbose[indx_τ] += Δτω² * xτ # derivative of Δτ⋅ω²/2⋅x² term
            dSbose[indx_τ] -= ( x[indx_τp1] + x[indx_τm1] - 2.0*xτ )/Δτ
        end
    end

    return nothing
end

end