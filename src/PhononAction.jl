module PhononAction

using ..Models: HolsteinModel, SSHModel
using ..Utilities: get_index

export calc_Sb, calc_dSbdxdx!

"""
Calculates the pure phonon action Sb such that exp{-Sb}.
"""
function calc_Sb(holstein::HolsteinModel{T1,T2}) where {T1,T2}

    x   = holstein.x::Vector{T1}
    N   = holstein.Nsites::Int
    Lτ  = holstein.Lτ::Int
    Δτ  = holstein.Δτ::T1
    ω   = holstein.ω::Vector{T1}
    ω₄  = holstein.ω₄::Vector{T1}
    Sb = 0.0::T1

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
            Sb += ω[i]^2*xᵢτ^2/2 + ω₄[i]*xᵢτ^4
            # calculate kintetic energy
            Sb += (xᵢτ-xᵢτm1)^2/Δτ^2/2
        end
    end

    # calculate phonon potential energy associated with phonon dispersion
    if length(holstein.ωᵢⱼ)>0
        ωᵢⱼ                = holstein.ωᵢⱼ::Vector{T2}
        sign_ωᵢⱼ           = holstein.sign_ωᵢⱼ::Vector{Int}
        neighbor_table_ωᵢⱼ = holstein.neighbor_table_ωᵢⱼ::Matrix{Int}
        # iterate over dispersive phonon modes
        for m in 1:length(ωᵢⱼ)
            ωᵢⱼ² = ωᵢⱼ[m] * ωᵢⱼ[m]
            sgn = sign_ωᵢⱼ[m]
            # getting pair of neighboring sites
            i = neighbor_table_ωᵢⱼ[1,m]
            j = neighbor_table_ωᵢⱼ[2,m]
            # iterating over time slices
            for τ in 1:L
                # indexing offset into vectors associated with τ time slice
                indx_i = get_index(τ,i,Lτ)
                indx_j = get_index(τ,j,Lτ)
                # updating partial derivative
                Sb += ωᵢⱼ² * ( x[indx_i] + sgn*x[indx_j] )^2/2
            end
        end
    end

    # necessary scaling factor as defintiion of Sb includes a Δτ out front
    Sb *= Δτ
    
    return Sb
end

function calc_Sb(ssh::SSHModel{T1,T2,T3}) where {T1,T2,T3}

    x  = ssh.x::Vector{T1}
    N  = ssh.Nph::Int
    Lτ = ssh.Lτ::Int
    Δτ = ssh.Δτ::T1
    ω  = ssh.ω::Vector{T1}
    ω₄ = ssh.ω₄::Vector{T1}
    Sb = 0.0

    # iterate over sites in lattice
    for i in 1:N
        # iterate over time slice in lattice
        for τ in 1:Lτ
            # get τ-1 accounting for periodic boundary conditions
            τm1   = mod1(τ-1,Lτ)
            # get field
            field = get_index(τ,i,Lτ)
            # xᵢ(τ)
            xᵢτ   = x[field]
            # xᵢ(τ-1)
            xᵢτm1 = x[get_index(τm1,i,Lτ)]
            # calculate potential energy
            val   = Δτ*ω[i]^2*xᵢτ^2/2 + Δτ*ω₄[i]*xᵢτ^4
            # calculate kintetic energy
            val  += (xᵢτ-xᵢτm1)^2/Δτ/2
            # add to Sb total, normalizing by the number of equivalent fields there are
            Sb   += val
        end
    end
    
    return Sb
end


"""
Calculates the dervative phonon action with respect to each phonon field and adds that value in place
to the vector dSbdx.
"""
function calc_dSbdx!(dSbdx::Vector{T2}, holstein::HolsteinModel{T1,T2})  where {T1<:AbstractFloat,T2<:Number}

    @assert length(dSbdx)==holstein.Ndof

    x          = holstein.x::Vector{T1}
    nsites     = holstein.Nsites::Int
    Lτ         = holstein.Lτ::Int
    Δτ         = holstein.Δτ::T1
    ω          = holstein.ω::Vector{T1}
    ω₄         = holstein.ω₄::Vector{T1}

    #####################################################
    ## Calculating Derivative Phonon Action Associated ##
    ## With Local Phonon Frequency And Phonon Momentum ##
    #####################################################]

    # iterating over site
    @fastmath @inbounds for site in 1:nsites
        Δτω² = Δτ * ω[site] * ω[site]
        Δτ4ω₄ = Δτ * 4 * ω₄[site]
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
            dSbdx[indx_τ] += Δτω² * xτ # derivative of Δτ⋅ω²/2⋅x² term
            dSbdx[indx_τ] += Δτ4ω₄ * xτ * xτ * xτ # derivative of Δτ⋅ω₄⋅x⁴ term.
            dSbdx[indx_τ] -= ( x[indx_τp1] + x[indx_τm1] - 2.0*xτ )/Δτ
        end
    end

    #############################################################
    ## Calculating Derivative Of Phonon Action Associated With ##
    ##               Dispersive Phonon Modes                   ##
    #############################################################

    if length(holstein.ωᵢⱼ)>0
        ωᵢⱼ                = holstein.ωᵢⱼ::Vector{T2}
        sign_ωᵢⱼ           = holstein.sign_ωᵢⱼ::Vector{Int}
        neighbor_table_ωᵢⱼ = holstein.neighbor_table_ωᵢⱼ::Matrix{Int}
        # iterate over dispersive phonon modes
        for m in 1:length(ωᵢⱼ)
            Δτωᵢⱼ² = Δτ * ωᵢⱼ[m] * ωᵢⱼ[m]
            sgn = sign_ωᵢⱼ[m]
            # getting pair of neighboring sites
            i = neighbor_table_ωᵢⱼ[1,m]
            j = neighbor_table_ωᵢⱼ[2,m]
            # iterating over time slices
            for τ in 1:Lτ
                # indexing offset into vectors associated with τ time slice
                indx_i = get_index(τ,i,Lτ)
                indx_j = get_index(τ,j,Lτ)
                # updating partial derivative
                Δ = Δτωᵢⱼ² * ( x[indx_i] + sgn*x[indx_j] )
                dSbdx[indx_i] += Δ
                dSbdx[indx_j] += sgn*Δ
            end
        end
    end

    return nothing
end

function calc_dSbdx!(dSbdx::Vector{T2}, ssh::SSHModel{T1,T2,T3})  where {T1,T2,T3}

    @assert length(dSbdx)==ssh.Ndof

    x   = ssh.x::Vector{T1}
    Nph = ssh.Nph::Int
    Lτ  = ssh.Lτ::Int
    Δτ  = ssh.Δτ::T1
    ω   = ssh.ω::Vector{T1}
    ω₄  = ssh.ω₄::Vector{T1}

    #####################################################
    ## Calculating Derivative Phonon Action Associated ##
    ## With Local Phonon Frequency And Phonon Momentum ##
    #####################################################

    # iterating over phonons in lattice
    @fastmath @inbounds for i in 1:Nph
        Δτω²  = Δτ * ω[i] * ω[i]
        Δτ4ω₄ = Δτ * 4 * ω₄[i]
        # iterating over time slices
        for τ in 1:Lτ
            # get τ+1 accounting for periodic boundary conditions
            τp1       = mod1(τ+1,Lτ)
            # get τ-1 accounting for periodic boundary conditions
            τm1       = mod1(τ-1,Lτ)
            # indexing offset into vectors associated with τ time slice
            field_τ   = get_index(τ,i,Lτ)
            # indexing offset into vectors associated with τ+1 time slice
            field_τp1 = get_index(τp1,i,Lτ)
            # indexing offset into vectors associated with τ-1 time slice
            field_τm1 = get_index(τm1,i,Lτ)
            # phonon field at current time slice
            xτ        = x[field_τ]
            # updating partial derivative
            val  = Δτω² * xτ # derivative of Δτ⋅ω²/2⋅x² term
            val += Δτ4ω₄ * xτ * xτ * xτ # derivative of Δτ⋅ω₄⋅x⁴ term.
            val -= ( x[field_τp1] + x[field_τm1] - 2.0*xτ )/Δτ # kinetic energy term
            # increment total derivative value
            dSbdx[field_τ] += val
        end
    end

    return nothing
end

end