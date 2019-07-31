import Base: eltype, size, *
import LinearAlgebra: mul!

using Langevin.Checkerboard: checkerboard_mul!, checkerboard_transpose_mul!
using Langevin.QuantumLattices: view_by_site, view_by_τ

export mulM!, mulMt!


# overload `eltype` from Base
function eltype(holstein::HolsteinModel{T})::DataType where {T<:AbstractFloat}

    return Complex{T}
end


# overloading `size` from Base
function size(holstein::HolsteinModel{T},dim::Int=1) where {T<:AbstractFloat}

    @assert 1<=dim<=2
    return holstein.qlattice.nindices
end


# overloading `*` operator from Base
function *(holstein::HolsteinModel{T},v::AbstractVector{Complex{T}})::Vector{Complex{T}} where {T<:AbstractFloat}

    y = Vector{Complex{T}}(undef,size(holstein))
    mul!(y,holstein,v)
    return y
end


# overload mul! function from LinearAlgebra so that it calculates y = MᵀM⋅v
function mul!(y::AbstractVector{Complex{T}},holstein::HolsteinModel{T},v::AbstractVector{Complex{T}})  where {T<:AbstractFloat}

    # y' = M⋅v
    mulM!(holstein.temporary_vector, holstein, v)

    # y = MᵀM⋅v = Mᵀ⋅y'
    mulMt!(y, holstein, holstein.temporary_vector)
end


"""
Perform the multiplication y = M⋅v
"""
function mulM!(y::AbstractVector{Complex{T}},holstein::HolsteinModel{T},v::AbstractVector{Complex{T}})  where {T<:AbstractFloat}

    neighbor_table_tij = holstein.neighbor_table_tij::Matrix{Int}
    coshtij = holstein.coshtij::Vector{Complex{T}}
    sinhtij = holstein.sinhtij::Vector{Complex{T}}
    expnΔτV = holstein.expnΔτV::Vector{Complex{T}}
    Lτ = holstein.qlattice.Lτ::Int
    nsites = holstein.lattice.nsites::Int
    τp1 = 1

    ####################################
    ## PERFORM MULTIPLICATION y = M⋅v ##
    ####################################

    # iterate over imaginary time axis
    for τ in 1:Lτ

        # Notes:
        # • y(τ) = [M⋅v](τ) = v(τ) - B(τ+1)⋅v(τ+1) for τ < Lτ
        # • y(τ) = [M⋅v](τ) = v(τ) + B(1)⋅v(1)     for τ = Lτ
        # • B(τ) = exp{-Δτ⋅K} exp{-Δτ⋅V[ϕ(τ)]}
        # • exp{-Δτ⋅K} is given by the checkerboard approximation.

        # get the τ+1 time slice account for periodic boundary conditions
        τp1 = τ%Lτ+1

        # get a view into y for current time slice τ
        yτ = view_by_τ(y,τ,nsites)
        # get a view into v for current time slice τ
        vτ = view_by_τ(v,τ,nsites)
        # getting view into v for time slice τ+1
        vτp1 = view_by_τ(v,τp1,nsites)
        # geting view into exp{-Δτ⋅V[ϕ(τ+1)]} matrix
        expnΔτV_τp1 = view_by_τ(expnΔτV,τp1,nsites)

        # first we need to do the multiplication exp{-Δτ⋅V[ϕ(τ+1)]}⋅v(τ+1).
        @. yτ = expnΔτV_τp1 * vτp1

        # next we need to multiply by exp{-Δτ⋅K}
        checkerboard_mul!(yτ,neighbor_table_tij,coshtij,sinhtij)
        # at this point y(τ) = B(τ+1)⋅v(τ+1)

        # finish up the multiplication to get final y(τ) vector
        if τ<Lτ
            @. yτ = vτ - yτ
        else
            yτ .+= vτ
        end

    end

    return nothing
end


"""
Perform the multiplication y = Mᵀ⋅v
"""
function mulMt!(y::AbstractVector{Complex{T}},holstein::HolsteinModel{T},v::AbstractVector{Complex{T}})  where {T<:AbstractFloat}

    neighbor_table_tij = holstein.neighbor_table_tij::Matrix{Int}
    coshtij = holstein.coshtij::Vector{Complex{T}}
    sinhtij = holstein.sinhtij::Vector{Complex{T}}
    expnΔτV = holstein.expnΔτV::Vector{Complex{T}}
    Lτ = holstein.qlattice.Lτ::Int
    nsites = holstein.lattice.nsites::Int
    τm1 = 1
    sgn = 1

    #####################################
    ## PERFORM MULTIPLICATION y = Mᵀ⋅v ##
    #####################################

    # iterate over imaginary time axis
    for τ in 1:Lτ

        # Notes:
        # • y(τ) = [Mᵀ⋅v](τ) = v(τ) - Bᵀ(τ)⋅v(τ-1)  for τ > 1
        # • y(τ) = [Mᵀ⋅v](τ) = v(τ) + Bᵀ(τ)⋅v(Lτ)   for τ = 1
        # • Bᵀ(τ) = exp{-Δτ⋅V[ϕ(τ)]} [exp{-Δτ⋅K}]ᵀ
        # • exp{-Δτ⋅K} is given by the checkerboard approximation.

        # get the τ-1 time slice account for periodic boundary conditions
        τm1 = (τ+Lτ-2)%Lτ+1

        # get a view into y for current time slice τ
        yτ = view_by_τ(y,τ,nsites)
        # get a view into v for current time slice τ
        vτ = view_by_τ(v,τ,nsites)
        # getting view into v for time slice τ-1
        vτm1 = view_by_τ(v,τm1,nsites)
        # geting view into exp{-Δτ⋅V[ϕ(τ)]} matrix
        expnΔτV_τ = view_by_τ(expnΔτV,τ,nsites)

        # set y(τ) = v(τ-1)
        yτ .= vτm1

        # first we need to multiply by the transpose of exp{-Δτ⋅K}
        checkerboard_transpose_mul!(yτ,neighbor_table_tij,coshtij,sinhtij)

        # next multiply by exp{-Δτ⋅V[ϕ(τ)]} matrix.
        yτ .*= expnΔτV_τ
        # at this point y(τ) = Bᵀ(τ)⋅v(τ-1)

        # finish up the multiplication to get final y(τ) vector
        if τ>1
            @. yτ = vτ - yτ
        else
            yτ .+= vτ
        end

    end

    return nothing
end