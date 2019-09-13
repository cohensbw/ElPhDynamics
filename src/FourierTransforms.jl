module FourierTransforms

using Langevin.Lattices: Lattice

export calc_fourier_transform_coefficients, fourier_transform!

"""
Calculates the cofficients needed in doing the fourier transforms.
"""
function calc_fourier_transform_coefficients(lattice::Lattice{T})::Array{Complex{T},6} where {T<:AbstractFloat}

    # number of unit cells in lattice <==> number of k-points
    N = lattice.ncells

    # size of finite lattice
    L1  = lattice.L1::Int
    L2  = lattice.L2::Int
    L3  = lattice.L3::Int
    LLL = L1*L2*L3

    # array to hold fourier transform coefficients
    coeff = zeros(Complex{T},(L1,L2,L3,L1,L2,L3))

    # calculating corresponding fourier coefficent
    for k3 in 0:L3-1
        for k2 in 0:L2-1
            for k1 in 0:L1-1
                for r3 in 0:L3-1
                    for r2 in 0:L2-1
                        for r1 in 0:L1-1
                            coeff[r1+1,r2+1,r3+1,k1+1,k2+1,k3+1] = exp(im*2*π*(r1*k1/L1+r2*k2/L2+r3*k3/L3))
                        end
                    end
                end
            end
        end
    end
    return coeff
end


"""
Calculates the fourier transform of a measurement.
The assumed ordering of the indices and size of the arrays in both position space
and momentum space is [L1,L2,L3,norbits,norbits,Lτ].
"""
function fourier_transform!(meas_kspace::Array{T,6},meas_rspace::Array{T,6},coeff::Array{Complex{T}}) where {T<:AbstractFloat}

    L1      = size(meas_rspace,1)
    L2      = size(meas_rspace,2)
    L3      = size(meas_rspace,3)
    norbits = size(meas_rspace,5)
    Lτ      = size(meas_rspace,6)
    temp::Complex{T} = 0.0
    # iterating oer imaginary time slices
    for τ in 0:Lτ-1
        # iterating over orbital combiniations
        for orbit1 in 1:norbits
            for orbit2 in 1:norbits
                # iterating over k-points
                for k3 in 0:L3-1
                    for k2 in 0:L2-1
                        for k1 in 0:L1-1
                            # reseting temporary value
                            temp = 0.0+0.0*im
                            # iterating over displacement vectors
                            for r3 in 0:L3-1
                                for r2 in 0:L2-1
                                    for r1 in 0:L1-1
                                        temp += coeff[r1+1,r2+1,r3+1,k1+1,k2+1,k3+1] * meas_rspace[r1+1,r2+1,r3+1,orbit2,orbit1,τ+1]
                                    end
                                end
                            end
                            # recording fourier transform of measurement for current k-point
                            meas_kspace[k1+1,k2+1,k3+1,orbit2,orbit1,τ+1] = real(temp)
                        end
                    end
                end
            end
        end
    end
    meas_kspace
end

end