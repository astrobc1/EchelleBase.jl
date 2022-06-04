module Maths

import DataInterpolations
using NaNStatistics
using LoopVectorization
using StatsBase
using Infiltrator
using Polynomials

const SPEED_OF_LIGHT_MPS = 299792458.0
const TWO_SQRT_2LOG2 = 2 * sqrt(2 * log(2))


"""
    rmsloss(residuals::AbstractArray{<:Real}, [weights::AbstractArray{<:Real}=nothing]; flag_worst::Int=0, remove_edges::Int=0)
Computes the root mean squared error (RMS) loss. Weights can also be provided, otherwise uniform weights will be used.
"""
function rmsloss(residuals::AbstractArray{<:Real}, weights::Union{AbstractArray{<:Real}, Nothing}=nothing; flag_worst::Int=0, remove_edges::Int=0)

    # Get good data
    if !isnothing(weights)
        good = findall(isfinite.(residuals) .&& isfinite.(weights) .&& (weights .> 0))
        rr, ww = residuals[good], weights[good]
    else
        good = findall(isfinite.(residuals))
        rr = residuals[good]
        ww = ones(length(rr))
    end
    
    # Ignore worst N pixels
    if flag_worst > 0
        ss = sortperm(abs.(rr))
        rr[ss[end-flag_worst+1:end]] .= NaN
        if !isnothing(weights)
            ww[ss[end-flag_worst+1:end]] .= 0
        end
    end

    # Remove edges
    if remove_edges > 0
        rr[1:remove_edges] .= NaN
        rr[end-remove_edges+1:end] .= NaN
        if !isnothing(weights)
            ww[1:remove_edges] .= 0
            ww[end-remove_edges+1:end] .= 0
        end
    end
        
    # Compute rms
    rms = sqrt(nansum(ww .* rr.^2) / nansum(ww))

    # Return
    return rms
end

"""
    redχ2loss(residuals::AbstractArray{<:Real}, [weights::AbstractArray{<:Real}=nothing]; flag_worst::Int=0, remove_edges::Int=0)
Computes the reduced chi square loss. Weights can also be provided, otherwise uniform weights will be used.
"""
function redχ2loss(residuals::AbstractArray{<:Real}, errors::AbstractArray{<:Real}, mask::AbstractArray{<:Real}=nothing; flag_worst=0, remove_edges=0, ν=nothing)

    # Compute diffs2
    if isnothing(mask)
        good = findall(isfinite.(residuals) .&& isfinite.(errors) .&& (mask .== 1))
        residuals, errors, mask = residuals[good], errors[good], mask[good]
    else
        good = findall(isfinite.(residuals) .&& isfinite.(errors))
        residuals, errors = residuals[good], errors[good]
        mask = ones(length(residuals))
    end

    # Remove edges
    if remove_edges > 0
        residuals[1:remove_edges-1] .= NaN
        residuals[end-remove_edges+1:end] .= NaN
        mask[1:remove_edges-1] .= 0
        mask[end-remove_edges+1:end] .= 0
    end
    
    # Ignore worst N pixels
    if flag_worst > 0
        ss = sortperm(abs.(residuals))
        residuals[ss[end-flag_worst+1:end]] .= NaN
        mask[ss[end-flag_worst+1:end]] .= 0
    end

    # Degrees of freedom
    if isnothing(ν)
        ν = sum(mm) - 2 * remove_edges - flag_worst - 1
    else
        ν = ν - 2 * remove_edges - flag_worst
    end

    @assert ν > 0

    # Compute chi2
    redχ² = nansum(diffs2) / ν

    # Return
    return redχ²
end


"""
    doppler_shift_λ(λ, vel::Real; mode::String="sr")
Applies a Doppler shift of velocitiy `vel` to `λ`. The special relativistic equation is used if `mode="sr"`, otherwise the classical equation is used.
"""
function doppler_shift_λ(λ, vel::Real; mode="sr")
    if lowercase(mode) == "sr"
        β = vel ./ SPEED_OF_LIGHT_MPS
        return λ .* sqrt((1 .+ β) ./ (1 .- β))
    else
        return λ .* (1 .+ vel ./ SPEED_OF_LIGHT_MPS)
    end
end

"""
    doppler_shift_flux(λ, flux; [mode::String="sr"])
Applies a Doppler shift of velocitiy `vel` to `λ`. The special relativistic equation is used if `mode="sr"`, otherwise the classical equation is used.
"""
function doppler_shift_flux(λ, flux, vel::Real)

    # The shifted wave
    λ_shifted = doppler_shift_λ(λ, vel)

    # Interpolate the flux
    flux_out = cspline_interp(λ_shifted, flux, λ)

    # Return
    return flux_out
end

"""
    cspline_interp(x, y, xnew)
Cubic spline interpolation without extrapolation using DataInterpolations.jl.
"""
function cspline_interp(x, y, xnew)
    good = findall(isfinite.(y))
    xx = @view x[good]
    yy = @view y[good]
    y_out = DataInterpolations.CubicSpline(yy, xx).(xnew)
    bad = findall((xnew .< x[good[1]]) .|| (xnew .> x[good[end]]))
    y_out[bad] .= NaN
    return y_out
end

"""
    lin_interp(x, y, xnew)
Linear interpolation without extrapolation using DataInterpolations.jl.
"""
function lin_interp(x, y, xnew)
    good = findall(isfinite.(y))
    xx = @view x[good]
    yy = @view y[good]
    y_out = DataInterpolations.LinearInterpolation(yy, xx).(xnew)
    bad = findall((xnew .< x[good[1]]) .|| (xnew .> x[good[end]]))
    y_out[bad] .= NaN
    return y_out
end

function gauss(x, a, μ, σ)
    return @. a * exp(-0.5 * ((x - μ) / σ)^2)
end

"""
    median_filter1d(x::AbstractVector, width::Real)
A standard median filter where x_out[i] = median(x[i-w2:i+w2]) where w2 = floor(width / 2).
"""
function median_filter1d(x::AbstractVector, width::Real)
    @assert isodd(width)
    nx = length(x)
    x_out = fill(NaN, nx)
    w2 = Int(floor(width / 2))
    for i=1:nx
        k1 = max(i - w2, 1)
        k2 = min(i + w2, nx)
        x_out[i] = NaNStatistics.nanmedian(@view x[k1:k2])
    end
    return x_out
end

"""
    median_filter2d(x::AbstractVector, width::Real)
A standard median filter where x_out[i, j] = median(x[i-w2:i+w2, j-w2:j+w2]) where w2 = floor(width / 2).
"""
function median_filter2d(x::AbstractVector, width::Real)
    @assert isodd(width)
    ny, nx = size(x)
    x_out = fill(NaN, (ny, nx))
    w2 = Int(floor(width / 2))
    for i=1:nx
        for j=1:ny
            kx1 = max(i - w2, 1)
            kx2 = min(i + w2, nx)
            ky1 = max(j - w2, 1)
            ky2 = min(j + w2, ny)
            x_out[j, i] = NaNStatistics.nanmedian((@view x[ky1:ky2, kx1:kx2]))
        end
    end
    return x_out
end

"""
    chebval(x::Real, n::Int)
Computes the Chebyshev polynomial of degree n at value x.
"""
function chebval(x::Real, n::Int)
    coeffs = zeros(n+1)
    coeffs[n+1] = 1.0
    return ChebyshevT(coeffs).(x)
end

"""
    robust_σ(x::AbstractArray; [w::AbstractArray] nσ::Real=4)
Computes a robust standard deviation value by flagging values through the median absolute deviation. 
"""
function robust_σ(x::AbstractArray; w=Union{Nothing, AbstractArray}, nσ::Real=4)
    med = weighted_median(x, w=w)
    adevs = abs.(med .- x)
    mad = weighted_median(adevs, w=w)
    good = findall(adevs .< 1.4826 * mad * nσ)
    if length(good) > 1
        return nanstd(@view x[good])
    else
        return NaN
    end
end

"""
    robust_σ(x::AbstractArray; [w::AbstractArray] nσ::Real=4)
Computes a robust standard mean and deviation value by flagging values through the median absolute deviation. 
"""
function robust_stats(x; w=nothing, nσ=4)
    med = weighted_median(x, w=w)
    adevs = abs.(med .- x)
    mad = weighted_median(adevs, w=w)
    good = findall(adevs .< 1.4826 * mad * nσ)
    if length(good) > 1
        return nanmean(@view x[good]), nanstd(@view x[good])
    else
        return NaN, NaN
    end
end

"""
    robust_σ(x::AbstractArray; [w::AbstractArray] nσ::Real=4)
Computes a robust standard mean value by flagging values through the median absolute deviation. 
"""
function robust_μ(x; w=nothing, nσ=4)
    med = weighted_median(x, w=w)
    adevs = abs.(med .- x)
    mad = weighted_median(adevs, w=w)
    good = findall(adevs .< 1.4826 * mad * nσ)
    if length(good) > 1
        return nanmean(@view x[good])
    else
        return NaN
    end
end

"""
    convolve1d(x::AbstractVector{<:Real}, k::AbstractArray{<:Real})
    1d direct convolution.
"""
function convolve1d(x::AbstractVector{<:Real}, k::AbstractArray{<:Real})
    nx = length(x)
    nk = length(k)
    n_pad = Int(floor(nk / 2))
    out = zeros(nx)
    kf = @view k[end:-1:1]
    valleft = x[1]
    valright = x[end]
    
    # Left values
    @inbounds for i=1:n_pad
        s = 0.0
        for j=1:nk
            ii = i - n_pad + j + 1
            if ii < 1
                s += valleft * kf[j]
            else
                s += x[ii] * kf[j]
            end
        end
        out[i] = s
    end

    # Middle values
    @turbo for i=n_pad+1:nx-n_pad
        s = 0.0
        for j=1:nk
            s += x[i - n_pad + j - 1] * kf[j]
        end
        out[i] = s
    end

    # Right values
    @inbounds for i=nx-n_pad+1:nx
        s = 0.0
        for j=1:nk
            ii = i - n_pad + j + 1
            if ii > nx
                s += valright * kf[j]
            else
                s += x[ii] * kf[j]
            end
        end
        out[i] = s
    end

    # Return out
    return out

end

function hermfun(x, deg)
    herm0 = π^-0.25 .* exp.(-0.5 .* x.^2)
    herm1 = sqrt(2) .* herm0 .* x
    if deg == 0
        herm = herm0
        return herm
    elseif deg == 1
        return [herm0 herm1]
    else
        herm = zeros(length(x), deg+1)
        herm[:, 1] .= herm0
        herm[:, 2] .= herm1
        for k=3:deg+1
            herm[:, k] .= sqrt(2 / (k - 1)) .* (x .* herm[:, k-1] .- sqrt((k - 2) / 2) .* herm[:, k-2])
        end
        return herm
    end
end

function weighted_mean(x, w)
    good = findall(isfinite.(x) .&& (w .> 0) .&& isfinite.(w))
    if length(good) > 0
        xx = @view x[good]
        ww = @view w[good]
        return sum(xx .* ww) / sum(ww)
    else
        return NaN
    end
end

function weighted_stddev(x, w)
    good = findall(isfinite.(x) .&& (w .> 0) .&& isfinite.(w))
    xx = x[good]
    ww = w[good]
    ww ./= sum(ww)
    μ = weighted_mean(xx, ww)
    dev = xx .- μ
    bias_estimator = 1.0 - sum(ww.^2)
    var = sum(dev .^2 .* ww) / bias_estimator
    return sqrt(var)
end

function weighted_median(x; w=nothing, p=0.5)
    if isnothing(w)
        w = ones(size(x))
    end
    good = findall(isfinite.(x) .&& (w .> 0) .&& isfinite.(w))
    if length(good) > 0
        xx = @view x[good]
        ww = @view w[good]
        return quantile(xx, Weights(ww), p)
    else
        return NaN
    end
end

# function cross_correlate_interp(x1, y1, x2, y2, lags; kind="rms")

#     # Shifts y2 and compares it to y1
#     nx1 = length(x1)
#     nx2 = length(x2)
  
#     nlags = length(lags)
#     kind = lowercase(kind)
    
#     ccf = fill(NaN, nlags)
#     y2_shifted = fill(NaN, nx1)
#     weights = ones(nx1)
#     for i=1:nlags
#         y2_shifted .= lin_interp(x2 .+ lags[i], y2, x1)
#         good = findall(isfinite.(y1) .&& isfinite.(y2_shifted))
#         if length(good) < 3
#             continue
#         end
#         weights .= 1
#         bad = findall(.~isfinite.(y1) .|| .~isfinite.(y2_shifted))
#         weights[bad] .= 0
#         if kind == "rms"
#             ccf[i] = sqrt(nansum(weights .* (y1 .- y2_shifted).^2) / nansum(weights))
#         else
#             ccf[i] = nansum(y1 .* y2_shifted .* weights) / nansum(weights)
#         end
#     end

#     return ccf

# end


function nanargmaximum(x)
    k = 1
    for i=1:length(x)
        if x[i] > x[k] && isfinite(x[k])
            k = i
        end
    end
    return k
end

function nanargminimum(x)
    k = 1
    for i=1:length(x)
        if x[i] < x[k] && isfinite(x[k])
            k = i
        end
    end
    return k
end

function poly_filter(x, y; width, deg)
    nx = length(x)
    y_out = fill(NaN, nx)
    for i=1:nx
        use = findall((abs.(x .- x[i]) .<= width) .&& isfinite.(y))
        if length(use) >= deg
            try
                y_out[i] = Polynomials.fit(x[use], y[use], deg)(x[i])
            catch
                nothing
            end
        end
    end
    return y_out
end

# lf = l0 * e^(dv/c)
# dl = l0 * (e^(dv/c) - 1)
# dl / l0 = (e^(dv/c) - 1)
# dl / l0 + 1 = e^(dv/c)
# ln(dl / l0 + 1) = dv/c
# c * ln(dl / l0 - 1) = dv
"""
    δλ2δv(δv::Real, λ::Real)
    Convert a change in wavelength (`δλ`; any units) to velocity (m/s)
"""
function δλ2δv(δλ::Real, λ::Real)
    x = δλ / λ + 1
    if x > 0
        return SPEED_OF_LIGHT_MPS * log(x)
    else
        return NaN
    end
end

# function δλ2δv(δλ, λ)
#     x = δλ ./ λ .+ 1
#     bad = findall(x .<= 0)
#     x[bad] .= NaN
#     return @. SPEED_OF_LIGHT_MPS * log(x)
# end

# dl = l0 * (e^(dv/c) - 1)

"""
δv2δλ(δv::Real, λ::Real)
    Convert a change in velocity (m/s) to wavelength (units of λ)
"""
function δv2δλ(δv::Real, λ::Real)
    return λ * (exp(δv / SPEED_OF_LIGHT_MPS) - 1)
end

function generalized_median_filter1d(x; width, p=0.5)
    nx = length(x)
    y = fill(NaN, nx)
    for i=1:nx
        ilow = Int(max(1, i - ceil(width / 2)))
        ihigh = Int(min(i + floor(width / 2), nx))
        if length(findall(isfinite.(@view x[ilow:ihigh]))) > 0
            y[i] = weighted_median((@view x[ilow:ihigh]), p=p)
        end
    end
    return y
end

"""
cross_correlate_doppler(λ1::AbstractVector, f1::AbstractVector, λ2::AbstractVector, f2::AbstractVector, vels::AbstractVector)
    Compute the cross-correlation function as a function of the Doppler shift `vels` between two signals.
"""
function cross_correlate_doppler(λ1::AbstractVector, f1::AbstractVector, λ2::AbstractVector, f2::AbstractVector, vels::AbstractVector)
    xc = fill(NaN, length(vels))
    vec_cross = fill(NaN, length(f1))
    for i=1:length(vels)
        f2s = doppler_shift_flux(λ2, f2, vels[i])
        f2s = lin_interp(λ2, f2s, λ1)
        vec_cross .= f1 .* f2s
        good = findall(isfinite.(vec_cross) )
        xc[i] = nansum(vec_cross[good])
    end
    return xc
end

end