module Maths

using DataInterpolations
using NaNStatistics
using LoopVectorization
using StatsBase
using Polynomials
using PyCall

const SPEED_OF_LIGHT_MPS = 299792458.0
const TWO_SQRT_2LOG2 = 2 * sqrt(2 * log(2))


function rmsloss(x, y, weights=nothing; flag_worst=0, remove_edges=0)

    # Compute diffs2
    if !isnothing(weights)
        good = findall(isfinite.(x) .&& isfinite.(y) .&& isfinite.(weights) .&& (weights .> 0))
        xx, yy, ww = x[good], y[good], weights[good]
        diffs2 = ww .* (xx .- yy).^2
    else
        good = findall(isfinite.(x) .&& isfinite.(y))
        xx, yy = x[good], y[good]
        diffs2 = (xx .- yy).^2
    end
    
    # Ignore worst N pixels
    if flag_worst > 0
        ss = sortperm(diffs2)
        diffs2[ss[end-flag_worst+1:end]] .= NaN
        if !isnothing(weights)
            ww[ss[end-flag_worst+1:end]] .= 0
        end
    end
                
    # Remove edges
    if remove_edges > 0
        diffs2[1:remove_edges] .= 0
        diffs2[end-remove_edges+1:end] .= 0
        if !isnothing(weights)
            ww[1:remove_edges] .= 0
            ww[end-remove_edges+1:end] .= 0
        end
    end
        
    # Compute rms
    if !isnothing(weights)
        rms = sqrt(nansum(diffs2) / nansum(ww))
    else
        rms = sqrt(nansum(diffs2) / length(diffs2))
    end

    # Return
    return rms
end


function doppler_shift_λ(λ, vel, mode="sr")
    if lowercase(mode) == "sr"
        β = vel ./ SPEED_OF_LIGHT_MPS
        return λ .* sqrt((1 .+ β) ./ (1 .- β))
    else
        return λ .* (1 .+ vel ./ SPEED_OF_LIGHT_MPS)
    end
end


function doppler_shift_flux(λ, flux, vel)

    # The shifted wave
    λ_shifted = doppler_shift_λ(λ, vel)

    # Interpolate the flux
    flux_out = cspline_interp(λ_shifted, flux, λ)

    # Return
    return flux_out
end

function cspline_interp(x, y, xnew)
    good = findall(isfinite.(y))
    xx = @view x[good]
    yy = @view y[good]
    y_out = DataInterpolations.CubicSpline(yy, xx).(xnew)
    bad = findall((xnew .< x[good[1]]) .|| (xnew .> x[good[end]]))
    y_out[bad] .= NaN
    return y_out
end

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


function median_filter1d(x, width)
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

function median_filter2d(x, width)
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

function chebyval2d(x::Real, y::Real, coeffs)
    numpy = pyimport("numpy")
    return numpy.polynomial.chebyshev.chebval2d(y, x, coeffs)
    # n, m = size(coeffs)
    # s = 0.0
    # ccpolysx = ccpolys(y, n)
    # ccpolysy = ccpolys(x, m)
    # for i=1:n
    #    for j=1:m
    #        s += coeffs[i, j] * ccpolysx[i] * ccpolysy[j]
    #    end
    # end
    # return s
end

mad(x) = nanmedian(abs.(x .- nanmedian(x)))

function robust_σ(x, nσ=6)
    med = nanmedian(x)
    _mad = mad(x)
    good = findall(abs.(x .- med) .< 1.4826 * _mad * nσ)
    if length(good) > 1
        return nanstd(@view x[good])
    else
        return NaN
    end
end

function ccpolys(x::Real, n::Int)
    T0 = 1
    T1 = x
    if n == 0
        return T0
    elseif n == 1
        return T1
    else
        Ts = zeros(n+1)
        Ts[1] = T0
        Ts[2] = T1
        for i=3:n+1
            Ts[i] = 2x * Ts[i-1] - Ts[i-2]
        end
        return Ts
    end
end

function convolve1d(x, k)
    nx = length(x)
    nk = length(k)
    n_pad = Int(floor(nk / 2))
    out = zeros(nx)
    kf = @view k[end:-1:1]
    valleft = x[1]
    valright = x[end]
    
    # Left values
    for i=1:n_pad
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
    for i=nx-n_pad+1:nx
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

flatten(x) = collect(Iterators.flatten(x))

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
    xx = @view x[good]
    ww = @view w[good]
    ww = ww ./ sum(w)
    μ = weighted_mean(xx, ww)
    dev = xx .- μ
    bias_estimator = 1.0 - sum(ww.^2) / sum(ww)^2
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

function cross_correlate_interp(x1, y1, x2, y2, lags; kind="rms")

    # Shifts y2 and compares it to y1
    nx1 = length(x1)
    nx2 = length(x2)
  
    nlags = length(lags)
    kind = lowercase(kind)
    
    ccf = fill(NaN, nlags)
    y2_shifted = fill(NaN, nx1)
    weights = ones(nx1)
    for i=1:nlags
        y2_shifted .= lin_interp(x2 .+ lags[i], y2, x1)
        good = findall(isfinite.(y1) .&& isfinite.(y2_shifted))
        if length(good) < 3
            continue
        end
        weights .= 1
        bad = findall(.~isfinite.(y1) .|| .~isfinite.(y2_shifted))
        weights[bad] .= 0
        if kind == "rms"
            ccf[i] = sqrt(nansum(weights .* (y1 .- y2_shifted).^2) / nansum(weights))
        else
            ccf[i] = nansum(y1 .* y2_shifted .* weights) / nansum(weights)
        end
    end

    return ccf

end


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
function δλ2δv(δλ, λ)
    return @. SPEED_OF_LIGHT_MPS * log(δλ / λ + 1)
end

# dl = l0 * (e^(dv/c) - 1)
function δv2δλ(δv, λ)
    return @. λ * (exp(δv / SPEED_OF_LIGHT_MPS) - 1)
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

end