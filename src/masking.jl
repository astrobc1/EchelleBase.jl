export mask!

"""
    mask!(data::SpecData1d, sregion::SpecRegion1d)
Mask bad pixels in-place according to the variables data.data.λ, data.data.flux, data.data.fluxerr, data.data.mask, as well as the bounding pixels in sregion (not wavelength).
"""
function mask!(data::SpecData1d, sregion::SpecRegion1d)
    if !isnothing(sregion.pixmin) && sregion.pixmin > 1
        if hasproperty(data.data, :λ)
            data.data.λ[1:sregion.pixmin-1] .= NaN
        end
        data.data.flux[1:sregion.pixmin-1] .= NaN
        data.data.fluxerr[1:sregion.pixmin-1] .= NaN
        data.data.mask[1:sregion.pixmin-1] .= 0
    end
    if !isnothing(sregion.pixmax) && sregion.pixmax < length(data.data.flux)
        if hasproperty(data.data, :λ)
            data.data.λ[sregion.pixmax+1:end] .= NaN
        end
        data.data.flux[sregion.pixmax+1:end] .= NaN
        data.data.fluxerr[sregion.pixmax+1:end] .= NaN
        data.data.mask[sregion.pixmax+1:end] .= 0
    end
    if hasproperty(data.data, :λ)
        bad = findall(.~isfinite.(data.data.λ) .|| .~isfinite.(data.data.flux) .|| .~isfinite.(data.data.fluxerr) .|| (data.data.mask .== 0) .|| (data.data.flux .<= 0))
        data.data.λ[bad] .= NaN
        data.data.flux[bad] .= NaN
        data.data.fluxerr[bad] .= NaN
        data.data.mask[bad] .= 0
    else
        bad = findall(.~isfinite.(data.data.flux) .|| .~isfinite.(data.data.fluxerr) .|| (data.data.mask .== 0) .|| (data.data.flux .<= 0))
        data.data.flux[bad] .= NaN
        data.data.fluxerr[bad] .= NaN
        data.data.mask[bad] .= 0
    end
end

"""
    mask!(image::AbstractMatrix{<:Number}, sregion::SpecRegion2d)
Mask bad pixels in-place according to the bounding polynomials and left/right ends.
"""
function mask!(image::AbstractMatrix{<:Number}, sregion::SpecRegion2d)
    ny, nx = size(image)
    if sregion.pixmin > 1
        image[:, 1:sregion.pixmin-1] .= NaN
    end
    if sregion.pixmax < nx
        image[:, sregion.pixmax+1:end] .= NaN
    end
    yarr = 1:ny
    for x=1:nx
        ybottom = sregion.poly_bottom(x)
        ytop = sregion.poly_top(x)
        bad = findall((yarr .< ybottom) .|| (yarr .> ytop))
        image[bad, x] .= NaN
    end
end