export mask!

function mask!(data::SpecData1d, sregion::SpecRegion1d)
    if sregion.pixmin > 1
        if hasproperty(data.data, :λ)
            data.data.λ[1:sregion.pixmin-1] .= NaN
        end
        data.data.flux[1:sregion.pixmin-1] .= NaN
        data.data.fluxerr[1:sregion.pixmin-1] .= NaN
        data.data.mask[1:sregion.pixmin-1] .= 0
    end
    if sregion.pixmax < length(data.data.flux)
        if hasproperty(data.data, :λ)
            data.data.λ[sregion.pixmax+1:end] .= NaN
        end
        data.data.flux[sregion.pixmax+1:end] .= NaN
        data.data.fluxerr[sregion.pixmax+1:end] .= NaN
        data.data.mask[sregion.pixmax+1:end] .= 0
    end
    bad = findall(.~isfinite.(data.data.flux) .|| .~isfinite.(data.data.fluxerr) .|| (data.data.mask .== 0))
    if hasproperty(data.data, :λ)
        data.data.λ[bad] .= NaN
    end
    data.data.flux[bad] .= NaN
    data.data.fluxerr[bad] .= NaN
    data.data.mask[bad] .= 0
end

function mask!(image::AbstractMatrix, sregion::SpecRegion2d)
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