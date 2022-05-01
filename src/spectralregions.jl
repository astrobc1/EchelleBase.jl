module SpectralRegions

using Polynomials

export SpecRegion1d, SpecRegion2d, label, mask_image!, ordermin, ordermax

struct SpecRegion1d
    pixmin::Union{Nothing, Int}
    pixmax::Union{Nothing, Int}
    λmin::Float64
    λmax::Float64
    order::Union{Nothing, Int}
    label::Union{Nothing, String}
    fiber::Union{Nothing, Float64}
end

SpecRegion1d(;pixmin=nothing, pixmax=nothing, λmin, λmax, order=nothing, label=nothing, fiber=nothing) = SpecRegion1d(pixmin, pixmax, λmin, λmax, order, label, fiber)

function label(s::SpecRegion1d)
    if !isnothing(s.label)
        return s.label
    else
        return "Order$(s.order)"
    end
end

Base.show(io::IO, s::SpecRegion1d) = println(io, "$(label(s)): Pixels = $(s.pixmin) - $(s.pixmax), λ = $(round(s.λmin, digits=3)) - $(round(s.λmax, digits=3)) nm")

Base.@kwdef struct SpecRegion2d
    pixmin::Union{Nothing, Int}
    pixmax::Union{Nothing, Int}
    orderbottom::Union{Nothing, Int}
    ordertop::Union{Nothing, Int}
    poly_bottom::Polynomial
    poly_top::Polynomial
end

ordermin(s::SpecRegion2d) = min(s.orderbottom, s.ordertop)
ordermax(s::SpecRegion2d) = max(s.orderbottom, s.ordertop)

Base.show(io::IO, s::SpecRegion2d) = println(io, "Echellogram Region: Pixels = $(s.pixmin) - $(s.pixmax), m = $(ordermin(s)) - $(ordermax(s))")

function mask_image!(image, sregion::SpecRegion2d)
    ny, nx = size(image)
    if sregion.pixmin > 1
        image[:, 1:sregion.pixmin-1] .= NaN
    end
    if sregion.pixmax < nx
        image[:, sregion.pixmax+1:end] .= NaN
    end
    xarr = [1:nx;]
    ybottom = sregion.poly_bottom.(xarr)
    ytop = sregion.poly_top.(xarr)
    yarr = [1:ny;]
    for i=1:nx
        bad = findall((yarr .< ybottom[i]) .|| (yarr .> ytop[i]))
        image[bad, i] .= NaN
    end
end

end