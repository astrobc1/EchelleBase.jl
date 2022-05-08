module SpectralRegions

using Polynomials

export SpecRegion1d, SpecRegion2d, label, mask_image!, ordermin, ordermax, num_orders

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
num_orders(s::SpecRegion2d) = ordermax(s) - ordermin(s) + 1

Base.show(io::IO, s::SpecRegion2d) = println(io, "Echellogram Region: Pixels = $(s.pixmin) - $(s.pixmax), m = $(ordermin(s)) - $(ordermax(s))")

end