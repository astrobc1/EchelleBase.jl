module SpectralRegions

export SpecRegion1d, SpecRegion2d, label, num_orders

using Polynomials

"""
A SpecRegion1d.

# Fields
- pixmin: The starting pixel.
- pixmax: The ending pixel.
- λmin: The starting wavelength.
- λmax: The ending wavelength.
- order: The echelle order number, if applicable.
- label: The label for this chunk.
- fiber: The fiber number, if applicable.
"""
struct SpecRegion1d{P<:Union{Nothing, Int}, O<:Union{Nothing, Int}, L<:Union{Nothing, String}, F<:Union{Nothing, Int}}
    pixmin::P
    pixmax::P
    λmin::Float64
    λmax::Float64
    order::O
    label::L
    fiber::F
end

SpecRegion1d(;pixmin=nothing, pixmax=nothing, λmin, λmax, order=nothing, label=nothing, fiber=nothing) = SpecRegion1d(pixmin, pixmax, λmin, λmax, order, label, fiber)

function label(s::SpecRegion1d)
    if !isnothing(s.label)
        return s.label
    else
        return "Order$(s.order)"
    end
end

Base.show(io::IO, s::SpecRegion1d) = println(io, "$(label(s)): Pixels = $(s.pixmin) - $(s.pixmax), λ = $(round(s.λmin, digits=4)) - $(round(s.λmax, digits=4)) nm")

"""
A SpecRegion2d.

# Fields
- pixmin: The starting pixel in the spectral direction.
- pixmax: The ending pixel in the spectral direction.
- orderbottom: The bottom order.
- ordertop: The top order.
- poly_bottom: The bottom bounding polynomial.
- poly_top: The top bounding polynomial.
"""
Base.@kwdef struct SpecRegion2d
    pixmin::Int
    pixmax::Int
    orderbottom::Int
    ordertop::Int
    poly_bottom::Polynomial
    poly_top::Polynomial
end

ordermin(s::SpecRegion2d) = min(s.orderbottom, s.ordertop)
ordermax(s::SpecRegion2d) = max(s.orderbottom, s.ordertop)
num_orders(s::SpecRegion2d) = ordermax(s) - ordermin(s) + 1

Base.show(io::IO, s::SpecRegion2d) = println(io, "Echellogram Region: Pixels = $(s.pixmin) - $(s.pixmax), m = $(ordermin(s)) - $(ordermax(s))")

end