module SpectralRegions

export SpecRegion1d, SpecRegion2d, label

using Polynomials

"""
A container for 1d spectral regions for reduced spectra.

# Fields
- pixmin: The starting pixel (if relevant).
- pixmax: The ending pixel (if relevant).
- λmin: The starting wavelength.
- λmax: The ending wavelength.
- order: The echelle order number (if relevant).
- label: The label for this chunk.
- fiber: The fiber number (if relevant).
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

"""
    SpecRegion1d(;pixmin=nothing, pixmax=nothing, λmin, λmax, order=nothing, label=nothing, fiber=nothing) = SpecRegion1d(pixmin, pixmax, λmin, λmax, order, label, fiber)
Construct a SpecRegion1d object.
"""
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
A container for 2d spectral regions for echellograms.

# Fields
- pixmin: The starting pixel in the spectral direction.
- pixmax: The ending pixel in the spectral direction.
- orderbottom: The bottom order.
- ordertop: The top order.
- poly_bottom: The bottom bounding polynomial.
- poly_top: The top bounding polynomial.
"""
struct SpecRegion2d
    pixmin::Int
    pixmax::Int
    orderbottom::Int
    ordertop::Int
    poly_bottom::Polynomial
    poly_top::Polynomial
end

ordermin(s::SpecRegion2d) = min(s.orderbottom, s.ordertop)
ordermax(s::SpecRegion2d) = max(s.orderbottom, s.ordertop)

"""
    SpecRegion2d(;pixmin, pixmax, orderbottom, ordertop, poly_bottom, poly_top)
Construct a SpecRegion2d object.
"""
function SpecRegion2d(;pixmin, pixmax, orderbottom, ordertop, poly_bottom, poly_top)
    return SpecRegion2d(pixmin, pixmax, orderbottom, ordertop, poly_bottom, poly_top)
end

Base.show(io::IO, s::SpecRegion2d) = println(io, "Echellogram Region: Pixels = $(s.pixmin) - $(s.pixmax), m = $(ordermin(s)) - $(ordermax(s))")

end