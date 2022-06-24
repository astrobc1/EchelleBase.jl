module SpectralData

using EchelleBase

using FITSIO
using DataFrames

# Exports
export SpecData, SpecData1d, SpecData2d, Echellogram, RawSpecData2d, MasterCal2d
export get_spectrograph, get_spec_module
export read_header, read_image, read_spec1d
export ordermin, ordermax, orderbottom, ordertop
export parse_exposure_start_time, parse_itime, parse_object, parse_sky_coord, parse_utdate
export get_exposure_midpoint, get_barycentric_velocity, get_barycentric_corrections
export get_λsolution_estimate, normalize!

"""
An abstract type for all spectral data, both 2d echellograms and extracted 1d spectra, parametrized by the spectrograph S.
"""
abstract type SpecData{S} end

"""
An abstract type for all 2d spectral data (echellograms), parametrized by the spectrograph S.
"""
abstract type SpecData2d{S} <: SpecData{S} end

"""
    get_spectrograph(data::SpecData{S})
Returns the name of the spectrograph as a string corresponding to this SpecData object.
"""
get_spectrograph(data::SpecData{S}) where {S} = String(typeof(data).parameters[1])

"""
    get_spec_module(data::SpecData{S})
Returns the module for this spectrograph.
"""
function get_spec_module(::SpecData{S}) where {S} end

"""
A SpecData1d.

# Fields
- fname: The starting pixel.
- header: The ending pixel.
- data: DataFrame.
"""
struct SpecData1d{S} <: SpecData{S}
    fname::String
    header::FITSHeader
    data::DataFrame
end

function SpecData1d(fname::String, spectrograph, sregion::SpecRegion1d)
    data = SpecData1d{Symbol(lowercase(spectrograph))}(fname, FITSHeader(String[], [], String[]), DataFrame())
    header = read_header(data)
    data.header.keys = header.keys
    data.header.values = header.values
    data.header.comments = header.comments
    data.header.map = header.map
    read_spec1d(data, sregion)
    return data
end

struct RawSpecData2d{S} <: SpecData2d{S}
    fname::String
    header::Union{FITSHeader, Nothing}
end

struct MasterCal2d{S} <: SpecData2d{S}
    fname::String
    group::Vector{SpecData2d{S}}
end

function RawSpecData2d(fname::String, spectrograph::Union{String, Symbol})
    data = RawSpecData2d{Symbol(lowercase(spectrograph))}(fname, FITSHeader(String[], [], String[]))
    header = read_header(data)
    data.header.keys = header.keys
    data.header.values = header.values
    data.header.comments = header.comments
    data.header.map = header.map
    return data
end

function MasterCal2d(fname::String, group::Vector)
    spec2d = MasterCal2d{Symbol(get_spectrograph(group[1]))}(fname, group)
    return spec2d
end

function normalize!(data::SpecData1d; p=0.98)
    medval = maths.weighted_median(data.data.flux, p=p)
    data.data.flux ./= medval
    data.data.fluxerr ./= medval
    data.header["scale"] = medval
end

const Echellogram = SpecData2d

# Print
Base.show(io::IO, d::SpecData1d) = print(io, "SpecData1d: $(basename(d.fname))")
Base.show(io::IO, d::RawSpecData2d) = print(io, "RawSpecData2d: $(basename(d.fname))")
Base.show(io::IO, d::MasterCal2d) = print(io, "MasterCal2d: $(basename(d.fname))")

# Equality
Base.:(==)(d1::SpecData{T}, d2::SpecData{V}) where {T, V} = d1.fname == d2.fname;

# Reading in header and data products
function read_header end
function read_image end
function read_spec1d end

# Orders
function orderbottom end
function ordertop end
ordermin(d::SpecData) = min(orderbottom(d), ordertop(d))
ordermax(d::SpecData) = max(orderbottom(d), ordertop(d))
num_orders(d::SpecData) = ordermax(d) - ordermin(d)

# Parsing header and/or filename info
function parse_itime end
function parse_object end
function parse_utdate end
function parse_sky_coord end
function parse_exposure_start_time end
function parse_image_num end

# Barycenter calculations
function get_exposure_midpoint end
function get_barycentric_velocity end
function get_barycentric_corrections end

# Wavelength info
function get_λsolution_estimate end

# End module
end