module SpectralData

using FITSIO
using DataFrames

export SpecData, SpecData1d, SpecData2d, RawSpecData2d, Echellogram, get_spectrograph, get_spec_module, MasterCal2d, read_header, read_image, read_spec1d, parse_exposure_start_time, parse_itime, parse_object, parse_sky_coord, parse_utdate, get_exposure_midpoint, get_barycentric_velocity, get_barycentric_corrections, get_λsolution_estimate

abstract type SpecData{S} end
abstract type SpecData2d{S} <: SpecData{S} end

get_spectrograph(data::SpecData) = String(typeof(data).parameters[1])
get_spec_module(::SpecData{T}) where {T} = nothing

mutable struct SpecData1d{S} <: SpecData{S}
    fname::String
    header::Union{FITSHeader, Nothing}
    data::Union{DataFrame, Nothing}
end

function SpecData1d(fname::String, spectrograph, sregion)
    spec1d = SpecData1d{Symbol(lowercase(spectrograph))}(fname, nothing, nothing)
    spec1d.header = read_header(spec1d)
    spec1d.data = read_spec1d(spec1d, sregion)
    return spec1d
end

mutable struct RawSpecData2d{S} <: SpecData2d{S}
    fname::String
    header::Union{FITSHeader, Nothing}
end

mutable struct MasterCal2d{S} <: SpecData2d{S}
    fname::String
    group::Vector{SpecData2d{S}}
end

function RawSpecData2d(fname::String, spectrograph::Union{String, Symbol})
    spec2d = RawSpecData2d{Symbol(lowercase(spectrograph))}(fname, nothing)
    spec2d.header = read_header(spec2d)
    return spec2d
end

function MasterCal2d(fname::String, group::Vector)
    spec2d = MasterCal2d{Symbol(get_spectrograph(group[1]))}(fname, group)
    return spec2d
end

const Echellogram = SpecData2d

# Print
Base.show(io::IO, d::SpecData1d) = print(io, "SpecData1d: $(basename(d.fname))")
Base.show(io::IO, d::RawSpecData2d) = print(io, "RawSpecData2d: $(basename(d.fname))")
Base.show(io::IO, d::MasterCal2d) = print(io, "MasterCal2d: $(basename(d.fname))")

# Reading in header
function read_header end

# Reading in image
function read_image end

# Reading in 1d spectrum
function read_spec1d end

# Parsing
function parse_itime end
function parse_object end
function parse_utdate end
function parse_sky_coord end
function parse_exposure_start_time end
function parse_image_num end

# Barycenter
function get_exposure_midpoint end
function get_barycentric_velocity end
function get_barycentric_corrections end

# Wavelength info
function get_λsolution_estimate end


end