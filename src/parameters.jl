import DataStructures: OrderedDict

export Parameter, Parameters, num_varied, is_varied

mutable struct Parameter
    name::Union{String, Nothing}
    value::Float64
    lower_bound::Float64
    upper_bound::Float64
    latex_str::Union{String, Nothing}
end

struct Parameters
    dict::OrderedDict{String, Parameter}
end

"""Construct a new parameter"""
Parameter(;name=nothing, value::Real, lower_bound::Real=-Inf, upper_bound::Real=Inf, latex_str=nothing) = Parameter(name, value, lower_bound, upper_bound, latex_str)

"""Construct an empty Parameters struct"""
Parameters() = Parameters(OrderedDict{String, Parameter}())

Base.length(pars::Parameters) = length(pars.dict)
Base.merge!(pars::Parameters, pars2::Parameters) = merge!(pars.dict, pars2.dict)
Base.getindex(pars::Parameters, key::String) = getindex(pars.dict, key)
Base.firstindex(pars::Parameters) = firstindex(pars.dict)
Base.lastindex(pars::Parameters) = lastindex(pars.dict)
Base.iterate(pars::Parameters) = iterate(pars.dict)
Base.keys(pars::Parameters) = keys(pars.dict)
Base.values(pars::Parameters) = values(pars.dict)

function set_name!(par::Parameter, name::String)
    if isnothing(par.name)
        par.name = name
    end
    if isnothing(par.latex_str)
        par.latex_str = name
    end
end

function Base.setindex!(pars::Parameters, par::Parameter, key::String)
    set_name!(par, key)
    setindex!(pars.dict, par, key)
end

function Base.show(io::IO, par::Parameter)
    println(io, " $(par.name) | Value = $(par.value) | Bounds = [$(par.lower_bound), $(par.upper_bound)]")
end

function Base.show(io::IO, pars::Parameters)
    for par ∈ values(pars)
        show(io, par)
    end
end

function num_varied(pars::Parameters)
    n = 0
    for par ∈ values(pars)
        if is_varied(par)
            n += 1
        end
    end
    return n
end

function is_varied(par::Parameter)
    return par.lower_bound == par.upper_bound
end