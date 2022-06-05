module EchelleBase

using Reexport

include("utils.jl")

include("maths.jl")
const maths = Maths
export maths

include("spectralregions.jl")
@reexport using .SpectralRegions

include("spectraldata.jl")
@reexport using .SpectralData

include("masking.jl")

end