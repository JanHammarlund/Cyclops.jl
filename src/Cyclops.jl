module Cyclops

using CUDA, Flux, Statistics, ProgressMeter, Random, CairoMakie

include("CyclopsErrors.jl")

export CyclopsError, CyclopsConstructorError, CyclopsFunctionError

include("CyclopsOperators.jl")
include("CyclopsConstructors.jl")

export nparams

include("CyclopsOverload.jl")

include("CyclopsLayers.jl")

export mhe, hsn, mhd

Flux.@layer cyclops
export cyclops

include("CyclopsDecode.jl")
export project, decode

include("CyclopsOptimization.jl")
export train!

end
