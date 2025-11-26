module Cyclops

using CUDA, Flux, Statistics, ProgressMeter, Plots, Random

include("CyclopsErrors.jl")

export CyclopsError, CyclopsConstructorError, CyclopsFunctionError
export CyclopsHypersphereDomainError, CyclopsInputAndHypersphereDomainError, CyclopsMultihotDomainError
export CyclopsMultihotMatrixShapeError, CyclopsMultihotOffsetShapeError
export CyclopsDenseInverseShapeError, CyclopsDenseShapeError
export CyclopsMultihotDimensionMismatch, CyclopsInputDimensionMismatch
export CyclopsHypersphereNaNError, CyclopsHypersphereDivideError

include("CyclopsOperators.jl")

export ⊙, ⊗, ⊕, ⊖, ⊘, ⩕

include("CyclopsConstructors.jl")

export CheckCyclopsConstructorInput
export nparams

include("CyclopsOverload.jl")

include("CyclopsLayers.jl")

export mhe, hsn, mhd

Flux.@layer cyclops
export cyclops

include("CyclopsOptimization.jl")

end
