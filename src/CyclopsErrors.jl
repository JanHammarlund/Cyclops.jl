#############################
# Abstract Errors ###########
#############################
"""
```text
CyclopsError
│
├── CyclopsConstructorError
│     │
│     ├── CyclopsConstructorDomainError
│     │      ├── CyclopsConstructorHypersphereDomainError
│     │      ├── CyclopsConstructorInputAndHypersphereDomainError
│     │      └── CyclopsConstructorMultihotDomainError
│     │
│     └── CyclopsConstructorShapeError
│            ├── CyclopsMultiHotParameterShapeError
│            │      ├── CyclopsMultiHotMatrixShapeError
│            │      └── CyclopsMultiHotOffsetShapeError
│            │
│            └── CyclopsDenseShapeError
│                   ├── CyclopsDenseDimensionError
│                   │      └── CyclopsDenseCompressionDimensionError
│                   │
│                   └── CyclopsInverseDimensionMismatch
│
└── CyclopsFunctionError
        │
        ├── CyclopsInputDimensionMismatch
        │      ├── CyclopsDimensionMismatch
        │      └── CyclopsMultiHotDimensionMismatch
        │
        ├── CyclopsBottleneckError
        │      ├── CyclopsHypersphereDomainError
        │      └── CyclopsHypersphereDivideError
        │
        └── CyclopsMethodError
```
"""
abstract type CyclopsError <: Exception end                                             #
# Cyclops Error is the parent to Constructor Errors and Function Errors                 #

# Constructor Errors                                                                    #
abstract type CyclopsConstructorError <: CyclopsError end                               #
# Constructor errors fall into either domain or shape errors                            #

# Domain errors concern the first constructor method                                    #
# using 1 to 3 integers: n[, m=0, c=2]                                                  #
abstract type CyclopsConstructorDomainError <: CyclopsConstructorError end              #
#                ├── CyclopsConstructorHypersphereDomainError                           #
#                ├── CyclopsConstructorInputAndHypersphereDomainError                   #
#                └── CyclopsConstructorMultihotDomainError                              #

# Shape errors concern the second constructor method                                    #
# supplying custom arrays and Flux.Dense layers                                         #
abstract type CyclopsConstructorShapeError <: CyclopsConstructorError end               #
# constructor shape erros fall into either multi-hot parameter or dense shape errors    #
abstract type CyclopsMultiHotParameterShapeError <: CyclopsConstructorShapeError end    #
#                ├── CyclopsMultiHotMatrixShapeError                                    #
#                └── CyclopsMultiHotOffsetShapeError                                    #

abstract type CyclopsDenseShapeError <: CyclopsConstructorShapeError end                #
#                └── CyclopsInverseDimensionMismatch                                    #
# dense shape errors are either dimension mismatches between dense layers               #
abstract type CyclopsDenseDimensionError <: CyclopsDenseShapeError end                  #
#                └── CyclopsDenseCompressionDimensionError                              #
# or dimension mismatches between the dense layers and the rest of the model            #

# Function Errors                                                                       #
# function errors are runtime errors                                                    #
# a cyclops model is provided inappropriate data                                        #
abstract type CyclopsFunctionError <: CyclopsError end                                  #
#                └── CyclopsMethodError                                                 #
abstract type CyclopsInputDimensionMismatch <: CyclopsFunctionError end                 #
#                ├── CyclopsDimensionMismatch                                           #
#                └── CyclopsMultiHotDimensionMismatch                                   #

# or there's a failure calculating the hypersphere node compression                     #
abstract type CyclopsBottleneckError <: CyclopsFunctionError end                        #
#                ├── CyclopsHypersphereDomainError                                      #
#                └── CyclopsHypersphereDivideError                                      #

#########################################################################################

###################
# Concrete Errors #
##########################
##### Constructor Errors #
#####################################
######### Constructor Domain Errors #
########################################
############# Hypersphere Domain Error #
########################################
"""
    CyclopsConstructorHypersphereDomainError(c::Int)

An error when `c < 2`.

# Examples
```julia-repl
julia> n = 5; m = 0; c = 1; cyclops(n, m, c)
ERROR: CyclopsConstructorHypersphereDomainError: `c` = 1, but `c` must be ≥ 2.
[...]
```

# Supertype Hierarchy
    CyclopsConstructorHypersphereDomainError <: CyclopsConstructorDomainError <: CyclopsConstructorError <: CyclopsError <: Exception <: Any

# See also
[`CheckCyclopsConstructorInput`](@ref), [`CyclopsConstructorInputAndHypersphereDomainError`](@ref),
[`CyclopsConstructorMultihotDomainError`](@ref), [`cyclops`](@ref)
"""
struct CyclopsConstructorHypersphereDomainError <: CyclopsConstructorDomainError 
    c::Int
end

Base.showerror(io::IO, e::CyclopsConstructorHypersphereDomainError) = begin
    print(io, "CyclopsConstructorHypersphereDomainError: `c` = $(e.c), but `c` must be ≥ 2.")
end

##################################################
############# Input And Hypersphere Domain Error #
##################################################
"""
    CyclopsConstructorInputAndHypersphereDomainError(n::Int, c::Int)

An error when `n ≤ c`.

# Examples
```julia-repl
julia> n = 5; m = 0; c = 5; cyclops(n, m, c)
ERROR: CyclopsConstructorInputAndHypersphereDomainError: `n` = 5 ≤ `c`, but `n` must be > 5.
[...]
```

# Supertype Hierarchy
    CyclopsConstructorInputAndHypersphereDomainError <: CyclopsConstructorDomainError <: CyclopsConstructorError <: CyclopsError <: Exception <: Any

# See also
[`CheckCyclopsConstructorInput`](@ref), [`CyclopsConstructorHypersphereDomainError`](@ref),
[`CyclopsConstructorMultihotDomainError`](@ref), [`cyclops`](@ref)
"""
struct CyclopsConstructorInputAndHypersphereDomainError <: CyclopsConstructorDomainError 
    n::Int
    c::Int
end

Base.showerror(io::IO, e::CyclopsConstructorInputAndHypersphereDomainError) = begin
    print(io, "CyclopsConstructorInputAndHypersphereDomainError: `n` = $(e.n) ≤ `c`, but `n` must be > $(e.c).")
end

#####################################
############# Multihot Domain Error #
#####################################
"""
    CyclopsConstructorMultihotDomainError(m::Int)

An error when `m < 0`.

# Examples
```julia-repl
julia> n = 5; m = -1; c = 3; cyclops(n, m, c)
ERROR: CyclopsConstructorMultihotDomainError: `m` = -1 < 0, but `m` must be ≥ 0
[...]
```

# Supertype Hierarchy
    CyclopsConstructorMultihotDomainError <: CyclopsConstructorDomainError <: CyclopsConstructorError <: CyclopsError <: Exception <: Any

# See also
[`CheckCyclopsConstructorInput`](@ref), [`CyclopsConstructorHypersphereDomainError`](@ref),
[`CyclopsConstructorInputAndHypersphereDomainError`](@ref), [`cyclops`](@ref)
"""
struct CyclopsConstructorMultihotDomainError <: CyclopsConstructorDomainError 
    m::Int
end

Base.showerror(io::IO, e::CyclopsConstructorMultihotDomainError) = begin
    print(io, "CyclopsConstructorMultihotDomainError: `m` = $(e.m) < 0, but `m` must be ≥ 0.")
end
