using CYCLOPS
using Test
using Random
using Flux
using InteractiveUtils: subtypes

@testset "CYCLOPS" begin

@testset "Cyclops Error Hierarchy" begin
    cyclops_error_hierarchy = Dict(
        CyclopsError => [
            CyclopsConstructorError,
            CyclopsFunctionError
        ],
        CyclopsConstructorError => [
            CyclopsConstructorDomainError,
            CyclopsConstructorShapeError
        ],
        CyclopsConstructorDomainError => [
            CyclopsConstructorHypersphereDomainError,
            CyclopsConstructorInputAndHypersphereDomainError,
            CyclopsConstructorMultiHotDomainError
        ],
        CyclopsConstructorShapeError => [
            CyclopsMultiHotParameterShapeError,
            CyclopsDenseShapeError
        ],
        CyclopsMultiHotParameterShapeError => [
            CyclopsMultiHotMatrixShapeError,
            CyclopsMultiHotOffsetShapeError
        ],
        CyclopsDenseShapeError => [
            CyclopsDenseDimensionError,
            CyclopsInverseDimensionMismatch
        ],
        CyclopsDenseDimensionError => [
            CyclopsDenseCompressionDimensionError
        ],
        CyclopsFunctionError => [
            CyclopsMethodError,
            CyclopsInputDimensionMismatch,
            CyclopsBottleneckError
        ],
        CyclopsInputDimensionMismatch => [
            CyclopsDimensionMismatch,
            CyclopsMultiHotDimensionMismatch
        ],
        CyclopsBottleneckError => [
            CyclopsHypersphereDomainError,
            CyclopsHypersphereDivideError
        ]
    )

    # Each parent’s current subtypes must be drawn from the expected set.
    for (parent, children) in cyclops_error_hierarchy
        @test Set(subtypes(parent)) ⊆ Set(children)
    end # 10 abstract types with concrete types as children, 10 tests

    concrete_errors = [
        CyclopsConstructorHypersphereDomainError,
        CyclopsConstructorInputAndHypersphereDomainError,
        CyclopsConstructorMultiHotDomainError,
        CyclopsMultiHotMatrixShapeError,
        CyclopsMultiHotOffsetShapeError,
        CyclopsDenseCompressionDimensionError,
        CyclopsInverseDimensionMismatch,
        CyclopsMethodError,
        CyclopsDimensionMismatch,
        CyclopsMultiHotDimensionMismatch,
        CyclopsHypersphereDomainError,
        CyclopsHypersphereDivideError
    ]

    for T in concrete_errors
        @test !isabstracttype(T)
    end # 13 concrete types, 13 tests

end # 23 tests

@testset "Expected Errors" begin
    
    @testset "Constructor Errors" begin
        # Errors encountered while initializing a variable::cyclops
        
        @testset "Constructor Domain Errors" begin
            # Method one for creating a variable::cyclops
            # Using the method cyclops(n::Int[, m::Int=0, c::Int=2])
            # For this method to work
            # 1) c ≥ 2
            @testset "Hypersphere Domain Error" begin
                @test CyclopsConstructorHypersphereDomainError isa DataType
                @test_throws CyclopsConstructorHypersphereDomainError cyclops(5, 0, 1)
                @test_throws "`c` = 1, but `c` must be ≥ 2." cyclops(5, 0, 1)
            end # 3 tests
            # 2) n > c
            @testset "Input and Hypersphere Domain Error" begin
                @test CyclopsConstructorInputAndHypersphereDomainError isa DataType
                @test_throws CyclopsConstructorInputAndHypersphereDomainError cyclops(5, 0, 6)
                @test_throws "`n` = 5 ≤ `c`, but `n` must be > 6." cyclops(5, 0, 6)
            end # 3 tests
            # 3) m ≥ 0
            @testset "Multi-hot Domain Error" begin
                @test CyclopsConstructorMultiHotDomainError isa DataType
                @test_throws CyclopsConstructorMultiHotDomainError cyclops(5, -1, 3)
                @test_throws "`m` = -1 < 0, but `m` must be ≥ 0." cyclops(5, -1, 3)
            end # 3 tests
            
        end # 9 tests

        @testset "Constructor Shape Errors" begin
            # Method two for creating variable::cyclops
            # Using the method cyclops(scale::Array{Float32}, mhoffset::Array{Float32}, offset::AbstractVecOrMat{Float32}, densein::Dense, denseout::Dense)
            # For this method to work multihot parameters need to have specific dimensions
            # scale is n x m
            # mhoffset is n x m
            # offset is n x 1 (or n x 0 if scale and mhoffset are n x 0)
            @testset "Multi-hot Parameter Shape Error" begin
                # When scale and mhoffset do not have the same dimensions
                @testset "Multi-hot Matrix Shape Error" begin
                    @test CyclopsMultiHotMatrixShapeError isa DataType
                    @test_throws CyclopsMultiHotMatrixShapeError cyclops(rand(Float32, 5, 3), rand(Float32, 6, 4), rand(Float32, 5), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
                    @test_throws "scale has dimensions (5, 3) ≠ (6, 4) dimensions of mhoffset." cyclops(rand(Float32, 5, 3), rand(Float32, 6, 4), rand(Float32, 5), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
                end # 3 tests
                # When offset has the wrong number of rows
                # Or when offset has the wrong number of columns
                # If scale and mhoffset are n x 0, then offset must be n x 0 as well
                # If scale and mhoffset are n x m, where m ≥ 1, then offset must be n x 1.
                @testset "Multi-hot Offset Shape Error" begin
                    @test CyclopsMultiHotOffsetShapeError isa DataType
                    @test_throws CyclopsMultiHotOffsetShapeError cyclops(rand(Float32, 5, 3), rand(Float32, 5, 3), rand(Float32, 6), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
                    @test_throws "expected dimensions (5,), but got (6,)." cyclops(rand(Float32, 5, 3), rand(Float32, 5, 3), rand(Float32, 6), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
                    @test_throws CyclopsMultiHotOffsetShapeError cyclops(rand(Float32, 5, 3), rand(Float32, 5, 3), rand(Float32, 5, 1), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
                    @test_throws "expected dimensions (5,), but got (5, 1)." cyclops(rand(Float32, 5, 3), rand(Float32, 5, 3), rand(Float32, 5, 1), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
                    @test_throws CyclopsMultiHotOffsetShapeError cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), zeros(Float32, 5), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
                    @test_throws "expected dimensions (5, 0), but got (5,)." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), zeros(Float32, 5), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
                end # 7 tests

            end # 10 tests
            # Accordingly, the dense layers need to have appropriate dimensions as well
            # densein has dimensions n => c
            # denseout has dimensions c => n
            @testset "Dense Shape Error" begin

                @testset "Dense Dimension Error" begin

                    @testset "Inverse Dimension Error" begin
                        @test CyclopsInverseDimensionMismatch isa DataType
                        # densein and denseout must have inverse dimensions
                        @test_throws CyclopsInverseDimensionMismatch cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(5 => 2), Flux.Dense(3 => 5))
                        @test_throws "Expected 5 => 2 compression to be mirrored by 2 => 5 expansion, but got 3 => 5." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(5 => 2), Flux.Dense(3 => 5))
                        
                        @test_throws CyclopsInverseDimensionMismatch cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(4 => 1), Flux.Dense(2 => 7))
                        @test_throws "Expected 4 => 1 compression to be mirrored by 1 => 4 expansion, but got 2 => 7." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(4 => 1), Flux.Dense(2 => 7))
                    end # 3 tests
                    
                    @testset "Dense Compression Error" begin
                        @test CyclopsDenseCompressionDimensionError isa DataType
                        
                        @test_throws CyclopsDenseCompressionDimensionError cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(3 => 4), Flux.Dense(4 => 3))
                        @test_throws "n => c ≥ 2, where n > c, but got 5 ≠ 3 => 4." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(3 => 4), Flux.Dense(4 => 3))
                        
                        @test_throws CyclopsDenseCompressionDimensionError cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(5 => 1), Flux.Dense(1 => 5))
                        @test_throws "n => c ≥ 2, where n > c, but got 5 => 1 < 2." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(5 => 1), Flux.Dense(1 => 5))
                        
                        @test_throws CyclopsDenseCompressionDimensionError cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(6 => 2), Flux.Dense(2 => 6))
                        @test_throws "n => c ≥ 2, where n > c, but got 5 ≠ 6 => 2." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(6 => 2), Flux.Dense(2 => 6))
                        
                        @test_throws CyclopsDenseCompressionDimensionError cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(7 => 1), Flux.Dense(1 => 7))
                        @test_throws "n => c ≥ 2, where n > c, but got 5 ≠ 7 => 1 < 2." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(7 => 1), Flux.Dense(1 => 7))
                    end # 7 tests

                end # 14 tests

            end # 10 tests

        end # 20 tests

    end # 29 tests

    @testset "Function Errors" begin
        # There are 3 types of Cyclops Function Errors
        # 1) Method Error
        # when a cyclops model without multi-hot parameters is provided input and a multi-hot encoding vector
        @testset "Method Error" begin
            @test CyclopsMethodError isa DataType
            @test_throws CyclopsMethodError cyclops(5,0,2)(randn(Float32, 5), ones(Int32, 2))
            @test_throws "Multi-hot encoding provided to model without multi-hot parameters." cyclops(5,0,2)(randn(Float32, 5), ones(Int32, 2))
        end # 3 tests
        # 2) Dimension Mismatch
        # when the input data to the model does not have the correct number of rows
        @testset "Dimension Mismatch" begin
           @test CyclopsDimensionMismatch isa DataType
           @test_throws CyclopsDimensionMismatch cyclops(5, 0, 3)(rand(Float32, 6))
           @test_throws "Input = 6 ≠ 5 = Multi-hot" cyclops(5, 0, 3)(rand(Float32, 6))
        end

        @testset "Multi-hot Dimension Mismatch" begin
            @test CyclopsMultiHotDimensionMismatch isa DataType
            @test_throws CyclopsMultiHotDimensionMismatch cyclops(5, 2, 3)(rand(Float32, 5), zeros(Int32, 3))
            @test_throws "Multi-hot encoding = 3 ≠ 2 = Multi-hot Parameters" cyclops(5, 2, 3)(rand(Float32, 5), zeros(Int32, 3))
        end

    end

end

# nparams
@testset "nparams" begin
    @test methods(nparams)[1].sig == Tuple{typeof(nparams), cyclops}
    @test nparams(cyclops(5, 0, 2)) == 27 # n = 5; m = 0; c = 2; 2*n*c + n + c # For standard model
    @test nparams(cyclops(6, 3, 3)) == 87 # n = 6; m = 3; c = 3; (4*n*m + 2*n + m) # For multi-hot model
    @test nparams(cyclops(5, 2, 2)) == 52 # n = 5; m = 2; c = 2; (4*n*m + 2*n + m) # For multi-hot model
    @test_throws MethodError nparams(Flux.Dense(5 => 2)) # Doesn't work on Flux.Dense layer
end

# ⊙, ⊗, ⊕, ⊖, ⊘, ⩕
@testset "operators" begin # 73 tests

    @testset "oplus" begin # 14 tests
        oplus_found_methods = Set(m.sig for m in methods(⊕));
        oplus_expected_methods = Set([
            Tuple{typeof(⊕), Number, AbstractArray{<:Number}},
            Tuple{typeof(⊕), AbstractArray{<:Number}, Number},
            Tuple{typeof(⊕), AbstractArray{<:Number}, AbstractArray{<:Number}}
        ])
            
        @test oplus_expected_methods ⊆ oplus_found_methods
            
        x = [1, 2, 3]   # ::AbstractArray{<:Number}
        y1 = 1          # ::Number
        y2 = [3, 2, 1]  # ::AbstractArray{<:Number}
            
        # Tuple{typeof(⊕), AbstractArray{<:Number}, Number},
        @test x ⊕ y1 == [2, 3, 4]
        @test [x x] ⊕ y1 == [2 2; 3 3; 4 4]

        # Tuple{typeof(⊕), Number, AbstractArray{<:Number}}
        @test y1 ⊕ x == [2, 3, 4]
        @test y1 ⊕ [x x] == [2 2; 3 3; 4 4]

        # Tuple{typeof(⊕), AbstractArray{<:Number}, AbstractArray{<:Number}}
        @test x ⊕ y2 == [4, 4, 4]
        @test [x x] ⊕ y2 == y2 ⊕ [x x] == [x x] ⊕ [y2 y2] == [4 4; 4 4; 4 4]
        
        # Tuple{typeof(⊕), Number, Number}
        @test_throws MethodError 1 ⊕ 1

        # Dimension Mismatch
        @test_throws DimensionMismatch ones(3) ⊕ ones(4)
        @test_throws "x has 3 and y has 4." ones(3) ⊕ ones(4)

        @test_throws DimensionMismatch [x x] ⊕ [y2 y2 y2]
        @test_throws "x and y don't have matching dimensions" [x x] ⊕ [y2 y2 y2]

        @test_throws DimensionMismatch [x x] ⊕ ones(4)
        @test_throws "x has 3 and y has 4." [x x] ⊕ ones(4)
    end     # oplus, 14 tests
    
    @testset "ominus" begin # 15 tests
        ominus_found_methods = Set(m.sig for m in methods(⊖));
        ominus_expected_methods = Set([
            Tuple{typeof(⊖), Number, AbstractArray{<:Number}},
            Tuple{typeof(⊖), AbstractArray{<:Number}, Number},
            Tuple{typeof(⊖), AbstractArray{<:Number}, AbstractArray{<:Number}}
        ])
            
        @test ominus_expected_methods ⊆ ominus_found_methods

        x = [1, 2, 3]   # ::AbstractArray{<:Number}
        y1 = 1          # ::Number
        y2 = [3, 2, 1]  # ::AbstractArray{<:Number}

        # Tuple{typeof(⊖), AbstractArray{<:Number}, Number}
        @test x ⊖ y1 == [0, 1, 2]
        @test [x x] ⊖ y1 == [0 0; 1 1; 2 2]

        # Tuple{typeof(⊖), Number, AbstractArray{<:Number}}
        @test y1 ⊖ x == [0, -1, -2]
        @test y1 ⊖ [x x] == [0 0; -1 -1; -2 -2]

        # Tuple{typeof(⊖), AbstractArray{<:Number}, AbstractArray{<:Number}}
        @test x ⊖ y2 == [-2, 0, 2]
        @test [x x] ⊖ y2 == [x x] ⊖ [y2 y2] == [-2 -2; 0 0; 2 2]
        @test y2 ⊖ [x x] == [y2 y2] ⊖ [x x] == [2 2; 0 0; -2 -2]

        # Tuple{typeof(⊖), Number, Number}
        @test_throws MethodError 1 ⊖ 1

        # Dimension Mismatch
        @test_throws DimensionMismatch ones(3) ⊖ ones(4)
        @test_throws "x has 3 and y has 4." ones(3) ⊖ ones(4)

        @test_throws DimensionMismatch [x x] ⊖ [y2 y2 y2]
        @test_throws "don't have matching dimensions" [x x] ⊖ [y2 y2 y2]

        @test_throws DimensionMismatch [x x] ⊖ ones(4)
        @test_throws "x has 3 and y has 4." [x x] ⊖ ones(4)
    end     # ominus, 15 tests
    
    @testset "otimes" begin # 9 tests
        otimes_found_methods = Set(m.sig for m in methods(⊗));
        otimes_expected_methods = Set([
            Tuple{typeof(⊗), AbstractArray{<:Number}, Union{Number, AbstractArray{<:Number}}}
        ])

        @test otimes_expected_methods ⊆ otimes_found_methods

        x = ones(3)
        y1 = [1, 0]
        y2 = [0, 1]
        y3 = [1, 0, 1]

        # Tuple{typeof(⊗), AbstractArray{<:Number}, Number}
        @test x ⊗ 2 == [2, 2, 2]

        # Tuple{typeof(⊗), AbstractArray{<:Number}, AbstractArray{<:Number}}
        @test [x 2*x] ⊗ y1 == [1, 1, 1]
        @test [x 2*x] ⊗ y2 == [2, 2, 2]

        # Tuple{typeof(⊗), Number, Number}
        @test_throws MethodError 1 ⊗ 1
        
        # Dimension Mismatch
        @test_throws DimensionMismatch x ⊗ y3
        @test_throws "x has 1 columns and y has 3 rows." x ⊗ y3

        @test_throws DimensionMismatch [x 2*x] ⊗ 1
        @test_throws "x has 2 columns and y has 1 rows." [x 2*x] ⊗ 1
    end     # otimes, 9 tests
    
    @testset "odot" begin # 14 tests
        odot_found_methods = Set(m.sig for m in methods(⊙));
        odot_expected_methods = Set([
            Tuple{typeof(⊙), AbstractArray{<:Number}, Number},
            Tuple{typeof(⊙), Number, AbstractArray{<:Number}},
            Tuple{typeof(⊙), AbstractArray{<:Number}, AbstractArray{<:Number}}
        ])

        @test odot_expected_methods ⊆ odot_found_methods

        x = [1, 2, 3]   # ::AbstractArray{<:Number}
        y1 = 3          # ::Number
        y2 = [2, 3, 4]  # ::AbstractArray{<:Number}
        
        # Tuple{typeof(⊙), AbstractArray{<:Number}, Number}
        @test x ⊙ y1 == [3, 6, 9]
        @test [x x] ⊙ y1 == [3 3; 6 6; 9 9]

        # Tuple{typeof(⊙), Number, AbstractArray{<:Number}}
        @test y1 ⊙ x == [3, 6, 9]
        @test y1 ⊙ [x x] == [3 3; 6 6; 9 9]

        # Tuple{typeof(⊙), AbstractArray{<:Number}, AbstractArray{<:Number}}
        @test x ⊙ y2 == [2, 6, 12]
        @test [x x] ⊙ y2 == y2 ⊙ [x x] == [x x] ⊙ [y2 y2] == [2 2; 6 6; 12 12]
        
        # Tuple{typeof(⊙), Number, Number}
        @test_throws MethodError 1 ⊙ 1

        # Dimension Mismatch
        @test_throws DimensionMismatch ones(3) ⊙ ones(4)
        @test_throws "x has 3 and y has 4." ones(3) ⊙ ones(4)

        @test_throws DimensionMismatch ones(3, 2) ⊙ ones(4, 5)
        @test_throws "x and y don't have matching dimensions" ones(3, 2) ⊙ ones(4, 5)

        @test_throws DimensionMismatch ones(3, 2) ⊙ ones(4)
        @test_throws "x has 3 and y has 4." ones(3, 2) ⊙ ones(4)
    end     # odot, 14 tests
    
    @testset "oslash" begin # 15 tests
        oslash_found_methods = Set(m.sig for m in methods(⊘));
        oslash_expected_methods = Set([
            Tuple{typeof(⊘), AbstractArray{<:Number}, Number},
            Tuple{typeof(⊘), Number, AbstractArray{<:Number}},
            Tuple{typeof(⊘), AbstractArray{<:Number}, AbstractArray{<:Number}}
        ])

        @test oslash_expected_methods ⊆ oslash_found_methods

        x = [1, 2, 3]   # ::AbstractArray{<:Number}
        y1 = 2          # ::Number
        y2 = [3, 2, 1]  # ::AbstractArray{<:Number}

        # Tuple{typeof(⊘), AbstractArray{<:Number}, Number}
        @test x ⊘ y1 == [0.5, 1, 1.5]
        @test [x x] ⊘ y1 == [0.5 0.5; 1 1; 1.5 1.5]

        # Tuple{typeof(⊘), Number, AbstractArray{<:Number}}
        @test y1 ⊘ x == [2, 1, 2/3]
        @test y1 ⊘ [x x] == [2 2; 1 1; 2/3 2/3]

        # Tuple{typeof(⊘), AbstractArray{<:Number}, AbstractArray{<:Number}}
        @test x ⊘ y2 == [1/3, 1, 3]
        @test [x x] ⊘ y2 == [x x] ⊘ [y2 y2] == [1/3 1/3; 1 1; 3 3]
        @test y2 ⊘ [x x] == [y2 y2] ⊘ [x x] == [3 3; 1 1; 1/3 1/3]

        # Tuple{typeof(⊘), Number, Number}
        @test_throws MethodError 1 ⊘ 1

        # Dimension Mismatch
        @test_throws DimensionMismatch ones(3) ⊘ ones(4)
        @test_throws "x has 3 and y has 4." ones(3) ⊘ ones(4)

        @test_throws DimensionMismatch ones(3, 2) ⊘ ones(5, 7)
        @test_throws "x and y don't have matching dimensions" ones(3, 2) ⊘ ones(5, 7)

        @test_throws DimensionMismatch ones(3, 2) ⊘ ones(4)
        @test_throws "x has 3 and y has 4." ones(3, 2) ⊘ ones(4)
    end     # oslash, 15 tests

    @testset "wedge on wedge" begin # 6 tests
        wedgeonwedge_found_methods = Set(m.sig for m in methods(⩕));
        wedgeonwedge_expected_methods = Set([
            Tuple{typeof(⩕), AbstractArray{<:Number}, Number}
        ])

        @test wedgeonwedge_expected_methods ⊆ wedgeonwedge_found_methods

        x = [1 2; 3 4]
        y1 = 2
        y2 = [2, 3]
        y3 = [2 3; 4 5]

        @test x ⩕ y1 == [1 4; 9 16]

        # Tuple{typeof(⩕), AbstractArray{<:Number}, AbstractArray{<:Number}}
        @test_throws MethodError x ⩕ y2
        @test_throws MethodError x ⩕ y3

        # Tuple{typeof(⩕), Number, AbstractArray{<:Number}}
        @test_throws MethodError y2 ⩕ x

        # Tuple{typeof(⩕), Number, Number}
        @test_throws MethodError y1 ⩕ y1
    end     # wedge on wedge, 6 tests
    
end              # operators 73 tests

end
