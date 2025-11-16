using CYCLOPS
using Test
using Random
using Flux

# cyclops
@testset "cyclops" begin

    @testset "constructor" begin

        @testset "check cyclops input" begin

            @testset "constructor errors" begin

                @testset "hypershpere dimension error" begin # 7 tests
                    @test Exception >: CyclopsHypersphereDimensionError isa DataType
                    @test_throws CyclopsHypersphereDimensionError throw(CyclopsHypersphereDimensionError(1))
                    @test_throws "`c` = 1, but `c` must be ≥ 2." throw(CyclopsHypersphereDimensionError(1))
                    
                    # Alternate methods that will pass because the number can be converted to integers
                    @test CyclopsHypersphereDimensionError(1f0) isa CyclopsHypersphereDimensionError
                    @test CyclopsHypersphereDimensionError(1.0) isa CyclopsHypersphereDimensionError

                    # Expected erros
                    @test_throws InexactError CyclopsHypersphereDimensionError(1.1)
                    @test_throws MethodError CyclopsHypersphereDimensionError("1")
                end # 7 tests

                @testset "input and hypershpere dimension conflict error" begin # 9 tests
                    @test Exception >: CyclopsInputHypersphereDimensionError isa DataType
                    @test_throws CyclopsInputHypersphereDimensionError throw(CyclopsInputHypersphereDimensionError(2, 2))
                    @test_throws "`n` = 2 ≤ `c`, but `n` must be > 2." throw(CyclopsInputHypersphereDimensionError(2, 2))

                    # Alternate methods that will pass because the number can be converted to integers
                    @test CyclopsInputHypersphereDimensionError(2f0,2f0) isa CyclopsInputHypersphereDimensionError
                    @test CyclopsInputHypersphereDimensionError(2.0,2.0) isa CyclopsInputHypersphereDimensionError

                    # Expected errors
                    @test_throws InexactError CyclopsInputHypersphereDimensionError(1.1, 2)
                    @test_throws InexactError CyclopsInputHypersphereDimensionError(1, 2.1)
                    @test_throws MethodError CyclopsInputHypersphereDimensionError("1", 2)
                    @test_throws MethodError CyclopsInputHypersphereDimensionError(1, "2")
                end # 9 tests

                @testset "multi-hot dimension error" begin # 7 tests
                    @test Exception >: CyclopsMultiHotDimensionError isa DataType
                    @test_throws CyclopsMultiHotDimensionError throw(CyclopsMultiHotDimensionError(-1))
                    @test_throws "`m` = -1 < 0, but `m` must be ≥ 0." throw(CyclopsMultiHotDimensionError(-1))

                    # Alternate methods that will pass because the number can be converted to integers
                    @test CyclopsMultiHotDimensionError(-1f0) isa CyclopsMultiHotDimensionError
                    @test CyclopsMultiHotDimensionError(-1.0) isa CyclopsMultiHotDimensionError
                    
                    # Expected errors
                    @test_throws MethodError CyclopsMultiHotDimensionError("-1f0")
                    @test_throws InexactError CyclopsMultiHotDimensionError(1.1)
                end # 7 tests

            end

        end

    end

end
#   constructor
#       CheckCyclopsInput
#           CyclopsHypersphereDimensionError
#           CyclopsInputHypersphereDimensionError
#           CyclopsMultiHotDimensionError
#   function
#       mhe, mhd
#           CheckMultiHotTransformation
#               CyclopsInputMultiHotDimensionMismatch
#               CyclopsMultiHotParameterDimensionMismatch
#       hsn
#           CheckHSNdomain
#               CyclopsHyperSphereDomainError
#               CyclopsHyperSphereDivideError

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




@testset "CYCLOPS" begin

        
    @testset "cyclops constructor" begin # 41 tests
        
        @test cyclops isa DataType          # 1
        @test length(methods(cyclops)) == 5 # 2
        
        m_cyclops = methods(cyclops);
        @test any(m_cyclops -> m_cyclops.sig == Tuple{Type{cyclops}, Int64, Int64, Int64}, m_cyclops)
        @test any(m_cyclops -> m_cyclops.sig == Tuple{Type{cyclops}, Int64, Int64}, m_cyclops)
        @test any(m_cyclops -> m_cyclops.sig == Tuple{Type{cyclops}, Int64}, m_cyclops)

        @testset "invalid arguments" begin # 18 tests
            
            m_CheckCyclopsInput = methods(CheckCyclopsInput);
            @test length(methods(CheckCyclopsInput)) == 1   # 1
            @test any(m_CheckCyclopsInput -> m_CheckCyclopsInput.sig == Tuple{typeof(CheckCyclopsInput), Int64, Int64, Int64}, m_CheckCyclopsInput)

            # c < 2 error, 4 tests
            @testset "invalid c" begin
                @test_throws CyclopsHypersphereDimensionError CheckCyclopsInput(5,0,1)  # 1
                @test_throws "`c` = 1, but `c` must be ≥ 2." CheckCyclopsInput(5,0,1)   # 2
                @test_throws CyclopsHypersphereDimensionError cyclops(5,0,1)            # 3
                @test_throws "`c` = 1, but `c` must be ≥ 2." cyclops(5,0,1)             # 4
            end

            # n < c error, 4 tests
            @testset "invalid n" begin
                @test_throws CyclopsInputHypersphereDimensionError CheckCyclopsInput(2,0,2) # 1
                @test_throws "`n` = 2 ≤ `c`, but `n` must be > 2." CheckCyclopsInput(2,0,2) # 2
                @test_throws CyclopsInputHypersphereDimensionError cyclops(2,0,2)           # 3
                @test_throws "`n` = 2 ≤ `c`, but `n` must be > 2." cyclops(2,0,2)           # 4
            end

            # m < 0 error, 4 tests
            @testset "invalid m" begin
                @test_throws CyclopsMultiHotDimensionError CheckCyclopsInput(5,-1,2)        # 1
                @test_throws "`m` = -1 < 0, but `m` must be ≥ 0." CheckCyclopsInput(5,-1,2) # 2
                @test_throws CyclopsMultiHotDimensionError cyclops(5,-1,2)                  # 3
                @test_throws "`m` = -1 < 0, but `m` must be ≥ 0." cyclops(5,-1,2)           # 4
            end

            # non-Int error, 6 tests
            @testset "method errors" begin
                @test_throws MethodError CheckCyclopsInput(5f0,0,2) # 1
                @test_throws MethodError CheckCyclopsInput(5,0f0,2) # 2
                @test_throws MethodError CheckCyclopsInput(5,0,2f0) # 3
                @test_throws MethodError cyclops(5f0,0,2)           # 4
                @test_throws MethodError cyclops(5,0f0,2)           # 5
                @test_throws MethodError cyclops(5,0,2f0)           # 6
            end

        end

        @testset "valid arguments" begin # 22 tests

            @testset "multi-hot model" begin # 11 tests
                n = 5; m = 3; c = 2;
                Random.seed!(1234); test_cyclops = cyclops(n, m)
                nmodelparams1 = nparams(test_cyclops)
                @test test_cyclops isa cyclops                          # 1
                @test length(methods(test_cyclops)) == 3                # 2
                @test length(propertynames(test_cyclops)) == 5          # 3
                @test nmodelparams1 == ((n*m)+(n*m)+n+(c*n)+c+(n*c)+n)  # 4
                @test size(test_cyclops.scale) == (n, m)                # 5
                @test size(test_cyclops.mhoffset) == (n, m)             # 6
                @test size(test_cyclops.offset) == (n,)               # 7
                @test size(test_cyclops.densein.weight) == (c, n)       # 8
                @test size(test_cyclops.densein.bias) == (c,)           # 9
                @test size(test_cyclops.denseout.weight) == (n, c)      # 10
                @test size(test_cyclops.denseout.bias) == (n,)          # 11
            end

            @testset "standard model" begin # 11 tests
                n = 5; m = 0; c = 2;
                Random.seed!(1234); test_cyclops = cyclops(n)
                nmodelparams = nparams(test_cyclops)
                @test test_cyclops isa cyclops                          # 1
                @test length(methods(test_cyclops)) == 3                # 2
                @test length(propertynames(test_cyclops)) == 5          # 3
                @test nmodelparams == ((c*n)+c+(n*c)+n)                 # 4
                @test size(test_cyclops.scale) == (n, m)                # 5
                @test size(test_cyclops.mhoffset) == (n, m)             # 6
                @test size(test_cyclops.offset) == (n, 0)               # 7
                @test size(test_cyclops.densein.weight) == (c, n)       # 8
                @test size(test_cyclops.densein.bias) == (c,)           # 9
                @test size(test_cyclops.denseout.weight) == (n, c)      # 10
                @test size(test_cyclops.denseout.bias) == (n,)          # 11
            end

        end

    end
    
    @testset "cyclops function" begin
        
        @testset "multi-hot node" begin # 26 tests
            
            @test length(methods(CheckMultiHotTransformation)) == 1 # 1
            m_CheckMultiHotTransformation = methods(CheckMultiHotTransformation);
            @test any(m_CheckMultiHotTransformation -> m_CheckMultiHotTransformation.sig == Tuple{typeof(CheckMultiHotTransformation), Vector{Float32}, Vector{Int32}, Array{Float32}}, m_CheckMultiHotTransformation)

            @test length(methods(mhe)) == 1                         # 2
            m_mhe = methods(mhe);
            @test any(m_mhe -> m_mhe.sig == Tuple{typeof(mhe), Vector{Float32}, Vector{Int32}, cyclops}, m_mhe)
            
            @test length(methods(mhd)) == 1                         # 3
            m_mhd = methods(mhd);
            @test any(m_mhd -> m_mhd.sig == Tuple{typeof(mhd), Vector{Float32}, Vector{Int32}, cyclops}, m_mhd)
            
            @testset "invalid arguments" begin # 18 tests
                
                # Model parameters and model input don't have same number of rows, 9 tests
                @testset "n mismatch input" begin # 9 tests
                    Random.seed!(1234); test_cyclops = cyclops(5, 3, 2);

                    @test_throws CyclopsInputMultiHotDimensionMismatch CheckMultiHotTransformation(ones(Float32, 6), Int32.([1, 0, 1]), test_cyclops.scale) # 2
                    @test_throws "Input = 6 ≠ 5 = Multi-hot Parameters" CheckMultiHotTransformation(ones(Float32, 6), Int32.([1, 0, 1]), test_cyclops.scale)    # 3
                    @test_throws CyclopsInputMultiHotDimensionMismatch mhe(ones(Float32, 6), Int32.([1, 0, 1]), test_cyclops)   # 4
                    @test_throws "Input = 6 ≠ 5 = Multi-hot Parameters" mhe(ones(Float32, 6), Int32.([1, 0, 1]), test_cyclops)  # 5
                    @test_throws CyclopsInputMultiHotDimensionMismatch mhd(ones(Float32, 6), Int32.([1, 0, 1]), test_cyclops)   # 6
                    @test_throws "Input = 6 ≠ 5 = Multi-hot Parameters" mhd(ones(Float32, 6), Int32.([1, 0, 1]), test_cyclops)  # 7
                    @test_throws CyclopsInputMultiHotDimensionMismatch test_cyclops(ones(Float32, 6), Int32.([1, 0, 1]))    # 8
                    @test_throws "Input = 6 ≠ 5 = Multi-hot Parameters" test_cyclops(ones(Float32, 6), Int32.([1, 0, 1]))   # 9
                end
                
                # Model parameters and multi-hot encoding don't have fitting dimensions, 9 tests
                @testset "m mismatch input" begin # 9 tests
                    Random.seed!(1234); test_cyclops = cyclops(5, 3, 2);
                    @test length(methods(test_cyclops)) == 3    # 1
                    @test_throws CyclopsMultiHotParameterDimensionMismatch CheckMultiHotTransformation(ones(Float32, 5), Int32.([1, 0]), test_cyclops.scale) # 2
                    @test_throws "Multi-hot encoding = 2 ≠ 3 = Multi-hot Parameters" CheckMultiHotTransformation(ones(Float32, 5), Int32.([1, 0]), test_cyclops.scale)  # 3
                    @test_throws CyclopsMultiHotParameterDimensionMismatch mhe(ones(Float32, 5), Int32.([1, 0]), test_cyclops) # 4
                    @test_throws "Multi-hot encoding = 2 ≠ 3 = Multi-hot Parameters" mhe(ones(Float32, 5), Int32.([1, 0]), test_cyclops)    # 5
                    @test_throws CyclopsMultiHotParameterDimensionMismatch mhd(ones(Float32, 5), Int32.([1, 0]), test_cyclops)  # 6
                    @test_throws "Multi-hot encoding = 2 ≠ 3 = Multi-hot Parameters" mhd(ones(Float32, 5), Int32.([1, 0]), test_cyclops)    # 7
                    @test_throws CyclopsMultiHotParameterDimensionMismatch test_cyclops(ones(Float32, 5), Int32.([1, 0]))   # 8
                    @test_throws "Multi-hot encoding = 2 ≠ 3 = Multi-hot Parameters" test_cyclops(ones(Float32, 5), Int32.([1, 0])) # 9
                end
                
            end
            
            @testset "valid arguments" begin # 8 tests
                n = 5; m = 3; c = 2
                h = Int32.([1, 0, 1])
                x = ones(Float32, n)
                
                Random.seed!(1234); test_cyclops = cyclops(n, m, c);
                
                @test CheckMultiHotTransformation(x, h, test_cyclops.scale) isa Nothing # 1
                
                mhe_out = mhe(x, h, test_cyclops)
                @test mhe_out isa Array{Float32}    # 2
                @test size(mhe_out) == (n,)         # 3
                
                manual_mhe_out = x .* (1 .+ test_cyclops.scale[:, 1] .+ test_cyclops.scale[:, 3]) .+ test_cyclops.mhoffset[:, 1] .+ test_cyclops.mhoffset[:, 3] .+ reshape(test_cyclops.offset, length(x))
                @test isapprox(mhe_out, manual_mhe_out, atol=1E-6)  # 4
                
                mhd_out = mhd(mhe_out, h, test_cyclops)
                @test mhd_out isa Array{Float32}        # 5
                @test size(mhd_out) == (n,)             # 6
                @test isapprox(x, mhd_out, atol=1E-6)   # 7
                
                manual_mhd_out = (mhe_out .- test_cyclops.mhoffset[:,1] .- test_cyclops.mhoffset[:,3] .- reshape(test_cyclops.offset, length(mhe_out))) ./ (1 .+ test_cyclops.scale[:,1] .+ test_cyclops.scale[:,3])
                @test isapprox(mhd_out, manual_mhd_out, atol=1E-6)  # 8
            end
            
        end
        
        @testset "hypersphere node" begin # 15 tests

            @test length(methods(CheckHSNdomain)) == 1
            m_CheckHSNdomain = methods(CheckHSNdomain);
            @test any(m_CheckHSNdomain -> m_CheckHSNdomain.sig == Tuple{typeof(CheckHSNdomain), Vector{Float32}}, m_CheckHSNdomain)

            @test length(methods(hsn)) == 1
            m_hsn = methods(hsn);
            @test any(m_hsn -> m_hsn.sig == Tuple{typeof(hsn), Vector{Float32}}, m_hsn)
            
            @testset "invalid arguments" begin # 11 tests
                
                @testset "NaN erros" begin # 4 tests
                    @test_throws CyclopsHyperSphereDomainError CheckHSNdomain([1f0, NaN32]) # 1
                    @test_throws "`NaN` at [2]" CheckHSNdomain([1f0, NaN32])                # 2
                    @test_throws CyclopsHyperSphereDomainError hsn([1f0, NaN32])            # 3
                    @test_throws "`NaN` at [2]" hsn([1f0, NaN32])                           # 4
                end
                
                @testset "All 0 errors" begin # 4 tests
                    @test_throws CyclopsHyperSphereDivideError CheckHSNdomain([0f0, 0f0])   # 1
                    @test_throws "All values passed to the hypershpere node are `0`." CheckHSNdomain([0f0, 0f0])    # 2
                    @test_throws CyclopsHyperSphereDivideError hsn([0f0, 0f0])  # 3
                    @test_throws "All values passed to the hypershpere node are `0`." hsn([0f0, 0f0])   # 4
                end

                @testset "Method errors" begin # 3 tests
                    @test_throws MethodError CheckHSNdomain([1.1, 1.1]) # 1
                    @test_throws MethodError CheckHSNdomain([1, 1])     # 2
                    @test_throws MethodError CheckHSNdomain("1, 1")     # 3
                end
            end

            @testset "valid arguments" begin # 4 tests
                hsn_in = [1f0, 1f0];

                @test hsn_in isa Array{Float32} # 1

                hsn_out = hsn(hsn_in);
                @test hsn_out isa Array{Float32}

                @test hsn_out == Float32.([sqrt(2)^-1, sqrt(2)^-1]) # 2
                @test isapprox(hsn_out, hsn(hsn_out), atol=1E-6)    # 3
                @test isapprox(atan(hsn_out...)*180/pi, 45, atol=1E-6)  # 4
            end
            
        end

        @testset "cyclops transformation" begin

            Random.seed!(1234); test_cyclops = cyclops(5, 3, 2);
            m_test_cyclops = methods(test_cyclops);
                    
            @test length(methods(test_cyclops)) == 3 # 1
            @test any(m_test_cyclops -> m_test_cyclops.sig == Tuple{cyclops, Vector{Float32}}, m_test_cyclops)
            @test any(m_test_cyclops -> m_test_cyclops.sig == Tuple{cyclops, Vector{Float32}, Vector{Int32}}, m_test_cyclops)
            @test any(m_test_cyclops -> m_test_cyclops.sig == Tuple{cyclops, Vector{Float32}, Missing}, m_test_cyclops)

            @testset "invalid arguments" begin
                n = 5; m = 3; c = 2;
                Random.seed!(1234); m1 = cyclops(n, 0, c);
                Random.seed!(1234); m2 = cyclops(n, m, c);
                
                Random.seed!(1234); x1 = rand(Float64, n);
                h = zeros(Int32, m);
                
                @test_throws MethodError m1(x1)
                @test_throws MethodError m2(x1)

                Random.seed!(1234); x2 = rand(Float32, n);

                @test_throws ErrorException m1(x2, h)
            end
            
            @testset "valid arguments" begin
                n = 5; m = 3; c = 2;
                Random.seed!(1234); m1 = cyclops(n, m, c);
                Random.seed!(1234); m2 = cyclops(n, 0, c);
                
                Random.seed!(1234); x = rand(Float32, n);
                h = zeros(Int32, m);
                
                m_out_with_h = m1(x, h);
                @test m_out_with_h isa Array{Float32}
                @test size(m_out_with_h) == (n,)
                
                m1_out_without_h = m1(x);
                @test m_out_without_h isa Array{Float32}
                @test size(m_out_without_h) == (n,)
                
                @test isapprox(m_out_with_h, m1_out_without_h, atol=1E-6)
                
                m2_out_without_h = m2(x);
                @test m2_out_without_h isa Array{Float32}
                @test size(m2_out_without_h) == (n,)
                @test isapprox(m1_out_without_h, m2_out_without_h, atol=1E-6)
                
                Random.seed!(1234); m2 = cyclops(n, m, c);
                @test (@test_logs (:warn, r"without") m2(x2)) == m2(x2)

            end

        end

    end

end
