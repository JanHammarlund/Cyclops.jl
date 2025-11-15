using CYCLOPS
using Test
using Random

# CYCLOPS
#   cyclops constructor, 41 tests tests
#       ✓ invalid arguments, 18 tests
#           ✓ invalid c, c < 2, 4 tests
#               ✓ CheckCyclopsInput, 2 tests
#               ✓ cyclops, 2 tests
#           ✓ invalid n, n ≤ c, 4 tests
#               ✓ CheckCyclopsInput, 2 tests
#               ✓ cyclops, 2 tests
#           ✓ invalid m, m < 0, 4 tests
#               ✓ CheckCyclopsInput, 2 tests
#               ✓ cyclops, 2 tests
#           ✓ method error, 6 tests
#               ✓ CheckCyclopsInput, 3 tests
#               ✓ cyclops, 3 tests
#       ✓ valid arguments, 22 tests
#           ✓ multi-hot model, 11 tests
#               ✓ type, 1 test
#               ✓ fieldnames, 1 test
#               ✓ parameter number, 1 test
#               ✓ property dimensions, 7 tests
#           ✓ standard model, 11 tests
#               ✓ type, 1 test
#               ✓ fieldnames, 1 test
#               ✓ parameter number, 1 test
#               ✓ property dimensions, 7 tests
#   cyclops function
#       multi-hot layer
#           invalid arguments, 16 tests
#               ✓ n doesn't match input, 8 tests
#                   ✓ CheckMultiHotTransformation, 2 tests
#                   ✓ mhe, 2 tests
#                   ✓ mhd, 2 tests
#                   ✓ m::cyclops, 2 tests
#               ✓ m doesn't match input, 8 tests
#                   ✓ CheckMultiHotTransformation, 2 tests
#                   ✓ mhe, 2 tests
#                   ✓ mhd, 2 tests
#                   ✓ m::cyclops, 2 tests
#           valid arguments
#       hypershpere node
#           invalid arguments
#           valid arguments, 15 tests
#       

@testset "CYCLOPS" begin

    @test cyclops isa DataType          # 1
    @test length(methods(cyclops)) == 5 # 2

    @testset "cyclops constructor" begin # 41 tests

        @test length(methods(CheckCyclopsInput)) == 1   # 1

        @testset "invalid arguments" begin # 18 tests

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

        @test length(methods(CheckMultiHotTransformation)) == 1 # 1
        @test length(methods(mhe)) == 1                         # 2
        @test length(methods(mhd)) == 1                         # 3
        @test length(methods(hsn)) == 1                         # 4

        @testset "multi-hot node" begin # 26 tests

            @testset "invalid arguments" begin # 18 tests
    
                # Model parameters and model input don't have same number of rows, 9 tests
                @testset "n mismatch input" begin # 9 tests
                    Random.seed!(1234); test_cyclops = cyclops(5, 3, 2);
                    @test length(methods(test_cyclops)) == 3 # 1
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

                @test hsn_out == Float32.([sqrt(2)^-1, sqrt(2)^-1]) # 2
                @test isapprox(hsn_out, hsn(hsn_out), atol=1E-6)    # 3
                @test isapprox(atan(hsn_out...)*180/pi, 45, atol=1E-6)  # 4
            end
            
        end

        @testset "cyclops transformation" begin

            @testset "invalid arguments" begin
                n = 5; m = 3; c = 2;
                Random.seed!(1234); m1 = cyclops(n, 0, c);
                
                Random.seed!(1234); x1 = rand(Float64, n);
                h = zeros(Int32, m);
                
                @test_throws MethodError m1(x1)

                Random.seed!(1234); x2 = rand(Float32, n);

                @test_throws ErrorException m1(x2, h)
            end
            
            @testset "valid arguments" begin
                n = 5; m = 3; c = 2;
                Random.seed!(1234); m1 = cyclops(n, m, c);
                
                Random.seed!(1234); x = rand(Float32, n);
                h = zeros(Int32, m);
                
                m_out_with_h = m1(x, h);
                @test m_out_with_h isa Array{Float32}
                
                m_out_without_h = m1(x);
                @test m_out_without_h isa Array{Float32}
                
                @test isapprox(m_out_with_h, m_out_without_h, atol=1E-6)
                
                Random.seed!(1234); m2 = cyclops(n, m, c);
                @test (@test_logs (:warn, r"without") m2(x2)) == m2(x2)
            end

        end
    end

end
