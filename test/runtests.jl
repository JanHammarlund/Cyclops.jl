using CYCLOPS
using Test
using Random

# CYCLOPS
#   cyclops constructor, 38 tests
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
#       ✓ valid arguments, 20 tests
#           ✓ multi-hot model, 10 tests
#               ✓ type, 1 test
#               ✓ fieldnames, 1 test
#               ✓ parameter number, 1 test
#               ✓ property dimensions, 7 tests
#           ✓ standard model, 10 tests
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

@testset "CYCLOPS" begin

    @test cyclops isa DataType
    @test length(methods(cyclops)) == 5

    @testset "cyclops constructor" begin

        @test length(methods(CheckCyclopsInput)) == 1

        @testset "invalid arguments" begin

            # c < 2 error
            @testset "invalid c" begin
                @test_throws CyclopsHypersphereDimensionError CheckCyclopsInput(5,0,1)
                @test_throws "`c` = 1, but `c` must be ≥ 2." CheckCyclopsInput(5,0,1)
                @test_throws CyclopsHypersphereDimensionError cyclops(5,0,1)
                @test_throws "`c` = 1, but `c` must be ≥ 2." cyclops(5,0,1)
            end

            # n < c error
            @testset "invalid n" begin
                @test_throws CyclopsInputHypersphereDimensionError CheckCyclopsInput(2,0,2)
                @test_throws "`n` = 2 ≤ `c`, but `n` must be > 2." CheckCyclopsInput(2,0,2)
                @test_throws CyclopsInputHypersphereDimensionError cyclops(2,0,2)
                @test_throws "`n` = 2 ≤ `c`, but `n` must be > 2." cyclops(2,0,2)
            end

            # m < 0 error
            @testset "invalid m" begin
                @test_throws CyclopsMultiHotDimensionError CheckCyclopsInput(5,-1,2)
                @test_throws "`m` = -1 < 0, but `m` must be ≥ 0." CheckCyclopsInput(5,-1,2)
                @test_throws CyclopsMultiHotDimensionError cyclops(5,-1,2)
                @test_throws "`m` = -1 < 0, but `m` must be ≥ 0." cyclops(5,-1,2)
            end

            # non-Int error
            @testset "method errors" begin
                @test_throws MethodError CheckCyclopsInput(5f0,0,2)
                @test_throws MethodError CheckCyclopsInput(5,0f0,2)
                @test_throws MethodError CheckCyclopsInput(5,0,2f0)
                @test_throws MethodError cyclops(5f0,0,2)
                @test_throws MethodError cyclops(5,0f0,2)
                @test_throws MethodError cyclops(5,0,2f0)
            end
        end

        @testset "valid arguments" begin

            @testset "multi-hot model" begin
                n = 5; m = 3; c = 2;
                Random.seed!(1234); test_cyclops = cyclops(n, m)
                nmodelparams1 = nparams(test_cyclops)
                @test test_cyclops isa cyclops                          # 1
                @test length(methods(test_cyclops)) == 3                # 2
                @test length(propertynames(test_cyclops)) == 5          # 3
                @test nmodelparams1 == ((n*m)+(n*m)+n+(c*n)+c+(n*c)+n)  # 4
                @test size(test_cyclops.scale) == (n, m)                # 5
                @test size(test_cyclops.mhoffset) == (n, m)             # 6
                @test size(test_cyclops.offset) == (n, 1)               # 7
                @test size(test_cyclops.densein.weight) == (c, n)       # 8
                @test size(test_cyclops.densein.bias) == (c,)           # 9
                @test size(test_cyclops.denseout.weight) == (n, c)      # 10
                @test size(test_cyclops.denseout.bias) == (n,)          # 11
            end

            @testset "standard model" begin
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

        @test length(methods(CheckMultiHotTransformation)) == 1
        @test length(methods(mhe)) == 1
        @test length(methods(mhd)) == 1

        @testset "multi-hot node" begin

            @testset "invalid arguments" begin
    
                # Model parameters and model input don't have same number of rows
                @testset "n mismatch input" begin
                    Random.seed!(1234); test_cyclops = cyclops(5, 3, 2);
                    @test length(methods(test_cyclops)) == 1
                    @test_throws CyclopsInputMultiHotDimensionMismatch CheckMultiHotTransformation(ones(Float32, 6), Int32.([1, 0, 1]), test_cyclops.scale)
                    @test_throws "Input = 6 ≠ 5 = Multi-hot Parameters" CheckMultiHotTransformation(ones(Float32, 6), Int32.([1, 0, 1]), test_cyclops.scale)
                    @test_throws CyclopsInputMultiHotDimensionMismatch mhe(ones(Float32, 6), Int32.([1, 0, 1]), test_cyclops)
                    @test_throws "Input = 6 ≠ 5 = Multi-hot Parameters" mhe(ones(Float32, 6), Int32.([1, 0, 1]), test_cyclops)
                    @test_throws CyclopsInputMultiHotDimensionMismatch mhd(ones(Float32, 6), Int32.([1, 0, 1]), test_cyclops)
                    @test_throws "Input = 6 ≠ 5 = Multi-hot Parameters" mhd(ones(Float32, 6), Int32.([1, 0, 1]), test_cyclops)
                    @test_throws CyclopsInputMultiHotDimensionMismatch test_cyclops(ones(Float32, 6), Int32.([1, 0, 1]))
                    @test_throws "Input = 6 ≠ 5 = Multi-hot Parameters" test_cyclops(ones(Float32, 6), Int32.([1, 0, 1]))
                end
    
                # Model parameters and multi-hot encoding don't have fitting dimensions
                @testset "m mismatch input" begin
                    Random.seed!(1234); test_cyclops = cyclops(5, 3, 2);
                    @test length(methods(test_cyclops)) == 1
                    @test_throws CyclopsMultiHotParameterDimensionMismatch CheckMultiHotTransformation(ones(Float32, 5), Int32.([1, 0]), test_cyclops.scale)
                    @test_throws "Multi-hot encoding = 2 ≠ 3 = Multi-hot Parameters" CheckMultiHotTransformation(ones(Float32, 5), Int32.([1, 0]), test_cyclops.scale)
                    @test_throws CyclopsMultiHotParameterDimensionMismatch mhe(ones(Float32, 5), Int32.([1, 0]), test_cyclops)
                    @test_throws "Multi-hot encoding = 2 ≠ 3 = Multi-hot Parameters" mhe(ones(Float32, 5), Int32.([1, 0]), test_cyclops)
                    @test_throws CyclopsMultiHotParameterDimensionMismatch mhd(ones(Float32, 5), Int32.([1, 0]), test_cyclops)
                    @test_throws "Multi-hot encoding = 2 ≠ 3 = Multi-hot Parameters" mhd(ones(Float32, 5), Int32.([1, 0]), test_cyclops)
                    @test_throws CyclopsMultiHotParameterDimensionMismatch test_cyclops(ones(Float32, 5), Int32.([1, 0]))
                    @test_throws "Multi-hot encoding = 2 ≠ 3 = Multi-hot Parameters" test_cyclops(ones(Float32, 5), Int32.([1, 0]))
                end
    
            end

            @testset "valid arguments" begin
                n = 5; m = 3; c = 2
                h = Int32.([1, 0, 1])
                x = ones(Float32, n)
                
                Random.seed!(1234); test_cyclops = cyclops(n, m, c);
                
                @test CheckMultiHotTransformation(x, h, test_cyclops.scale) isa Nothing
                
                mhe_out = mhe(x, h, test_cyclops)
                @test mhe_out isa Array{Float32}
                @test size(mhe_out) == (n,)
                
                manual_mhe_out = x .* (1 .+ test_cyclops.scale[:, 1] .+ test_cyclops.scale[:, 3]) .+ test_cyclops.mhoffset[:, 1] .+ test_cyclops.mhoffset[:, 3] .+ reshape(test_cyclops.offset, length(x))
                @test isapprox(mhe_out, manual_mhe_out, atol=1E-6)

                mhd_out = mhd(mhe_out, h, test_cyclops)
                @test mhd_out isa Array{Float32}
                @test size(mhd_out) == (n,)
                @test isapprox(x, mhd_out, atol=1E-6)
            end
            
        end

        @testset "hypersphere node" begin

            @test length(methods(hsn)) == 1

            @testset "invalid arguments" begin

                @testset "NaN erros" begin
                    @test_throws CyclopsHyperSphereDomainError CheckHSNdomain([1f0, NaN32])
                    @test_throws "`NaN` at [2]" CheckHSNdomain([1f0, NaN32])
                    @test_throws CyclopsHyperSphereDomainError hsn([1f0, NaN32])
                    @test_throws "`NaN` at [2]" hsn([1f0, NaN32])
                end

                @testset "All 0 errors" begin
                    @test_throws CyclopsHyperSphereDivideError CheckHSNdomain([0f0, 0f0])
                    @test_throws "All values passed to the hypershpere node are `0`." CheckHSNdomain([0f0, 0f0])
                    @test_throws CyclopsHyperSphereDivideError hsn([0f0, 0f0])
                    @test_throws "All values passed to the hypershpere node are `0`." hsn([0f0, 0f0])                    
                end

                @testset "Method errors" begin
                    @test_throws MethodError CheckHSNdomain([1.1, 1.1])
                    @test_throws MethodError CheckHSNdomain([1, 1])
                    @test_throws MethodError CheckHSNdomain("1, 1")
                end
            end

            @testset "valid arguments" begin
                hsn_in = [1f0, 1f0];

                @test hsn_in isa Array{Float32}

                hsn_out = hsn(hsn_in);

                @test hsn_out == Float32.([sqrt(2)^-1, sqrt(2)^-1])
                @test isapprox(hsn_out, hsn(hsn_out), atol=1E-6)
                @test isapprox(atan(hsn_out...)*180/pi, 45, atol=1E-6)
            end
            
        end


    end

    Random.seed!(1234); inputData = Float32.(rand(5));
    multiHot = Int32.([1, 0, 1]);

    mhEncodedTransform = inputData .* (1 .+ test_cyclops.scale[:, 1] .+ test_cyclops.scale[:, 3]) .+ test_cyclops.mhoffset[:, 1] .+ test_cyclops.mhoffset[:, 3] .+ reshape(test_cyclops.offset, length(inputData))

    denseEncodedTransform = [sum(test_cyclops.densein.weight[1,:] .* mhEncodedTransform), 
                                sum(test_cyclops.densein.weight[2,:] .* mhEncodedTransform)]

    circularTransform = [denseEncodedTransform[1]/sqrt(denseEncodedTransform[1]^2 + denseEncodedTransform[2]^2), denseEncodedTransform[2]/sqrt(denseEncodedTransform[1]^2 + denseEncodedTransform[2]^2)]

    denseDecodedTransform = [sum(test_cyclops.denseout.weight[1,:] .* circularTransform), 
                            sum(test_cyclops.denseout.weight[2,:] .* circularTransform), 
                            sum(test_cyclops.denseout.weight[3,:] .* circularTransform), 
                            sum(test_cyclops.denseout.weight[4,:] .* circularTransform), 
                            sum(test_cyclops.denseout.weight[5,:] .* circularTransform)]

    mhDecodedTransform = (denseDecodedTransform .- test_cyclops.mhoffset[:,1] .- test_cyclops.mhoffset[:,3] .- reshape(test_cyclops.offset, length(denseDecodedTransform))) ./ (1 .+ test_cyclops.scale[:,1] .+ test_cyclops.scale[:,3])

    output1 = test_cyclops(inputData, multiHot)
    output2 = test_cyclops(inputData)

    @test output1 isa Array{Float32}
    @test output2 isa Array{Float32}
    @test size(inputData) == size(output1)
    @test size(inputData) == size(output2)
        
    @test isapprox(mhe(inputData, multiHot, test_cyclops), mhEncodedTransform, atol = 1E-6)
    @test isapprox(test_cyclops.densein(mhEncodedTransform), denseEncodedTransform, atol = 1E-6)
    @test isapprox(hsn(denseEncodedTransform), circularTransform, atol = 1E-6)
    @test isapprox(test_cyclops.denseout(circularTransform), denseDecodedTransform, atol = 1E-6)
    @test isapprox(mhd(denseDecodedTransform, multiHot, test_cyclops), mhDecodedTransform, atol = 1E-6)
        
    mhSkipDenseEncodedTransform = [sum(test_cyclops.densein.weight[1,:] .* inputData), sum(test_cyclops.densein.weight[2,:] .* inputData)]
    mhSkipCircularTransform = [mhSkipDenseEncodedTransform[1]/sqrt(mhSkipDenseEncodedTransform[1]^2 + mhSkipDenseEncodedTransform[2]^2), mhSkipDenseEncodedTransform[2]/sqrt(mhSkipDenseEncodedTransform[1]^2 + mhSkipDenseEncodedTransform[2]^2)]
    mhSkipDenseDecodedTransform = [sum(test_cyclops.denseout.weight[1,:] .* mhSkipCircularTransform), sum(test_cyclops.denseout.weight[2,:] .* mhSkipCircularTransform), sum(test_cyclops.denseout.weight[3,:] .* mhSkipCircularTransform), sum(test_cyclops.denseout.weight[4,:] .* mhSkipCircularTransform), sum(test_cyclops.denseout.weight[5,:] .* mhSkipCircularTransform)]
    mhSkipOutput = test_cyclops(inputData, silence=true)

    @test isapprox(mhSkipDenseDecodedTransform, mhSkipOutput, atol = 1E-6)
    
    @testset "Hypersphere node errors" begin
        # NaN error
        @test_throws CyclopsHyperSphereDomainError CheckHSNdomain([1f0, NaN32])
        @test_throws "`NaN` at [2]" CheckHSNdomain([1f0, NaN32])
        @test_throws CyclopsHyperSphereDomainError hsn([1f0, NaN32])
        @test_throws "`NaN` at [2]" hsn([1f0, NaN32])
        # All 0 error    
        @test_throws CyclopsHyperSphereDivideError CheckHSNdomain([0f0, 0f0])
        @test_throws "All values passed to the hypershpere node are `0`." CheckHSNdomain([0f0, 0f0])
        @test_throws CyclopsHyperSphereDivideError hsn([0f0, 0f0])
        @test_throws "All values passed to the hypershpere node are `0`." hsn([0f0, 0f0])
    end

end