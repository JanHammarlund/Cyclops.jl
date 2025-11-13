using CYCLOPS
using Test
using Random

@testset "CYCLOPS Domain Errors" begin
    testexprs = [
        :(cyclops(-1)),         # n_eig cannot be negative              [method 1]  1
        :(cyclops(0)),          # n_eig must be greater than 0          [method 1]  2
        :(cyclops(1)),          # n_eig must be greater than c (c ≥ 2)  [method 1]  3
        :(cyclops(-1, 1)),      # n_eig cannot be negative              [method 2]  4
        :(cyclops(0, 1)),       # n_eig must be greater than 0          [method 2]  5
        :(cyclops(1, 1)),       # n_eig must be greater than c (c ≥ 2)  [method 2]  6
        :(cyclops(-1, 1, 2)),   # n_eig cannot be negative              [method 3]  7
        :(cyclops(0, 1, 2)),    # n_eig must be greater than 0          [method 3]  8
        :(cyclops(1, 1, 2)),    # n_eig must be greater than c (c ≥ 2)  [method 3]  9
        :(cyclops(5, -1)),      # n_multi cannot be negative            [method 2]  10
        :(cyclops(5, -1, 2)),   # n_multi cannot be negative            [method 3]  11
        :(cyclops(5, 0, 1)),    # n_circ must be greater than 1 (c ≥ 2) [method 3]  12
        :(cyclops(5, 0, 5))     # n_eig must be greater than n_circ     [method 3]  13
    ]
    for testexpr in testexprs
        @test_throws DomainError eval(testexpr)
    end
end

@testset "CYCLOPS Model Parameters" begin
    Random.seed!(1234); testCyclops1 = cyclops(5, 3)
    nmodelparams1 = nparams(testCyclops1)

    Random.seed!(1234); testCyclops2 = cyclops(5)
    nmodelparams2 = nparams(testCyclops2)

    @test testCyclops1 isa cyclops
    @test testCyclops2 isa cyclops
    @test length(propertynames(testCyclops1)) == 5
    @test nmodelparams1 == ((5*3)+(5*3)+5+(2*5)+2+(5*2)+5)
    @test nmodelparams2 == ((5*0)+(5*0)+0+(2*5)+2+(5*2)+5)

    Random.seed!(1234);
    inputData = Float32.(rand(5))
    multiHot = Int32.([1, 0, 1])
    mhEncodedTransform = inputData .* (1 .+ testCyclops1.scale[:, 1] .+ testCyclops1.scale[:, 3]) .+ testCyclops1.mhoffset[:, 1] .+ testCyclops1.mhoffset[:, 3] .+ reshape(testCyclops1.offset, length(inputData))
    denseEncodedTransform = [sum(testCyclops1.densein.weight[1,:] .* mhEncodedTransform), sum(testCyclops1.densein.weight[2,:] .* mhEncodedTransform)]
    circularTransform = [denseEncodedTransform[1]/sqrt(denseEncodedTransform[1]^2 + denseEncodedTransform[2]^2), denseEncodedTransform[2]/sqrt(denseEncodedTransform[1]^2 + denseEncodedTransform[2]^2)]
    denseDecodedTransform = [sum(testCyclops1.denseout.weight[1,:] .* circularTransform), sum(testCyclops1.denseout.weight[2,:] .* circularTransform), sum(testCyclops1.denseout.weight[3,:] .* circularTransform), sum(testCyclops1.denseout.weight[4,:] .* circularTransform), sum(testCyclops1.denseout.weight[5,:] .* circularTransform)]
    mhDecodedTransform = (denseDecodedTransform .- testCyclops1.mhoffset[:,1] .- testCyclops1.mhoffset[:,3] .- reshape(testCyclops1.offset, length(denseDecodedTransform))) ./ (1 .+ testCyclops1.scale[:,1] .+ testCyclops1.scale[:,3])
    output = testCyclops1(inputData, multiHot)

    @test size(inputData) == size(output)
    @test isapprox(mhe(inputData, multiHot, testCyclops1), mhEncodedTransform, atol = 1E-6)
    @test isapprox(testCyclops1.densein(mhEncodedTransform), denseEncodedTransform, atol = 1E-6)
    @test isapprox(hsn(denseEncodedTransform), circularTransform, atol = 1E-6)
    @test isapprox(testCyclops1.denseout(circularTransform), denseDecodedTransform, atol = 1E-6)
    @test isapprox(mhd(denseDecodedTransform, multiHot, testCyclops1), mhDecodedTransform, atol = 1E-6)
    
    mhSkipDenseEncodedTransform = [sum(testCyclops1.densein.weight[1,:] .* inputData), sum(testCyclops1.densein.weight[2,:] .* inputData)]
    mhSkipCircularTransform = [mhSkipDenseEncodedTransform[1]/sqrt(mhSkipDenseEncodedTransform[1]^2 + mhSkipDenseEncodedTransform[2]^2), mhSkipDenseEncodedTransform[2]/sqrt(mhSkipDenseEncodedTransform[1]^2 + mhSkipDenseEncodedTransform[2]^2)]
    mhSkipDenseDecodedTransform = [sum(testCyclops1.denseout.weight[1,:] .* mhSkipCircularTransform), sum(testCyclops1.denseout.weight[2,:] .* mhSkipCircularTransform), sum(testCyclops1.denseout.weight[3,:] .* mhSkipCircularTransform), sum(testCyclops1.denseout.weight[4,:] .* mhSkipCircularTransform), sum(testCyclops1.denseout.weight[5,:] .* mhSkipCircularTransform)]
    mhSkipOutput = testCyclops1(inputData, silence=true)

    @test isapprox(mhSkipDenseDecodedTransform, mhSkipOutput, atol = 1E-6)
end
