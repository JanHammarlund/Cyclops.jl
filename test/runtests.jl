using Cyclops
using Test

@testset "CYCLOPS.jl" begin
    @test plusTwo() == 2
    @test plusTwo(2) == 4
end
