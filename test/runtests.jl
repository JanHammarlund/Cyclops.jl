using Cyclops
using Test
using Random
using Flux
using InteractiveUtils: subtypes

@testset "Cyclops" begin

@testset "Cyclops Error Hierarchy" begin
    cyclops_error_hierarchy = Dict(
        CyclopsError => [
            CyclopsConstructorError,
            CyclopsFunctionError
        ],
        CyclopsConstructorError => [
            CyclopsHypersphereDomainError,
            CyclopsInputAndHypersphereDomainError,
            CyclopsMultihotDomainError,
            CyclopsMultihotMatrixShapeError,
            CyclopsMultihotOffsetShapeError,
            CyclopsDenseInverseShapeError,
            CyclopsDenseShapeError
        ],
        CyclopsFunctionError => [
            CyclopsMultihotDimensionMismatch,
            CyclopsInputDimensionMismatch,
            CyclopsHypersphereNaNError,
            CyclopsHypersphereDivideError
        ]
    )

    # Each parent’s current subtypes must be drawn from the expected set.
    for (parent, children) in cyclops_error_hierarchy
        @test Set(subtypes(parent)) ⊆ Set(children)
    end # 3 abstract types with concrete types as children, 3 tests

    concrete_errors = [
        CyclopsHypersphereDomainError,
        CyclopsInputAndHypersphereDomainError,
        CyclopsMultihotDomainError,
        CyclopsMultihotMatrixShapeError,
        CyclopsMultihotOffsetShapeError,
        CyclopsDenseInverseShapeError,
        CyclopsDenseShapeError,
        CyclopsMultihotDimensionMismatch,
        CyclopsInputDimensionMismatch,
        CyclopsHypersphereNaNError,
        CyclopsHypersphereDivideError
    ]

    for T in concrete_errors
        @test !isabstracttype(T)
    end # 11 concrete types, 11 tests

end # 14 tests

@testset "Expected Errors" begin
    
    @testset "Constructor Errors" begin
        # Errors encountered while initializing a variable::cyclops
        
        @testset "Hypersphere Domain Error" begin
            @test CyclopsHypersphereDomainError isa DataType
            @test_throws CyclopsHypersphereDomainError cyclops(5, 0, 1)
            @test_throws "`c` = 1, but `c` must be ≥ 2." cyclops(5, 0, 1)
        end # 3 tests

        @testset "Input and Hypersphere Domain Error" begin
            @test CyclopsInputAndHypersphereDomainError isa DataType
            @test_throws CyclopsInputAndHypersphereDomainError cyclops(5, 0, 6)
            @test_throws "`n` = 5 ≤ `c`, but `n` must be > 6." cyclops(5, 0, 6)
        end # 3 tests

        @testset "Multi-hot Domain Error" begin
            @test CyclopsMultihotDomainError isa DataType
            @test_throws CyclopsMultihotDomainError cyclops(5, -1, 3)
            @test_throws "`m` = -1 < 0, but `m` must be ≥ 0." cyclops(5, -1, 3)
        end # 3 tests
            
        @testset "Multi-hot Matrix Shape Error" begin
            @test CyclopsMultihotMatrixShapeError isa DataType
            @test_throws CyclopsMultihotMatrixShapeError cyclops(rand(Float32, 5, 3), rand(Float32, 6, 4), rand(Float32, 5), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
            @test_throws "scale has dimensions (5, 3) ≠ (6, 4) dimensions of mhoffset." cyclops(rand(Float32, 5, 3), rand(Float32, 6, 4), rand(Float32, 5), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
        end # 3 tests
        
        @testset "Multi-hot Offset Shape Error" begin
            @test CyclopsMultihotOffsetShapeError isa DataType
            @test_throws CyclopsMultihotOffsetShapeError cyclops(rand(Float32, 5, 3), rand(Float32, 5, 3), rand(Float32, 6), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
            @test_throws "expected dimensions (5,), but got (6,)." cyclops(rand(Float32, 5, 3), rand(Float32, 5, 3), rand(Float32, 6), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
            @test_throws CyclopsMultihotOffsetShapeError cyclops(rand(Float32, 5, 3), rand(Float32, 5, 3), rand(Float32, 5, 1), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
            @test_throws "expected dimensions (5,), but got (5, 1)." cyclops(rand(Float32, 5, 3), rand(Float32, 5, 3), rand(Float32, 5, 1), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
            @test_throws CyclopsMultihotOffsetShapeError cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), zeros(Float32, 5), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
            @test_throws "expected dimensions (5, 0), but got (5,)." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), zeros(Float32, 5), Flux.Dense(5 => 2), Flux.Dense(2 => 5))
        end # 7 tests
        
        @testset "Inverse Dimension Error" begin
            @test CyclopsDenseInverseShapeError isa DataType
            # densein and denseout must have inverse dimensions
            @test_throws CyclopsDenseInverseShapeError cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(5 => 2), Flux.Dense(3 => 5))
            @test_throws "Expected 5 => 2 compression to be mirrored by 2 => 5 expansion, but got 3 => 5." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(5 => 2), Flux.Dense(3 => 5))
            
            @test_throws CyclopsDenseInverseShapeError cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(4 => 1), Flux.Dense(2 => 7))
            @test_throws "Expected 4 => 1 compression to be mirrored by 1 => 4 expansion, but got 2 => 7." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(4 => 1), Flux.Dense(2 => 7))
        end # 3 tests
        
        @testset "Dense Compression Error" begin
            @test CyclopsDenseShapeError isa DataType
            
            @test_throws CyclopsDenseShapeError cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(3 => 4), Flux.Dense(4 => 3))
            @test_throws "n => c ≥ 2, where n > c, but got 5 ≠ 3 => 4." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(3 => 4), Flux.Dense(4 => 3))
            
            @test_throws CyclopsDenseShapeError cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(5 => 1), Flux.Dense(1 => 5))
            @test_throws "n => c ≥ 2, where n > c, but got 5 => 1 < 2." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(5 => 1), Flux.Dense(1 => 5))
            
            @test_throws CyclopsDenseShapeError cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(6 => 2), Flux.Dense(2 => 6))
            @test_throws "n => c ≥ 2, where n > c, but got 5 ≠ 6 => 2." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(6 => 2), Flux.Dense(2 => 6))
            
            @test_throws CyclopsDenseShapeError cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(7 => 1), Flux.Dense(1 => 7))
            @test_throws "n => c ≥ 2, where n > c, but got 5 ≠ 7 => 1 < 2." cyclops(rand(Float32, 5, 0), rand(Float32, 5, 0), rand(Float32, 5, 0), Flux.Dense(7 => 1), Flux.Dense(1 => 7))
        end # 7 tests
    end # 29 tests

    @testset "Function Errors" begin

        @testset "Multi-hot Dimension Mismatch" begin
            @test CyclopsMultihotDimensionMismatch isa DataType
            @test_throws CyclopsMultihotDimensionMismatch cyclops(5, 2, 3)(rand(Float32, 5), zeros(Int32, 3))
            @test_throws "Multi-hot encoding = 3 ≠ 2 = Multi-hot Parameters" cyclops(5, 2, 3)(rand(Float32, 5), zeros(Int32, 3))
            @test_throws CyclopsMultihotDimensionMismatch cyclops(5, 0, 3)(rand(Float32, 5), zeros(Int32, 3))
            @test_throws "Multi-hot encoding = 3 ≠ 0 = Multi-hot Parameters" cyclops(5, 0, 3)(rand(Float32, 5), zeros(Int32, 3))
        end

        @testset "Input Dimension Mismatch" begin
           @test CyclopsInputDimensionMismatch isa DataType
           @test_throws CyclopsInputDimensionMismatch cyclops(5, 0, 3)(rand(Float32, 6))
           @test_throws "Input = 6 ≠ 5 = Multi-hot" cyclops(5, 0, 3)(rand(Float32, 6))
        end

        @testset "Input Array and Multihot Array Column Mismatch" begin
            @test_throws DimensionMismatch cyclops(5, 2, 3)(ones(Float32, 5, 2), ones(Int32, 2, 3))
            @test_throws "`x` and `h` do not have matching" cyclops(5, 2, 3)(ones(Float32, 5, 2), ones(Int32, 2, 3))
        end

        @testset "Hypersphere NaN Error" begin
            @test CyclopsHypersphereNaNError isa DataType
            @test_throws CyclopsHypersphereNaNError hsn([1f0, NaN32])
            @test_throws "`NaN` at [2]" hsn([1f0, NaN32])
        end

        @testset "Hypersphere Divide Error" begin
            @test CyclopsHypersphereDivideError isa DataType
            @test_throws CyclopsHypersphereDivideError hsn([0f0, 0f0])
            @test_throws "hypershpere node are `0`." hsn([0f0, 0f0])
        end

    end

end # 47 tests

# ⊙, ⊗, ⊕, ⊖, ⊘, ⩕
@testset "Operators" begin # 73 tests

    @testset "oplus" begin # 14 tests
        found_oplus_methods = Set(m.sig for m in methods(⊕));
        @test Tuple{typeof(⊕), Tx, AbstractVector{Ty}} where {Tx<:Real, Ty<:AbstractFloat} ∈ found_oplus_methods
        @test Tuple{typeof(⊕), AbstractVector{T}, AbstractVector{T}} where {T<:AbstractFloat} ∈ found_oplus_methods
        
        x = [1f0, 2f0, 3f0]     # ::AbstractArray{<:Number}
        y1 = 1                  # ::Number
        y2 = [3f0, 2f0, 1f0]    # ::AbstractArray{<:Number}
            
        # Tuple{typeof(⊕), Number, AbstractArray{<:Number}}
        @test y1 ⊕ x == [2, 3, 4]

        # Tuple{typeof(⊕), AbstractArray{<:Number}, AbstractArray{<:Number}}
        @test x ⊕ y2 == [4, 4, 4]
    end # 4 tests
    
    @testset "ominus" begin # 15 tests
        @test Tuple{typeof(⊖), AbstractVector{T}, AbstractVector{T}} where {T <: AbstractFloat} ∈ Set(m.sig for m in methods(⊖))

        x = [1f0, 2f0, 3f0]     # ::AbstractArray{<:Number}
        y2 = [3f0, 2f0, 1f0]    # ::AbstractArray{<:Number}

        # Tuple{typeof(⊖), AbstractVector{T}, AbstractVector{T}} where T <: AbstractFloat
        @test x ⊖ y2 == [-2, 0, 2]
    end
    
    @testset "otimes" begin # 9 tests
        @test Tuple{typeof(⊗), AbstractMatrix{Tx}, AbstractVector{Ty}} where {Tx <: AbstractFloat, Ty <: Real} ∈ Set(m.sig for m in methods(⊗))

        x = ones(3)
        y1 = [1, 0]
        y2 = [0, 1]
        y3 = [1, 0, 1]

        # Tuple{typeof(⊗), AbstractArray{<:Number}, AbstractArray{<:Number}}
        @test [x 2*x] ⊗ y1 == [1, 1, 1]
        @test [x 2*x] ⊗ y2 == [2, 2, 2]
    end     # otimes, 9 tests
    
    @testset "odot" begin # 14 tests
        @test Tuple{typeof(⊙), AbstractVector{T}, AbstractVector{T}} where {T <: AbstractFloat} ∈ Set(m.sig for m in methods(⊙))

        x = [1f0, 2f0, 3f0]   # ::AbstractArray{<:Number}
        y2 = [2f0, 3f0, 4f0]  # ::AbstractArray{<:Number}

        @test x ⊙ y2 == [2, 6, 12]
    end     # odot, 14 tests
    
    @testset "oslash" begin # 15 tests
        @test Tuple{typeof(⊘), AbstractVector{T}, AbstractVector{T}} where {T <: AbstractFloat} ∈ Set(m.sig for m in methods(⊘))
        @test Tuple{typeof(⊘), AbstractVector{T}, T} where {T <: AbstractFloat} ∈ Set(m.sig for m in methods(⊘))

        x = [1f0, 2f0, 3f0]   # ::AbstractArray{<:Number}
        y1 = 2f0          # ::Number
        y2 = [3f0, 2f0, 1f0]  # ::AbstractArray{<:Number}

        @test x ⊘ y1 == [0.5, 1, 1.5]
        @test y2 ⊘ x == [3f0, 1f0, 1/3f0]
    end     # oslash, 15 tests

    @testset "wedge on wedge" begin # 6 tests
        @test Tuple{typeof(⩕), AbstractVector{Tx}, Ty} where {Tx <: AbstractFloat, Ty <: Integer} ∈ Set(m.sig for m in methods(⩕))

        x = [1f0, 2f0, 3f0]
        y = 2
        
        @test x ⩕ y == [1, 4, 9]
    end # wedge on wedge, 6 tests
    
end # operators 73 tests

@testset "Constructor" begin

    @testset "cyclops" begin
        @test cyclops isa DataType
        @test Set(fieldnames(cyclops)) ⊆ Set([:scale, :mhoffset, :offset, :densein, :denseout])
        @test Set(m.sig for m in methods(cyclops)) ⊆ Set([
            Tuple{Type{cyclops}, Int64},
            Tuple{Type{cyclops}, Int64, Int64},
            Tuple{Type{cyclops}, Int64, Int64, Int64},
            Tuple{Type{cyclops}, AbstractMatrix{<:Real}, AbstractMatrix{<:Real}, AbstractVecOrMat{<:Real}, Dense, Dense}
        ])
        @test cyclops(3) isa cyclops 
        @test cyclops(3, 0) isa cyclops
        @test cyclops(3, 0, 2) isa cyclops
        @test cyclops(zeros(Float32, 3, 0), zeros(Float32, 3, 0), zeros(Float32, 3, 0), Flux.Dense(3 => 2), Flux.Dense(2 => 3)) isa cyclops
        @test cyclops(zeros(Float32, 3, 1), zeros(Float32, 3, 1), zeros(Float32, 3), Flux.Dense(3 => 2), Flux.Dense(2 => 3)) isa cyclops
        Random.seed!(1234); test_model = cyclops(3,2,2)
        @test test_model.scale isa Array{Float32}
        @test test_model.mhoffset isa Array{Float32}
        @test test_model.offset isa Array{Float32}
        @test test_model.densein isa Dense
        @test test_model.denseout isa Dense
        @test test_model.scale |> size == (3, 2)
        @test test_model.mhoffset |> size == (3, 2)
        @test test_model.offset |> size == (3,)
        @test test_model.densein.weight |> size == (2, 3)
        @test test_model.densein.bias |> size == (2,)
        @test test_model.denseout.weight |> size == (3, 2)
        @test test_model.denseout.bias |> size == (3,)
    end
    
    @testset "nparams" begin
        @test nparams isa Function
        @test methods(nparams)[1].sig == Tuple{typeof(nparams), cyclops}
        @test nparams(cyclops(5, 0, 2)) == 27 # n = 5; m = 0; c = 2; 2*n*c + n + c # For standard model
        @test nparams(cyclops(6, 3, 3)) == 87 # n = 6; m = 3; c = 3; (4*n*m + 2*n + m) # For multi-hot model
        @test nparams(cyclops(5, 2, 2)) == 52 # n = 5; m = 2; c = 2; (4*n*m + 2*n + m) # For multi-hot model
    end
    
end

@testset "Function" begin
    @test Tuple{cyclops, AbstractMatrix{T}} where {T <: AbstractFloat} ∈ Set(m.sig for m in methods(cyclops(3)))
    @test Tuple{cyclops, AbstractMatrix{T}, Missing} where {T <: AbstractFloat} ∈ Set(m.sig for m in methods(cyclops(3)))
    @test Tuple{cyclops, AbstractMatrix{T}, AbstractMatrix{<:Integer}} where {T <: AbstractFloat} ∈ Set(m.sig for m in methods(cyclops(3)))
    @test Tuple{cyclops, AbstractVector{T}} where {T <: AbstractFloat} ∈ Set(m.sig for m in methods(cyclops(3)))
    @test Tuple{cyclops, AbstractMatrix{T}, Missing} where {T <: AbstractFloat} ∈ Set(m.sig for m in methods(cyclops(3)))
    @test Tuple{cyclops, AbstractVector{T}, AbstractVector{<:Integer}} where {T <: AbstractFloat} ∈ Set(m.sig for m in methods(cyclops(3)))
    
    Random.seed!(1234); test_cyclops = cyclops(3, 2, 2)
    @test test_cyclops(ones(Float32, 3), ones(Int32, 2)) isa Vector{Float32}
    @test test_cyclops(ones(Float32, 3)) isa Vector{Float32}
    @test test_cyclops(ones(Float32, 3), missing) isa Vector{Float32}
    @test test_cyclops(ones(Float32, 3), ones(Int32, 2)) |> size == (3,)
    @test test_cyclops(ones(Float32, 3)) |> size == (3,)
    @test test_cyclops(ones(Float32, 3), missing) |> size == (3,)
    @test test_cyclops(ones(Float32, 3, 2), ones(Int32, 2, 2)) isa Matrix{Float32}
    @test test_cyclops(ones(Float32, 3, 2), ones(Int32, 2, 2)) |> size == (3,2)

    Random.seed!(1234); test_cyclops_2 = cyclops(3)
    @test test_cyclops_2(ones(Float32, 3)) isa Vector{Float32}
    @test test_cyclops_2(ones(Float32, 3), missing) isa Vector{Float32}
    @test test_cyclops_2(ones(Float32, 3)) |> size == (3,)
    @test test_cyclops_2(ones(Float32, 3), missing) |> size == (3,)
    @test test_cyclops_2(ones(Float32, 3, 2), missing) isa Matrix{Float32}
    @test test_cyclops_2(ones(Float32, 3, 2), missing) |> size == (3,2)
end

@testset "Layers" begin
    @testset "Multihot Layers" begin
        @test mhe isa Function
        @test Tuple{typeof(mhe), AbstractVector{T}, AbstractVector{<:Integer}, cyclops} where {T <: AbstractFloat} ∈ Set(m.sig for m in methods(mhe))
        @test Tuple{typeof(mhd), AbstractVector{T}, AbstractVector{<:Integer}, cyclops} where {T <: AbstractFloat} ∈ Set(m.sig for m in methods(mhd))
        Random.seed!(1234); test_cyclops = cyclops(3,2,2)
        @test mhe(ones(Float32, 3), ones(Int32, 2), test_cyclops) isa Vector{Float32}
        @test mhe(ones(Float32, 3), ones(Int32, 2), test_cyclops) |> size == (3,)
        @test isapprox(mhe(ones(Float32, 3), ones(Int32, 2), test_cyclops), [3.889213f0, 2.0930424f0, -0.064593464f0], atol = 1e-6)
        @test mhd(ones(Float32, 3), ones(Int32, 2), test_cyclops) isa Vector{Float32}
        @test mhd(ones(Float32, 3), ones(Int32, 2), test_cyclops) |> size == (3,)
        @test isapprox(mhd(ones(Float32, 3), ones(Int32, 2), test_cyclops), [0.23437645f0, 0.37366495f0, 113.92013f0], atol = 1e-6)
        @test isapprox(mhd(mhe(ones(Float32, 3), ones(Int32, 2), test_cyclops), ones(Int32, 2), test_cyclops), [1f0, 1f0, 1f0], atol=1e-6)
        @test isapprox(mhe(mhd(ones(Float32, 3), ones(Int32, 2), test_cyclops), ones(Int32, 2), test_cyclops), [1f0, 1f0, 1f0], atol=1e-6)
    end

    @testset "Hypersphere Node" begin
        @test hsn isa Function
        @test Tuple{typeof(hsn), AbstractVector{<:AbstractFloat}} ∈ Set(m.sig for m in methods(hsn))
        @test hsn([1f0, 1f0]) isa Vector{Float32}
        @test hsn([1f0, 1f0]) |> size == (2,)
        @test hsn([1f0, 0f0]) == [1f0, 0f0]
        @test isapprox(hsn([sqrt(0.5f0), sqrt(0.5f0)]), [sqrt(0.5f0), sqrt(0.5f0)], atol = 1e-6)
    end
end

end # 142 tests
