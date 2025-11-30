#############################
##### Cyclops Overload Call #
#############################
"""
    (m::cyclops)(x::Vector{Float32}, h::Vector{Int32}=missing[; silence::Bool=false])
    (m::cyclops)(x::Vector{Float32}, h::Missing[; silence::Bool=false])

Returns reconstruction of input (`x`) after compression through an n-sphere node,
with optional linear transformation according to multi-hot encoding (`h`).

    x₁ = mhe(x, h, m) = x ⊙ (1 + m.scale ⊗ h) + m.mhoffset ⊗ h + m.offset
    x₂ = m.densein(x₁) =  m.densein.weight ⊗ x₁ + m.densein.bias
    x₃ = hsn(x₂) = x₂ / √(∑(x₂ᵢ²))
    x₄ = m.denseout(x₃) = m.denseout.weight * x₃ + m.denseout.bias
    x₅ = mhd(x₄, h, m) = (x - m.mhoffset ⊗ h - m.offset) / (1 + m.scale ⊗ h)

When `h` is a `Vector{Int32}` the same length as `m`:
1) `x` is transformed according to its multi-hot encoding using `mhe`,
2) reduced from `n` to `c` dimensions using `m.densein`, 
3) normalized to a point on the `c`-dimensional hypersphere using `hsn`, 
4) expanded from `c` to `n` dimensions using `m.denseout`,
5) and decoded accroding to `x`'s multi-hot encoding using `mhd`.

When `h` is `missing` steps `1` and `5` are skipped and steps `2`-`4` are performed.

    x₁ = m.densein(x) =  m.densein.weight ⊗ x + m.densein.bias
    x₂ = hsn(x₁) = x₁ / √(∑(x₁ᵢ²))
    x₃ = m.denseout(x₂) = m.denseout.weight ⊗ x₂ + m.denseout.bias

# See also
[`mhe`](@ref), [`mhd`](@ref), [`hsn`](@ref), [`nparams`](@ref)

# Examples
```julia-repl
julia> Random.seed!(1234); covariate_cyclops_model = cyclops(5, 3)
cyclops(
    5x3 Matrix{Float32},                  # 15 parameters
    5x3 Matrix{Float32},                  # 15 parameters
    5x1 Matrix{Float32},                  # 5 parameters
    Dense(5 => 2),                        # 12 parameters
    Dense(2 => 5),                        # 15 parameters
)                   # Total: 7 arrays, 62 parameters, 640 bytes.

julia> Random.seed!(1234); x = rand(Float32, 5)
5-element Vector{Float32}:
    0.72619927
    0.32597667
    0.30699807
    0.5490511
    0.7889189

julia> h = Int32[1, 0, 1];

julia> x₁ = mhe(x, h, covariate_cyclops_model)
5-element Vector{Float32}:
    2.1732361
    2.2111228
    2.8991385
    3.2093358
    4.121205

julia> x₂ = covariate_cyclops_model.densein.weight * x₁ ⊕ covariate_cyclops_model.densein.bias
2-element Vector{Float32}:
    -0.29033667
    -2.1314127

julia> x₃ = hsn(x₂)
2-element Vector{Float32}:
    -0.13497148
    -0.9908495

julia> x₄ = covariate_cyclops_model.denseout.weight * x₃ + covariate_cyclops_model.denseout.bias
5-element Vector{Float32}:
    0.7395259
    -0.38189223
    -0.10315811
    0.25443074
    -0.30430767

julia> x₅ = mhd(x₄, h, covariate_cyclops_model)
5-element Vector{Float32}:
    0.02674055
    -1.1813868
    -1.6844633
    -0.6318831
    -1.5611184    

julia> y = covariate_cyclops_model(x, h)
5-element Vector{Float32}:
    0.02674055
    -1.1813868
    -1.6844633
    -0.6318831
    -1.5611184

julia> isapprox(x₅, y, atol=1E-6)
true
```

# Errors
- Throws an `ErrorException` if `m` has 0 parameters

# Extended help
TO DO:
- Both methods for covariate model
- Only one method for non-covariate model
"""
function (m::cyclops)(input_data::AbstractVector{T}, multihot::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    CheckCyclopsInput(input_data, multihot, m.scale)
    multihot_encoding = mhe(input_data, multihot, m, skip_check=true)
    dense_encoding = m.densein(multihot_encoding)
    circular_encoding = hsn(dense_encoding)
    dense_decoding = m.denseout(circular_encoding)
    output = mhd(dense_decoding, multihot, m, skip_check=true)
    return output
end

function (m::cyclops)(input_data::AbstractVector{T}, multihot::Missing=missing; silence::Bool=false) where {T<:AbstractFloat}
    silence || length(m.scale) == 0 || @warn "Cyclops model with multi-hot parameters used without multi-hot encoding."
    CheckCyclopsInput(input_data, multihot, m.scale)
    dense_encoding = m.densein(input_data)
    circular_encoding = hsn(dense_encoding)
    output = m.denseout(circular_encoding)
    return output
end

function (m::cyclops)(input_data::AbstractMatrix{T}, multi_hot::AbstractMatrix{<:Integer}) where {T<:AbstractFloat}
    size(input_data, 2) == size(multi_hot, 2) || throw(DimensionMismatch("`x` and `h` do not have matching number of columns."))
    output = similar(input_data)
    @inbounds for jj in axes(input_data, 2)
        xj = view(input_data, :, jj)
        hj = view(multi_hot, :, jj)
        output[:, jj] = m(xj, hj)
    end
    return output
end

function (m::cyclops)(input_data::AbstractMatrix{T}, multi_hot::Missing=missing; silence::Bool=false) where {T<:AbstractFloat}
    output = similar(input_data)
    @inbounds for (jj, xj) in enumerate(eachcol(input_data))
        output[:, jj] = m(xj, multi_hot, silence = silence)
    end
    return output
end




####################################
##### Check Cyclops Overload Input #
####################################
"""
    CheckCyclopsInput(x::Vector{Float32}, h::Vector{Int}, m::Vector{Float32})

Checks inputs to multi-hot transformation and returns `nothing` if input data `x` and multi-hot parameters `m` have the
same number of rows, and if the multi-hot encoding `h` has as many rows as the multi-hot parameters `m` have columns.
"""
function CheckCyclopsInput(x::AbstractVector{T}, h::AbstractVector{<:Integer}, m::Array{Float32}) where {T<:AbstractFloat}
    (length(x) != size(m, 1)) && throw(CyclopsInputDimensionMismatch(x, m))
    (length(h) != size(m, 2)) && throw(CyclopsMultihotDimensionMismatch(h, m))

    return nothing
end

function CheckCyclopsInput(x::AbstractVector{T}, h::Missing, m::Array{Float32}) where {T<:AbstractFloat}
    (length(x) != size(m, 1)) && throw(CyclopsInputDimensionMismatch(x, m))

    return nothing
end