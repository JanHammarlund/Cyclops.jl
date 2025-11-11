module CYCLOPS
export cyclops, mhe, hsn, mhd, nparams
using CUDA, Flux, Statistics, ProgressMeter, Plots, Random

    """
        CyclopsHypersphereDimensionError(c::Int)

    An error when `c < 2`.

    # Examples
    ```julia-repl
    julia> n = 5; m = 0; c = 1; cyclops(n, m, c)
    ERROR: CyclopsHypersphereDimensionError: `c` = 1, but `c` must be ≥ 2.
    [...]
    ```

    # See also
    [`CheckCyclopsInput`](@ref), [`CyclopsInputHypersphereDimensionError`](@ref),
    [`CyclopsMultiHotDimensionError`](@ref), [`cyclops`](@ref)
    """
    struct CyclopsHypersphereDimensionError <: Exception end
    CyclopsHypersphereDimensionError(val::Int) = error("CyclopsHypersphereDimensionError: `c` = $(val), but `c` must be ≥ 2.")

    """
        CyclopsInputHypersphereDimensionError(n::Int, c::Int)

    An error when `n ≤ c`.

    # Examples
    ```julia-repl
    julia> n = 5; m = 0; c = 5; cyclops(n, m, c)
    ERROR: CyclopsInputHypersphereDimensionError: `n` = 5 ≤ `c`, but `n` must be > 5 or `c` must be < 5.
    [...]
    ```

    # See also
    [`CheckCyclopsInput`](@ref), [`CyclopsHypersphereDimensionError`](@ref),
    [`CyclopsMultiHotDimensionError`](@ref), [`cyclops`](@ref)
    """
    struct CyclopsInputHypersphereDimensionError <: Exception end
    CyclopsInputHypersphereDimensionError(nval::Int, cval::Int) = error("CyclopsInputHypersphereDimensionError: `n` = $(nval) ≤ `c`, but `n` must be > $(cval) or `c` must be < $(nval).")

    """
        CyclopsMultiHotDimensionError(m::Int)

    An error when `m < 0`.

    # Examples
    ```julia-repl
    julia> n = 5; m = -1; c = 3; cyclops(n, m, c)
    ERROR: CyclopsMultiHotDimensionError: `m` = -1 < 0, but `m` must be ≥ 0
    [...]
    ```

    # See also
    [`CheckCyclopsInput`](@ref), [`CyclopsHypersphereDimensionError`](@ref),
    [`CyclopsInputHypersphereDimensionError`](@ref), [`cyclops`](@ref)
    """
    struct CyclopsMultiHotDimensionError <: Exception end
    CyclopsMultiHotDimensionError(val::Int) = error("CyclopsMultiHotDimensionError: `m` = $(val) < 0, but `m` must be ≥ 0 ")

    """
        CheckCyclopsInput(n::Int, m::Int, c::Int)
    
    Checks domains of input arguments to `cyclops`, and returns `nothing` if all checks are passed.

    # Errors
    - `CyclopsHypersphereDimensionError` when `c < 2`
    - `CyclopsInputHypersphereDimensionError` when `n ≤ c`
    - `CyclopsMultiHotDimensionError` when `m < 0`

    # See also
    [`CyclopsHypersphereDimensionError`](@ref), [`CyclopsInputHypersphereDimensionError`](@ref)
    [`CyclopsMultiHotDimensionError`](@ref), [`cyclops`](@ref)

    # Examples
    ```julia-repl
    julia> n = 5; m = 0; c = 1; CYCLOPS.CheckCyclopsInput(n, m, c)
    ERROR: CyclopsHypersphereDimensionError: `c` = 1, but `c` must be ≥ 2.
    [...]

    julia> n = 5; m = 0; c = 5; CYCLOPS.CheckCyclopsInput(n, m, c)
    ERROR: CyclopsInputHypersphereDimensionError: `n` = 5 ≤ `c`, but `n` must be > 5 or `c` must be < 5.
    [...]

    julia> n = 5; m = -1; c = 3; CYCLOPS.CheckCyclopsInput(n, m, c)
    ERROR: CyclopsMultiHotDimensionError: `m` = -1 < 0, but `m` must be ≥ 0
    [...]
    ```
    """
    function CheckCyclopsInput(n_val::Int, m_val::Int, c_val::Int)
        c_val ≥ 2 || throw(CyclopsHypersphereDimensionError(c_val))
        n_val > c_val || throw(CyclopsInputHypersphereDimensionError(n_val, c_val))
        m_val ≥ 0 || throw(CyclopsMultiHotDimensionError(m_val))

        return nothing
    end

    struct cyclops
        scale::Array{Float32}
        mhoffset::Array{Float32}
        offset::Array{Float32}
        densein::Dense
        denseout::Dense
    end

    """
        cyclops(n::Int, m::Int[, c::Int=2])
        cyclops(n::Int)

    Creates an instance of cyclops.

    Type `cyclops` has fieldnames:
    - `scale` (`::Array{Float32}`)
    - `mhoffset` (`::Array{Float32}`)
    - `offset` (`::Array{Float32}`)
    - `densein` (`::Dense`)
    - `denseout` (`::Dense`)

    # Arguments
    - `n` (`n ∈ ℕ⁺`, `n > c`): Number of rows in the model's input data.
    - `m` (`m ∈ ℕ₀`): Number of groups in the input data's multi-hot encoding.
    - `c` (`c ∈ ℕ⁺`, `c ≥ 2`): Dimensionality of the n-sphere node, where `2 ≤ c < n`.

    # Initialization
    `scale`, `mhoffset`, and `offset` are initialized using random normal distributions. 
    `densein` and `denseout` are initialized according to `Flux.Dense`.

    `n` dictates the number of rows in `scale`, `mhoffset`, `offset`, 
    `denseout.weight`, `denseout.bias`, and the number of columns in `densein.weight`,
    and must match the number of rows in the model's input data.

    `m` dictates the number of columns in `scale` and `mhoffset`.

    `c` dictates the number of rows in `densein.weight` and `densein.bias`, 
    the number of columns in `denseout.weight`, and consequently the number of dimensions
    in the hypersphere node.

    When only `n` is provided, a model without multi-hot parameters, and with a
    2-dimensional hypersphere node is initialized. To initialize a model without multi-hot
    parameters, but with a c-dimensional hypersphere node, provide `n`, `m=0`, and `c`.

    # Examples
    ```julia-repl
    julia> Random.seed!; n = 5; cyclops(n)
    cyclops(
      5x0 Matrix{Float32},                  # 0 parameters  (all zero)
      5x0 Matrix{Float32},                  # 0 parameters  (all zero)
      5x0 Matrix{Float32},                  # 0 parameters  (all zero)
      Dense(5 => 2),                        # 12 parameters
      Dense(2 => 5),                        # 15 parameters
    )                   # Total: 7 arrays, 27 parameters, 468 bytes.
    
    julia> Random.seed!; n = 5; m = 0; c = 4; cyclops(n, m, c)
    cyclops(
      5x0 Matrix{Float32},                  # 0 parameters  (all zero)
      5x0 Matrix{Float32},                  # 0 parameters  (all zero)
      5x0 Matrix{Float32},                  # 0 parameters  (all zero)
      Dense(5 => 4),                        # 24 parameters
      Dense(4 => 5),                        # 25 parameters
    )                   # Total: 7 arrays, 49 parameters, 556 bytes.

    julia> Random.seed!; n = 5; m = 3; cyclops(n, m)
    cyclops(
      5x3 Matrix{Float32},                  # 15 parameters
      5x3 Matrix{Float32},                  # 15 parameters
      5x1 Matrix{Float32},                  # 5 parameters
      Dense(5 => 2),                        # 12 parameters
      Dense(2 => 5),                        # 15 parameters
    )                   # Total: 7 arrays, 62 parameters, 640 bytes.
    ```

    # Errors
    Throws:
    - `CyclopsHypersphereDimensionError` when `c < 2`
    - `CyclopsInputHypersphereDimensionError` when `n ≤ c`
    - `CyclopsMultiHotDimensionError` when `m < 0`

    # See also
    [`CheckCyclopsInput`](@ref), [`CyclopsHypersphereDimensionError`](@ref),
    [`CyclopsInputHypersphereDimensionError`](@ref), [`CyclopsMultiHotDimensionError`](@ref)
    """
    function cyclops(n_eig::Int, n_multi::Int, n_circ::Int=2)        
        CheckCyclopsInput(n_eig, n_multi, n_circ)
        n_batch = n_multi == 0 ? 0 : 1

        return cyclops(rand(Float32, n_eig, n_multi), rand(Float32, n_eig, n_multi), rand(Float32, n_eig, n_batch), Dense(n_eig => n_circ), Dense(n_circ => n_eig))
    end

    function cyclops(n_eig::Int)
        CheckCyclopsInput(n_eig, 0, 2)

        return cyclops(rand(Float32, n_eig, 0), rand(Float32, n_eig, 0), rand(Float32, n_eig, 0), Dense(n_eig => 2), Dense(2 => n_eig))
    end

    """
        (m::cyclops)(x::Vector{Float32}, h::Vector{Int32}=missing[; silence::Bool=false])
        (m::cyclops)(x::Vector{Float32}, h::Missing[; silence::Bool=false])

    Returns reconstruction of input (`x`) after compression through an n-sphere node,
    with optional linear transformation according to multi-hot encoding (`h`).

        x₁ = x ○ (1 + m.scale • h) + m.mhoffset • h + m.offset
        x₂ = m.densein.weight • x₁ + m.densein.bias
        x₃ = x₂ / √∑x₂ᵢ²
        x₄ = m.denseout.weight * x₃ + m.denseout.bias
        x₅ = (x - m.mhoffset • h - m.offset) ÷ (1 + m.scale • h)

    When `h` is a `Vector{Int32}` the same length as `m`:
    1) `x` is transformed according to its multi-hot encoding using `mhe`,
    2) reduced from `n` to `c` dimensions using `m.densein`, 
    3) normalized to a point on the `c`-dimensional hypersphere using `hsn`, 
    4) expanded from `c` to `n` dimensions using `m.denseout`,
    5) and decoded accroding to `x`'s multi-hot encoding using `mhd`.

    When `h` is `missing` steps `1` and `5` are skipped and steps `2`-`4` are performed.

    # See also
    [`mhe`](@ref), [`mhd`](@ref), [`hsn`](@ref), [`nparams`](@ref), [`Flux.Dense`](@ref)

    # Example
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

    julia> x₂ = covariate_cyclops_model.densein.weight * x₁ .+ covariate_cyclops_model.densein.bias
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
    function (m::cyclops)(input_data::Vector{Float32}, multihot::Vector{Int32}=missing; silence::Bool=false)
        length(m.scale) == 0 && throw(ErrorException("Multi-hot encoding provided to model without multi-hot parameters."))
        multihot_encoding = mhe(input_data, multihot, m)
        dense_encoding = m.densein(multihot_encoding)
        circular_encoding = hsn(dense_encoding)
        dense_decoding = m.denseout(circular_encoding)
        output = mhd(dense_decoding, multihot, m)
        return output
    end

    function (m::cyclops)(input_data::Vector{Float32}, multihot::Missing; silence::Bool=false)
        silence || length(m.scale) == 0 || @warn "CYCLOPS model with multi-hot parameters used without multi-hot encoding."
        dense_encoding = m.densein(input_data)
        circular_encoding = hsn(dense_encoding)
        output = m.denseout(circular_encoding)
        return output
    end

    """
        CyclopsHyperSphereDomainError()

    An error when any of the inputs to `hsn` are `NaN`.

    # Examples
    ```julia-repl
    julia> hsn(Float32.([1, NaN]))
    ERROR: CyclopsHyperSphereDomainError: `NaN` at [2].
    [...]
    ```

    # See also
    [`CheckHSNdomain`](@ref), [`CyclopsHyperSphereDivideError`](@ref), [`cyclops`](@ref)
    """
    struct CyclopsHyperSphereDomainError <: Exception end
    CyclopsHyperSphereDomainError(x::Vector{Float32}) = error("CyclopsHyperSphereDomainError: `NaN` at $(findall(isnan.(x))).")

    """
        CyclopsHyperSphereDivideError()

    An error when all of the inputs to `hsn` are `0`.

    # Examples
    ```julia-repl
    julia> hsn(Float32.([0, 0]))
    ERROR: CyclopsHyperSphereDivideError: All values passed to `hsn` are `0`.
    [...]
    ```

    # See also
    [`CheckHSNdomain`](@ref), [`CyclopsHyperSphereDomainError`](@ref), [`cyclops`](@ref)
    """
    struct CyclopsHyperSphereDivideError <: Exception end
    Base.showerror(io::IO, e::CyclopsHyperSphereDivideError) =
        print(io, "CyclopsHyperSphereDivideError: All values passed to `hsn` are `0`.")

    """
        CheckHSNdomain(x::Vector{Float32})

    Checks domains of values in `x` and returns `nothing` when none of the values are `NaN` or not all of the values are `0`.

    # Errors
    - `CyclopsHyperSphereDomainError`: when any value in `x` is `NaN`
    - `CyclopsHyperSphereDivideError`: when all values in `x` are `0`

    # See also
    [`CyclopsHyperSphereDomainError`](@ref), [`CyclopsHyperSphereDivideError`](@ref),
    [`hsn`](@ref), [`cyclops`](@ref)

    # Examples
    ```julia-repl
    julia> CheckHSNdomain([1f0, 1f0])

    julia> CheckHSNdomain([1, NaN])
    ```
    """
    function CheckHSNdomain(x::Vector{Float32})
        any(isnan.(x)) && throw(CyclopsHyperSphereDomainError(x))
        all(x .== 0) && throw(CyclopsHyperSphereDivideError())

        return nothing
    end

    """
        hsn(x::Vector{Float32})

    Returns the element-wise quotient of `x` and its Euclidean norm.

        ‖x‖₂ = √∑(xᵢ²)
        x̂ → x / ‖x‖₂

    Output has the same dimensions as input.
    
    # Errors
    - Throws a `CyclopsHyperSphereDomainError` if any element of `x` is `NaN`.
    - Throws a `CyclopsHyperSphereDivideError` if all elements of `x` are `0`.

    # Examples
    ```julia-repl
    julia> atan(1, 1)*180/pi # Angle in degrees for the direction vector [1, 1]
    45.0

    julia> hsn_output = hsn(Float32.([1, 1])) # Direction vector normalized to unit vector
    2-element Vector{Float32}:
     0.70710677
     0.70710677

    julia> atan(hsn_output...)*180/pi # Angle of direction vector is retained
    45.0f0
    ```

    # See also 
    [`cyclops`](@ref), [`mhe`](@ref), [`mhd`](@ref), [`nparams`](@ref), [`Flux.Dense`](@ref),
    [`CYCLOPS.CyclopsHyperSphereDomainError`](@ref), 
    [`CYCLOPS.CyclopsHyperSphereDivideError`](@ref), 
    [`CYCLOPS.CheckHSNdomain`](@ref)
    """
    function hsn(x::Vector{Float32})
        CheckHSNdomain(x)
        return x./sqrt(sum(x .^ 2))
    end

    """
        CyclopsInputMultiHotDimensionMismatch(x::Vector{Float32}, m::Array{Float32})

    An error when `x` and `m` do not have the same number of rows.
    """
    struct CyclopsInputMultiHotDimensionMismatch <: Exception end
    function CyclopsInputMultiHotDimensionMismatch(x::Vector{Float32}, m::Array{Float32})
        error("CyclopsInputMultiHotDimensionMismatch: Input `x` and multi-hot parameters do not have the same number of rows.
\nInput = $(length(x)) ≠ $(size(m, 1)) = Multi-hot Parameters\n")
    end

    """
        CyclopsMultiHotParameterDimensionMismatch(h::Vector{Int}, m::Array{Float32})

    An error when the multi-hot encoding does not have as many rows as the multi-hot parameters has columns.
    """
    struct CyclopsMultiHotParameterDimensionMismatch <: Exception end
    function CyclopsMultiHotParameterDimensionMismatch(h::Vector{Int}, m::Array{Float32})
        error("CyclopsMultiHotParameterDimensionMismatch: Multi-hot encoding `h` and multi-hot parameters do not have fitting dimensions.
\nMulti-hot encoding must have as many rows as the multi-hot parameters have columns.\n
\nMulti-hot encoding = $(length(h)) ≠ $(size(m, 2)) = Multi-hot Parameters\n")
    end

    """
        CheckMultiHotTransformation(x::Vector{Float32}, h::Vector{Int}, m::Vector{Float32})

    Checks inputs to multi-hot transformation and returns `nothing` if input data `x` and multi-hot parameters `m` have the
    same number of rows, and if the multi-hot encoding `h` has as many rows as the multi-hot parameters `m` have columns.
    """
    function CheckMultiHotTransformation(x::Vector{Float32}, h::Vector{Int32}, m::Array{Float32})
        (length(x) != size(m, 1)) && throw(CyclopsInputMultiHotDimensionMismatch(x, m))
        (length(h) != size(m, 2)) && throw(CyclopsMultiHotParameterDimensionMismatch(h, m))

        return nothing
    end

    """
        mhe(x::Vector{Float32}, h::Vector{Int32}, m::cyclops)

    Returns `x` in 'multi-hot'-encoded space.

        x ○ (1 + m.scale • h) + m.mhoffset • h + m.offset

    Inverse of [`mhd`](@ref).

    # Operations
    - `○` is element-wise matrix multiplication
    - `•` is matrix multiplication

    # See also
    [`cyclops`](@ref), [`mhd`](@ref), [`hsn`](@ref), [`nparams`](@ref), [`Flux.Dense`](@ref)

    # Example
    ```julia-repl
    julia> using CYCLOPS, Random

    julia> Random.seed!(1234);

    julia> covariate_cyclops_model = cyclops(5,3)

    julia> Random.seed!(1234);

    julia> x = rand(Float32, 5)
    5-element Vector{Float32}:
     0.72619927
     0.32597667
     0.30699807
     0.5490511
     0.7889189

    julia> h = [1, 0, 1];

    julia> mhe_transform = mhe(x, h, covariate_cyclops_model)
    5-element Vector{Float32}:
     2.1732361
     2.2111228
     2.8991385
     3.2093358
     4.121205

    julia> mhd_recovery = mhd(mhe_transform, h, covariate_cyclops_model)
    5-element Vector{Float32}:
     0.7261993
     0.32597664
     0.30699813
     0.5490511
     0.7889189

    julia> isapprox(x, mhd_recovery, atol=1E-6)
    true
    ```
    """
    function mhe(input_data::Vector{Float32}, multihot::Vector{Int32}, m::cyclops)
        CheckMultiHotTransformation(input_data, multihot, m.scale)
        
        return input_data .* (1 .+ (m.scale * multihot)) .+ (m.mhoffset * multihot) .+ reshape(m.offset, length(input_data))
    end

    """
        mhd(x::Vector{Float32}, h::Vector{Int32}, m::cyclops)

    Restores `x` from 'multi-hot'-encoded space.

        (x - m.mhoffset • h - m.offset) ÷ (1 + m.scale • h)

    Inverse of [`mhe`](@ref).

    # Operations
    - `÷` is element-wise matrix division
    - `•` is matrix multiplication

    # See also
    [`cyclops`](@ref), [`mhe`](@ref), [`hsn`](@ref), [`nparams`](@ref), [`Flux.Dense`](@ref)

        # Example
    ```julia-repl
    julia> using CYCLOPS, Random

    julia> Random.seed!(1234);

    julia> covariate_cyclops_model = cyclops(5,3)

    julia> Random.seed!(1234);

    julia> x = rand(Float32, 5)
    5-element Vector{Float32}:
     0.72619927
     0.32597667
     0.30699807
     0.5490511
     0.7889189

    julia> h = [1, 0, 1];

    julia> mhe_transform = mhe(x, h, covariate_cyclops_model)
    5-element Vector{Float32}:
     2.1732361
     2.2111228
     2.8991385
     3.2093358
     4.121205

    julia> mhd_recovery = mhd(mhe_transform, h, covariate_cyclops_model)
    5-element Vector{Float32}:
     0.7261993
     0.32597664
     0.30699813
     0.5490511
     0.7889189

    julia> isapprox(x, mhd_recovery, atol=1E-6)
    true
    ```
    """
    function mhd(dense_decoding::Vector{Float32}, multihot::Vector{Int32}, m::cyclops)
        CheckMultiHotTransformation(dense_decoding, multihot, m.scale)

        return (dense_decoding .- (m.mhoffset * multihot) .- reshape(m.offset, length(dense_decoding))) ./ (1 .+ (m.scale * multihot))
    end

    Flux.@layer cyclops

    """
        nparams(m::cyclops)

    Returns the total number of parameters in a `cyclops` model.

    # See also
    [`cyclops`](@ref), [`Flux.params`](@ref)

    # Examples
    ```julia-repl
    julia> using CYCLOPS, Random

    julia> Random.seed!(1234);

    julia> covariate_cyclops_model = cyclops(5,3)

    julia> nparams(covariate_cyclops_model)
    62
    ```
    """
    function nparams(m::cyclops)
        return sum(length, Flux.params(m))
    end
end
