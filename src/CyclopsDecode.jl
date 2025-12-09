function project(input_data::AbstractVector{T}, multihot::AbstractVector{<:Integer}, m::cyclops)::AbstractVector{T} where T <: AbstractFloat
    CheckCyclopsInput(input_data, multihot, m.scale)
    multihot_encoding = mhe(input_data, multihot, m, skip_check = true)
    dense_encoding = m.densein(multihot_encoding)
    return dense_encoding
end

function project(input_data::AbstractVector{T}, m::cyclops)::AbstractVector{T} where T <: AbstractFloat
    CheckCyclopsInput(input_data, m.scale)
    dense_encoding = m.densein(input_data)
    return dense_encoding
end

function project(input_data::AbstractMatrix{T}, multihot::AbstractMatrix{<:Integer}, m::cyclops)::AbstractMatrix{T} where T <: AbstractFloat
    return hcat(project(view(input_data, :, jj), view(multihot, :, jj), m) for jj in axes(input_data, 2)...)
end

function project(input_data::AbstractMatrix{T}, m::cyclops)::AbstractMatrix{T} where T <: AbstractFloat
    return hcat([project(jj, m) for jj in eachcol(input_data)]...)
end

function decode(input_data::AbstractVector{T}, multihot::AbstractVector{<:Integer}, m::cyclops)::T where T <: AbstractFloat
    dense_encoding = project(input_data, multihot, m)
    return mod(atan(dense_encoding...), 2*pi)
end

function decode(input_data::AbstractVector{T}, m::cyclops)::T where T <: AbstractFloat
    dense_encoding = project(input_data, m)
    return mod(atan(dense_encoding...), 2*pi)
end

function decode(input_data::AbstractMatrix{T}, multihot::AbstractMatrix{<:Integer}, m::cyclops)::AbstractVector{T} where T <: AbstractFloat
    return vcat(decode(view(input_data, :, jj), view(multihot, :, jj), m) for jj in axes(input_data, 2)...)
end

function decode(input_data::AbstractMatrix{T}, m::cyclops)::AbstractVector{T} where T <: AbstractFloat
    return [decode(jj, m) for jj in eachcol(input_data)]
end
