function dim_tuple(n::T) where T <: Integer
    nsqrt = sqrt(n)
    ndim = Int(ceil(nsqrt))
    dim2 = (ndim * (ndim-1)) ≥ n ? ndim-1 : ndim
    return (ndim, dim2)
end

function make_coords(pnrows::T, pncols::T; 
        n=false, flip=false
    ) where T <: Integer
    n = n isa Bool ? pnrows*pncols : n isa Int ? n : throw(ArgumentError("`n` must be an integer or a boolean."))
    if flip 
        return [Tuple(x) for x in eachrow(hcat(repeat(1:pnrows, outer=pncols), repeat(1:pncols, inner=pnrows)))][1:n]
    end
    return [Tuple(x) for x in eachrow(hcat(repeat(1:pnrows, inner=pncols), repeat(1:pncols, outer=pnrows)))][1:n]
end

const XTICK_POSITIONS = (0:pi/2:2pi)
const XTICK_LABELS = [
    L"0",
    L"\frac{\pi}{2}",
    L"\pi",
    L"\frac{3\pi}{2}",
    L"2\pi"
]

Base.@kwdef struct AxisStyle
    xticks = (XTICK_POSITIONS, XTICK_LABELS)
    yticks = LinearTicks(10)
    xticklabelsize::Int = 10
    yticklabelsize::Int = 10
    titlegap::Int = 2
    topspinevisible::Bool = false
    rightspinevisible::Bool = false
    bottomspinevisible::Bool = false
    leftspinevisible::Bool = false
    xlabel::String = ""
    ylabel::String = ""
end

function axis_from_style(parent; style::AxisStyle = AxisStyle(), axis_kwargs...)
    axis_kwargs = (; axis_kwargs...)
    fnames = fieldnames(AxisStyle)
    fvals = getproperty.(Ref(style), fnames)
    default_kwargs = NamedTuple{fnames}(fvals)
    all_kwargs = merge(default_kwargs, axis_kwargs)

    return Axis(parent; all_kwargs...)
end

function axis_from_style(parent, coords::V; 
        style::AxisStyle = AxisStyle(), 
        axis_kwargs...
    ) where V <: AbstractVector{T} where T <: Tuple{S, S} where S <: Integer

    return [axis_from_style(parent[i, j]; style=style, axis_kwargs...) for (i, j) in coords]
end

function plotData(d::AbstractMatrix{T}, x::AbstractVector{T};
        fig_size::Tuple{Int64, Int64} = (1200, 1000),
        style::AxisStyle = AxisStyle(),
        yticks = LinearTicks(5),
        axis_kwargs...
    ) where T <: AbstractFloat
    
    out = Figure(size = fig_size);
    
    n = size(d, 1)
    pnrows, pncols = dim_tuple(n)
    
    coords = make_coords(pnrows, pncols; n=n)
    # ax = [axis_from_style(out[i, j], axis_kwargs...) for (i, j) in coords]
    ax = axis_from_style(out, coords; style=style, yticks=yticks, axis_kwargs...)
    
    for k in 1:n
        a = ax[k]
        scatter!(a, x, d[k, :])
        a.title = "Dimension $k"
        a.ylabel = "Expression"
        a.xlabel = "radian"
    end

    return out
end

function plotData(d::AbstractMatrix{T};
        i::AbstractVector{<:AbstractFloat},
        f::AbstractVector{<:AbstractFloat},
        fig_size::Tuple{Int64, Int64} = (1200, 1000),
        style::AxisStyle = AxisStyle(),
        yticks = LinearTicks(5),
        axis_kwargs...
    ) where T <: AbstractFloat
    
    out = Figure(size = fig_size);
    
    n = size(d, 1)
    pnrows, pncols = (n, 2)
    
    coords = make_coords(pnrows, pncols)
    # ax = [axis_from_style(out[i, j], axis_kwargs...) for (i, j) in coords]
    ax = axis_from_style(out, coords; style=style, yticks=yticks, axis_kwargs...)
    
    for k in 1:n
        a = ax[2*(k-1)+1]
        scatter!(a, i, d[k, :])
        a.title = "Dimension $k"
        a.ylabel = "Expression"
        a.xlabel = "Preidcted radian"

        a = ax[2*k]
        scatter!(a, f, d[k, :])
        a.title = "Dimension $k"
        a.ylabel = "Expression"
        a.xlabel = "Preidcted radian"
    end

    return out
end

function plotTransformation(x::AbstractVector{T};
        i::AbstractMatrix{T}, 
        f::AbstractMatrix{T},
        fig_size::Tuple{Int64, Int64} = (1200, 1000),
        style::AxisStyle = AxisStyle(),
        yticks = LinearTicks(5),
        axis_kwargs...
    ) where T <: AbstractFloat
    
    out = Figure(size = fig_size);

    n = size(i, 1)
    coords = make_coords(n, 2)
    ax = axis_from_style(out, coords; style=style, yticks=yticks, axis_kwargs...)

    for k in 1:n
        a = ax[2*(k-1)+1]
        scatter!(a, x, i[k, :])
        a.title = "Initial Transformation $k"
        a.ylabel = "Transformation"
        a.xlabel = "radian"
        a = ax[2*k]
        scatter!(a, x, f[k, :])
        a.title = "Final Transformation $k"
        a.ylabel = "Transformation"
        a.xlabel = "radian"
    end

    return out
end

function plotTransformation(y::AbstractMatrix{<:AbstractFloat};
        i::AbstractMatrix{T}, 
        f::AbstractMatrix{T},
        fig_size::Tuple{Int64, Int64} = (1200, 1000),
        style::AxisStyle = AxisStyle(),
        xticks = LinearTicks(5),
        yticks = LinearTicks(5),
        axis_kwargs...
    ) where T <: AbstractFloat
    
    out = Figure(size = fig_size);

    n = size(i, 1)
    coords = make_coords(n, 2)
    ax = axis_from_style(out, coords; style=style, xticks=xticks, yticks=yticks, axis_kwargs...)

    for k in 1:n
        a = ax[2*(k-1)+1]
        scatter!(a, y[k, :], i[k, :])
        a.title = "Initial Transformation $k"
        a.ylabel = "Transformation"
        a.xlabel = "radian"
        a = ax[2*k]
        scatter!(a, y[k, :], f[k, :])
        a.title = "Final Transformation $k"
        a.ylabel = "Transformation"
        a.xlabel = "radian"
    end

    return out
end

function plotProjection(;
        i::AbstractMatrix{T},
        f::AbstractMatrix{T},
        fig_size::Tuple{Int64, Int64} = (1200, 1000),
        style::AxisStyle = AxisStyle(),
        xticks = LinearTicks(10),
        axis_kwargs...
    ) where T <: AbstractFloat

    out = Figure(size = fig_size);

    coords = make_coords(1, 2)
    ax = axis_from_style(out, coords; style=style, xticks=xticks, axis_kwargs...)

    a = ax[1]
    scatter!(a, i[1, :], i[2, :])
    a.title = "Initial Projection"
    a.ylabel = "y"
    a.xlabel = "x"
    
    a = ax[2]
    scatter!(a, f[1, :], f[2, :])
    a.title = "Final Projection"
    a.ylabel = "y"
    a.xlabel = "x"

    return out
end

function plotAngle(x::AbstractVector{T};
        i::AbstractVector{T},
        f::AbstractVector{T},
        fig_size::Tuple{Int64, Int64} = (1200, 1000),
        style::AxisStyle = AxisStyle(),
        yticks = (XTICK_POSITIONS, XTICK_LABELS),
        axis_kwargs...
    ) where T <: AbstractFloat

    out = Figure(size = fig_size);

    # coords = [Tuple(x) for x in eachrow(hcat(repeat(1:1, inner=2), repeat(1:2, outer=1)))]
    coords = make_coords(1, 2)
    # ax = ax[axis_from_style(out[ii, jj], axis_kwargs...) for (ii, jj) in coords]
    ax = axis_from_style(out, coords; style=style, yticks=yticks, axis_kwargs...)

    a = ax[1]
    scatter!(a, x, i)
    a.title = "Initial Angles"
    a.ylabel = "Prediction"
    a.xlabel = "radian"

    a = ax[2]
    scatter!(a, x, f)
    a.title = "Final Angles"
    a.ylabel = "Prediction"
    a.xlabel = "radian"

    return out
end

function plotLoss(l::AbstractVector{T};
        fig_size::Tuple{Int64, Int64} = (1200, 1000),
        style::AxisStyle = AxisStyle(),
        xticks = LinearTicks(10),
        axis_kwargs...
    ) where T <: AbstractFloat

    n = length(l)
    
    out = Figure(size = fig_size);

    coords = make_coords(1, 1)
    ax = axis_from_style(out, coords; style=style, xticks=xticks, axis_kwargs...)

    a = ax[1]

    loss_curve = log10.(l)
    grad_curve = -[0, log10.(abs.(diff(l)))...]

    lines!(a, 1:n, loss_curve)
    lines!(a, 1:n, grad_curve)

    text!(a, n, loss_curve[end];
        text = "loss",
        align = (:left, :center),
        offset = (5,0)
    )

    text!(a, n, grad_curve[end];
        text = "Δloss",
        align = (:left, :center),
        offset = (5,0)
    )
    
    a.title = "Training Performance"
    a.ylabel = "log 10"
    a.xlabel = "epoch"

    return out
end

function plot(
        x::AbstractVector{<:Real}, 
        y::AbstractMatrix{<:AbstractFloat}, 
        o::NamedTuple
    )
    @assert all(∈(keys(o)), (:p, :c, :t, :l)) "o must have keys :p, :c, :t, :l"

    dplt = plotData(y, x);
    tplt = plotTransformation(x; o.t...);
    pplt = plotProjection(;o.c...);
    aplt = plotAngle(x; o.p...);
    lplt = plotLoss(o.l);

    out = (d = dplt, t = tplt, p = pplt, a = aplt, l = lplt)

    return out
end

function plot(
        y::AbstractMatrix{<:AbstractFloat},
        o::NamedTuple
    )
    @assert all(∈(keys(o)), (:p, :c, :t, :l)) "o must have keys :p, :c, :t, :l"

    dplt = plotData(y; o.p...)
    tplt = plotTransformation(y; o.t...);
    pplt = plotProjection(;o.c...);
    lplt = plotLoss(o.l);

    out = (d = dplt, t = tplt, p = pplt, l = lplt)

    return out
end
