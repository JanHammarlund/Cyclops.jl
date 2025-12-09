function mse(m::cyclops, x::AbstractVecOrMat{T}) where T <: AbstractFloat
    return Flux.mse(m(x), x)
end

function train!(m::cyclops, d::AbstractArray{T1};
    l::Function     = mse,
    epochs::T2      = 100,
    steps::T2       = 100,
    batchsize::T2   = 1,
    shuffled::Bool  = true,
    optimiser::T3   = Adam()
    ) where {T1<:AbstractFloat, T2<:Integer, T3<:AbstractRule}

    epoch_default = 100
    step_default = 100

    es = [epochs != epoch_default, steps != step_default]

    epochs = any(es) ? [epochs, steps][findfirst(es)] : epochs

    initial_phases = decode(d, m)
    initial_projections = project(d, m)
    initial_transformation = m(d)
    
    training_data = Flux.DataLoader((d,), batchsize=batchsize, shuffle=shuffled)
    optimiser_rule = Flux.setup(optimiser, m)

    mean_sample_loss = [mean([l(m, di) for di in eachcol(d)])]

    @showprogress for _ in 1:epochs
        Flux.train!(l, m, training_data, optimiser_rule)
        push!(mean_sample_loss, mean([l(m, di) for di in eachcol(d)]))
    end

    final_phases = decode(d, m)
    final_projections = project(d, m)
    final_transformation = m(d)

    phase_output = (i = initial_phases, f = final_phases)
    projection_output = (i = initial_projections, f = final_projections)
    transformation_output = (i = initial_transformation, f = final_transformation)

    output = (
        p = phase_output,
        c = projection_output,
        t = transformation_output,
        l = mean_sample_loss
    )

    return output
end