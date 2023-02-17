mutable struct Context
    args::Args
    game::Game

    exploration_count::Int64
    timestamps::Vector{Float64}
    lb_values::Vector{Float64}
    ub_values::Vector{Float64}
    gaps::Vector{Float64}
    gamma_sizes::Vector{Int64}
    upsilon_sizes::Vector{Int64}
    exploration_depths::Vector{Int64}

    time_limit::Float64
    clock_start::Float64
end

function Context(args, game, time_limit)
    context = Context(args, game, 0, [], [], [], [], [], [], [], time_limit, time())

    check_neigh_param_d(context)

    log_initial(context)

    return context
end

function Base.show(io::IO, context::Context)
    println(io, "Context:")
    println(io, context.args)
    println(io, context.game)
    println(io, "exploration_count = $(context.exploration_count)")
    println(io, "timestamps = $(context.timestamps)")
    println(io, "lb_values = $(context.lb_values)")
    println(io, "ub_values = $(context.ub_values)")
    println(io, "gaps = $(context.gaps)")
    println(io, "gamma_sizes = $(context.gamma_sizes)")
    println(io, "upsilon_sizes = $(context.upsilon_sizes)")
    println(io, "exploration_depths = $(context.exploration_depths)")
    println(io, @sprintf("time_limit %7.3fs", context.time_limit))
    println(io, @sprintf("running %7.3fs", time() - context.clock_start))
end
