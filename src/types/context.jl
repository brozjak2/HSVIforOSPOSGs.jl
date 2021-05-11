mutable struct Context
    args::Args
    game::Game
    exploration_count::Int64
    exploration_depths::Vector{Int64}
    time_limit::Float64
    clock_start::Float64
end

function Context(args, game, time_limit)
    context = Context(args, game, 0, [], time_limit, time())

    check_neigh_param_d(context)

    if args.ub_value_method != :lp && args.ub_value_method != :nn
        throw(InvalidArgumentValue("ub_value_method", args.ub_value_method))
    end

    if args.stage_game_method != :lp && args.stage_game_method != :qre
        throw(InvalidArgumentValue("stage_game_method", args.stage_game_method))
    end

    return context
end

function Base.show(io::IO, context::Context)
    println(io, "Context:")
    println(io, context.args)
    println(io, context.game)
    println(io, "exploration_count = $(context.exploration_count)")
    println(io, "exploration_depths = $(context.exploration_depths)")
    println(io, @sprintf("time_limit %7.3fs", context.time_limit))
    println(io, @sprintf("running %7.3fs", time() - context.clock_start))
end
