struct Context
    args::Args
    game::Game
    clock_start::Float64
    gurobi_env::Union{Nothing,Gurobi.Env}
end

function Context(args, game)
    if args.lp_solver == :gurobi
        gurobi_env = Gurobi.Env()
    else
        gurobi_env = nothing
    end

    return Context(args, game, time(), gurobi_env)
end

function Base.show(io::IO, context::Context)
    println(io, "Context:")
    println(io, context.args)
    println(io, context.game)
    println(io, @sprintf("running %7.3fs", time() - context.clock_start))
end
