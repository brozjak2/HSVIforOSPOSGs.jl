struct Context
    args::Args
    game::Game
    clock_start::Float64
end

Context(args, game) = Context(args, game, time())

function Base.show(io::IO, context::Context)
    println(io, "Context:")
    println(io, context.args)
    println(io, context.game)
    println(io, @sprintf("running %7.3fs", time() - context.clock_start))
end
