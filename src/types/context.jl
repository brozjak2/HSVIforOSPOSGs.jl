mutable struct Context
    args::Args
    game::Game
    exploration_count::Int64
    exploration_depths::Vector{Int64}
    clock_start::Float64
end

Context(args, game) = Context(args, game, 0, [], time())

function Base.show(io::IO, context::Context)
    println(io, "Context:")
    println(io, context.args)
    println(io, context.game)
    println(io, "exploration_count = $(context.exploration_count)")
    println(io, "exploration_depths = $(context.exploration_depths)")
    println(io, @sprintf("running %7.3fs", time() - context.clock_start))
end
