struct Context
    params::Params
    game::Game
    clock_start::Float64
end

function Base.show(io::IO, context::Context)
    println(io, "Context:")
    println(io, "  $(context.params)")
    println(io, "  $(context.game)")
    println(io, @sprintf("  running %7.3fs", time() - context.clock_start))
end
