struct Params
    epsilon::Float64
    neigh_param_d::Float64
    presolve_min_delta::Float64
    presolve_time_limit::Float64
end

Base.show(io::IO, params::Params) = show_struct(io, params)

struct Context
    params::Params
    game::Game
    clock_start::Float64
end

function Base.show(io::IO, context::Context)
    println(io, "Context:")
    println(io, " $(context.params)")
    println(io, " $(context.game)")
    println(io, @sprintf(" running  %7.3fs", time() - context.clock_start))
end
