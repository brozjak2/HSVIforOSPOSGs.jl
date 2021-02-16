struct Params
    epsilon::Float64
    neigh_param_d::Float64
    presolve_min_delta::Float64
    presolve_time_limit::Float64
end

struct Context
    params::Params
    game::Game
    clock_start::Float64
end

Context(params::Params, game::Game) = Context(params, game, time())
