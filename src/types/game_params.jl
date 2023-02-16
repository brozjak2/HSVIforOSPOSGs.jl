struct GameParams
    state_count::Int64
    partition_count::Int64
    leader_action_count::Int64
    follower_action_count::Int64
    observation_count::Int64
    transition_count::Int64
    reward_count::Int64
    discount_factor::Float64
end

function GameParams(params_string)
    params_strings = split(params_string, ' ')
    int_params = map((x) -> parse(Int64, x), params_strings[1:7])
    discount_factor = parse(Float64, params_strings[8])

    return GameParams(int_params..., discount_factor)
end

function Base.show(io::IO, game_params::GameParams)
    println(io, "GameParams:")
    for field in fieldnames(GameParams)
        println(io, "$field = $(getfield(game_params, field))")
    end
end
