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

GameParams(params_string::String) = struct_from_ints_and_floats_string(params_string, GameParams, 8)
