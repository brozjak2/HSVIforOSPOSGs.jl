struct ParsedGameDefinition
    game_params::GameParams

    states_names::Vector{String}
    states_partitions::Vector{Int64}

    leader_actions_names::Vector{String}
    follower_actions_names::Vector{String}
    observations_names::Vector{String}

    follower_actions::Vector{Vector{Int64}}
    leader_actions::Vector{Vector{Int64}}

    transitions::Vector{Transition}
    rewards::Vector{Reward}

    init_partition_index::Int64
    init_belief::Vector{Float64}
end

function ParsedGameDefinition(file::IO)
    game_params = GameParams(readline(file))
    @unpack state_count, partition_count, leader_action_count, follower_action_count,
        observation_count, transition_count, reward_count = game_params

    states_names, states_partitions = parse_states(file, state_count)

    leader_actions_names = parse_names(file, leader_action_count)
    follower_actions_names = parse_names(file, follower_action_count)
    observations_names = parse_names(file, observation_count)

    follower_actions = parse_actions(file, state_count)
    leader_actions = parse_actions(file, partition_count)

    transitions = parse_transitions(file, transition_count)

    rewards = parse_rewards(file, reward_count)

    init_partition_index = parse(Int64, readuntil(file, ' ')) + 1
    init_belief = [parse(Float64, x) for x in split(readline(file), ' ')]

    return ParsedGameDefinition(game_params, states_names, states_partitions,
        leader_actions_names, follower_actions_names, observations_names,
        follower_actions, leader_actions, transitions, rewards, init_partition_index,
        init_belief)
end

function parse_states(file::IO, count::Int64)
    states_names = Vector{String}(undef, count)
    states_partitions = Vector{Int64}(undef, count)

    for i = 1:count
        states_names[i] = readuntil(file, ' ')
        states_partitions[i] = parse(Int64, readuntil(file, '\n')) + 1
    end

    return states_names, states_partitions
end

parse_names(file::IO, count::Int64) = [readline(file) for i = 1:count]

parse_actions(file::IO, count::Int64) = [parse_ints_plus_one(readline(file)) for i = 1:count]
parse_ints_plus_one(line::String) = [parse(Int64, x) + 1 for x in split(line, ' ')]

parse_transitions(file::IO, count::Int64) = [Transition(readline(file)) for i = 1:count]
parse_rewards(file::IO, count::Int64) = [Reward(readline(file)) for i = 1:count]
