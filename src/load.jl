function load(args)
    @unpack game_file_path, normalize_rewards = args

    parsed_game_definition = open(game_file_path) do file
        game_params_string = readline(file)
        game_params = GameParams(game_params_string)

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
        if normalize_rewards
            min_r = minimum([r.r for r in rewards])
            max_r = maximum([r.r for r in rewards])
            normalize(r) = (r - min_r) / (max_r - min_r)
            rewards = [Reward(r.s, r.a1, r.a2, normalize(r.r)) for r in rewards]
        end

        init_partition_index = parse(Int64, readuntil(file, ' ')) + 1
        init_belief = [parse(Float64, x) for x in split(readline(file), ' ')]

        return ParsedGameDefinition(
            game_params, states_names, states_partitions, leader_actions_names,
            follower_actions_names, observations_names, follower_actions, leader_actions,
            transitions, rewards, init_partition_index, init_belief
        )
    end

    return Game(parsed_game_definition, args)
end

function parse_states(file, count)
    states_names = Vector{String}(undef, count)
    states_partitions = Vector{Int64}(undef, count)

    for i in 1:count
        states_names[i] = readuntil(file, ' ')
        states_partitions[i] = parse(Int64, readuntil(file, '\n')) + 1
    end

    return states_names, states_partitions
end

parse_names(file, count) = [readline(file) for i in 1:count]

parse_actions(file, count) = [parse_ints_plus_one(readline(file)) for i in 1:count]
parse_ints_plus_one(line) = [parse(Int64, x) + 1 for x in split(line, ' ')]

parse_transitions(file, count) = [Transition(readline(file)) for i in 1:count]
parse_rewards(file, count) = [Reward(readline(file)) for i in 1:count]
