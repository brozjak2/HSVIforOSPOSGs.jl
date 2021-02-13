function load(game_file_path::String)
    open(game_file_path, "r") do file
        parameters = split(readline(file), ' ')
        game = Game(parameters)

        for i = 1:game.partition_count
            game.partitions[i] = Partition(game, i)
        end

        for i = 1:game.state_count
            name = readuntil(file, ' ')
            partition_index = parse(Int64, readuntil(file, '\n')) + 1
            partition = game.partitions[partition_index]
            push!(partition.states, i)
            game.states[i] = State(game, i, length(partition.states),
                name, partition, partition_index, Array{Int64,1}(undef, 0))
        end

        for i = 1:game.leader_action_count
            game.leader_actions[i] = readline(file)
        end

        for i = 1:game.follower_action_count
            game.follower_actions[i] = readline(file)
        end

        for i = 1:game.observation_count
            game.observations[i] = readline(file)
        end

        for i = 1:game.state_count
            actions = [parse(Int64, x) + 1 for x in split(readline(file), ' ')]
            append!(game.states[i].follower_actions, actions)
        end

        for i = 1:game.partition_count
            actions = [parse(Int64, x) + 1 for x in split(readline(file), ' ')]
            append!(game.partitions[i].leader_actions, actions)
        end

        for i = 1:game.transition_count
            parsedLine = split(readline(file), ' ')
            parsedInts = [parse(Int64, x) + 1 for x in parsedLine[1:5]]
            prob = parse(Float64, parsedLine[6])

            game.transitions[i] = (parsedInts..., prob)
        end

        for i = 1:game.reward_count
            parsedLine = split(readline(file), ' ')
            parsedInts = [parse(Int64, x) + 1 for x in parsedLine[1:3]]
            reward = parse(Float64, parsedLine[4])

            game.rewards[i] = (parsedInts..., reward)
        end

        init_partition_index = parse(Int64, readuntil(file, ' ')) + 1
        init_partition = game.partitions[init_partition_index]
        init_belief = [parse(Float64, x) for x in split(readline(file), ' ')]

        return game, init_partition, init_belief
    end
end
