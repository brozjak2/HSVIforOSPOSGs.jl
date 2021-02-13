function load(gameFilePath::String)
    open(gameFilePath, "r") do file
        parameters = split(readline(file), ' ')
        game = Game(parameters)

        for i = 1:game.nPartitions
            game.partitions[i] = Partition(game, i)
        end

        for i = 1:game.nStates
            name = readuntil(file, ' ')
            partitionIndex = parse(Int64, readuntil(file, '\n')) + 1
            partition = game.partitions[partitionIndex]
            push!(partition.states, i)
            game.states[i] = State(game, i, length(partition.states),
                name, partition, partitionIndex, Array{Int64,1}(undef, 0))
        end

        for i = 1:game.nLeaderActions
            game.leaderActions[i] = readline(file)
        end

        for i = 1:game.nFollowerActions
            game.followerActions[i] = readline(file)
        end

        for i = 1:game.nObservations
            game.observations[i] = readline(file)
        end

        for i = 1:game.nStates
            actions = [parse(Int64, x) + 1 for x in split(readline(file), ' ')]
            append!(game.states[i].followerActions, actions)
        end

        for i = 1:game.nPartitions
            actions = [parse(Int64, x) + 1 for x in split(readline(file), ' ')]
            append!(game.partitions[i].leaderActions, actions)
        end

        for i = 1:game.nTransitions
            parsedLine = split(readline(file), ' ')
            parsedInts = [parse(Int64, x) + 1 for x in parsedLine[1:5]]
            prob = parse(Float64, parsedLine[6])

            game.transitions[i] = (parsedInts..., prob)
        end

        for i = 1:game.nRewards
            parsedLine = split(readline(file), ' ')
            parsedInts = [parse(Int64, x) + 1 for x in parsedLine[1:3]]
            reward = parse(Float64, parsedLine[4])

            game.rewards[i] = (parsedInts..., reward)
        end

        initPartitionIndex = parse(Int64, readuntil(file, ' ')) + 1
        initPartition = game.partitions[initPartitionIndex]
        initBelief = [parse(Float64, x) for x in split(readline(file), ' ')]

        prepare(game)

        return game, initPartition, initBelief
    end
end
