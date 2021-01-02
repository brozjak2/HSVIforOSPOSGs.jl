function load(gameFilePath::String)
    game, initPartition, initBelief = nothing, nothing, nothing

    open(gameFilePath, "r") do file
        parameters = split(readline(file), ' ')

        nStates = parse(Int64, parameters[1])
        nPartitions = parse(Int64, parameters[2])
        nLeaderActions = parse(Int64, parameters[3])
        nFollowerActions = parse(Int64, parameters[4])
        nObservations = parse(Int64, parameters[5])
        nTransitions = parse(Int64, parameters[6])
        nRewards = parse(Int64, parameters[7])
        disc = parse(Float64, parameters[8])

        states = Array{State,1}(undef, nStates)
        partitions = Array{Partition,1}(undef, nPartitions)
        leaderActions = Array{String,1}(undef, nLeaderActions)
        followerActions = Array{String,1}(undef, nFollowerActions)
        observations = Array{String,1}(undef, nObservations)
        transitions = Array{Tuple{Int64,Int64,Int64,Int64,Int64,Float64},1}(undef, nTransitions)
        rewards = Array{Tuple{Int64,Int64,Int64,Float64},1}(undef, nRewards)

        for i = 1:nPartitions
            partitions[i] = Partition(i)
        end

        for i = 1:nStates
            name = readuntil(file, ' ')
            partition = parse(Int64, readuntil(file, '\n')) + 1
            push!(partitions[partition].states, i)
            states[i] = State(i, length(partitions[partition].states), name, partition, Array{Int64,1}(undef, 0))
        end

        for i = 1:nLeaderActions
            leaderActions[i] = readline(file)
        end

        for i = 1:nFollowerActions
            followerActions[i] = readline(file)
        end

        for i = 1:nObservations
            observations[i] = readline(file)
        end

        for i = 1:nStates
            actions = [parse(Int64, x) + 1 for x in split(readline(file), ' ')]
            append!(states[i].followerActions, actions)
        end

        for i = 1:nPartitions
            actions = [parse(Int64, x) + 1 for x in split(readline(file), ' ')]
            append!(partitions[i].leaderActions, actions)
        end

        for i = 1:nTransitions
            parsedLine = split(readline(file), ' ')
            parsedInts = [parse(Int64, x) + 1 for x in parsedLine[1:5]]
            prob = parse(Float64, parsedLine[6])

            transitions[i] = (parsedInts..., prob)
        end

        for i = 1:nRewards
            parsedLine = split(readline(file), ' ')
            parsedInts = [parse(Int64, x) + 1 for x in parsedLine[1:3]]
            reward = parse(Float64, parsedLine[4])

            rewards[i] = (parsedInts..., reward)
        end

        initPartitionIndex = parse(Int64, readuntil(file, ' ')) + 1
        initPartition = partitions[initPartitionIndex]
        initBelief = [parse(Float64, x) for x in split(readline(file), ' ')]

        game = Game(nStates, nPartitions, nLeaderActions, nFollowerActions,
                    nObservations, nTransitions, nRewards, disc,
                    states, leaderActions, followerActions, observations,
                    transitions, rewards, partitions)
    end

    return game, initPartition, initBelief
end
