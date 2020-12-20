function load(gameFilePath::String)
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
            partitions[i] = Partition(
                i,
                Array{Int64,1}(undef, 0),
                Array{Int64,1}(undef, 0),
                Dict{Int64,Array{Int64,1}}([]))
            end

        for i = 1:nStates
            name = readuntil(file, ' ')
            partition = parse(Int64, readuntil(file, '\n')) + 1
            states[i] = State(i, name, partition, Array{Int64,1}(undef, 0))
            append!(partitions[partition].states, [i])
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
            parsedInts = [parse(Int64, x) + 1 for x in parsedLine[1:end-1]]
            prob = parse(Float64, parsedLine[end])

            transitions[i] = (parsedInts..., prob)
        end

        for i = 1:nRewards
            parsedLine = split(readline(file), ' ')
            parsedInts = [parse(Int64, x) + 1 for x in parsedLine[1:end-1]]
            reward = parse(Float64, parsedLine[end])

            rewards[i] = (parsedInts..., reward)
        end

        initPartition = parse(Int64, readuntil(file, ' ')) + 1
        initBelief = [parse(Float64, x) for x in split(readline(file), ' ')]

        # println("nStates: $nStates")
        # println("nPartitions: $nPartitions")
        # println("nLeaderActions: $nLeaderActions")
        # println("nFollowerActions: $nFollowerActions")
        # println("nObservations: $nObservations")
        # println("nTransitions: $nTransitions")
        # println("nRewards: $nRewards")
        # println("disc: $disc")
        # println("---------- States ----------")
        #
        # for i = 1:nStates
        #     println("$(states[i].index): State $(states[i].name) belongs to partition num. $(states[i].partition)")
        #     for j = 1:length(states[i].followerActions)
        #         a = states[i].followerActions[j]
        #         println("\t$a: Action $(followerActions[a])")
        #     end
        # end
        #
        # println("---------- Partitions ----------")
        #
        # for i = 1:nPartitions
        #     println("$(partitions[i].index): Partition")
        #     println("\tStates:")
        #     for j = 1:length(partitions[i].states)
        #         s = partitions[i].states[j]
        #         println("\t\t$s: State $(states[s].name)")
        #     end
        #     println("\tActions:")
        #     for j = 1:length(partitions[i].leaderActions)
        #         a = partitions[i].leaderActions[j]
        #         println("\t\t$a: Action $(leaderActions[a])")
        #     end
        # end
        #
        # println("---------- Leader actions ----------")
        #
        # for i = 1:nLeaderActions
        #     println("$i: Leader action $(leaderActions[i])")
        # end
        #
        # println("---------- Follower actions ----------")
        #
        # for i = 1:nFollowerActions
        #     println("$i: Follower action $(followerActions[i])")
        # end
        #
        # println("---------- Observations ----------")
        #
        # for i = 1:nObservations
        #     println("$i: Observation $(observations[i])")
        # end
        #
        # println("---------- Transitions ----------")
        #
        # for i = 1:nTransitions
        #     println(@sprintf("Trasition from %d to %d by %d and %d observing %d with probability %.2f",
        #         transitions[i][1],
        #         transitions[i][5],
        #         transitions[i][2],
        #         transitions[i][3],
        #         transitions[i][4],
        #         transitions[i][6]))
        # end
        #
        # println("---------- Rewards ----------")
        #
        # for i = 1:nRewards
        #     println(@sprintf("Reward for %d and %d in %d is %.2f",
        #         rewards[i][2],
        #         rewards[i][3],
        #         rewards[i][1],
        #         rewards[i][4]))
        # end
        #
        # println("---------- Init ----------")
        #
        # println("Starting in partition $initPartition with belief:")
        # for i = 1:length(initBelief)
        #     println("\t$(partitions[initPartition].states[i]): $(initBelief[i])")
        # end
    end
end
