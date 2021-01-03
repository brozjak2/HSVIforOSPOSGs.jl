struct Game
    nStates::Int64
    nPartitions::Int64
    nLeaderActions::Int64
    nFollowerActions::Int64
    nObservations::Int64
    nTransitions::Int64
    nRewards::Int64

    disc::Float64

    states::Array{State,1}
    leaderActions::Array{String,1}
    followerActions::Array{String,1}
    observations::Array{String,1}
    transitions::Array{Tuple{Int64,Int64,Int64,Int64,Int64,Float64},1}
    rewards::Array{Tuple{Int64,Int64,Int64,Float64},1}
    partitions::Array{Partition,1}

    minReward::Float64
    maxReward::Float64

    transitionMap::Dict{Tuple{Int64,Int64,Int64,Int64,Int64},Float64}
end

function Game(nStates, nPartitions, nLeaderActions, nFollowerActions,
    nObservations, nTransitions, nRewards, disc,
    states, leaderActions, followerActions, observations,
    transitions, rewards, partitions)

    minReward = minimum([r[4] for r in rewards])
    maxReward = maximum([r[4] for r in rewards])

    transitionMap = Dict{Tuple{Int64,Int64,Int64,Int64,Int64},Float64}([])
    for transition in transitions
        transitionMap[transition[1:5]] = transition[6]
    end

    game = Game(nStates, nPartitions, nLeaderActions, nFollowerActions,
        nObservations, nTransitions, nRewards, disc,
        states, leaderActions, followerActions, observations,
        transitions, rewards, partitions,
        minReward, maxReward, transitionMap)

    prepare(game)

    return game
end

Lmin(game::Game) = game.minReward / (1 - game.disc)
Umax(game::Game) = game.maxReward / (1 - game.disc)
lipschitzdelta(game::Game) = (Umax(game) - Lmin(game)) / 2

function prepare(game::Game)
    for transition in game.transitions
        s, a1, a2, o, sp, prob = transition
        partition = game.partitions[game.states[s].partition]

        if haskey(partition.aoTransitions, (a1, o))
            push!(partition.aoTransitions[(a1, o)], transition)
        else
            partition.aoTransitions[(a1, o)] = [transition]
        end

        if haskey(partition.observations, a1)
            push!(partition.observations[a1], o)
        else
            partition.observations[a1] = [o]
        end

        if haskey(partition.transitions, s)
            push!(partition.transitions[s], transition)
        else
            partition.transitions[s] = [transition]
        end

        targetPartition = game.states[sp].partition
        if !haskey(partition.partitionTransitions, (a1, o))
            partition.partitionTransitions[(a1, o)] = targetPartition
        else
            @assert partition.partitionTransitions[(a1, o)] == targetPartition "Multipartition transition"
        end
    end

    for reward in game.rewards
        s, a1, a2, r = reward
        partition = game.partitions[game.states[s].partition]

        partition.rewards[(s, a1, a2)] = r
    end

    for partition in game.partitions
        prepare(partition, game)
    end
end
