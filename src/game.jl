mutable struct Game <: AbstractGame
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

function Game(parameters)
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

    return Game(nStates, nPartitions, nLeaderActions, nFollowerActions,
        nObservations, nTransitions, nRewards, disc,
        states, leaderActions, followerActions, observations,
        transitions, rewards, partitions)
end

function Game(nStates, nPartitions, nLeaderActions, nFollowerActions,
    nObservations, nTransitions, nRewards, disc,
    states, leaderActions, followerActions, observations,
    transitions, rewards, partitions)

    transitionMap = Dict{Tuple{Int64,Int64,Int64,Int64,Int64},Float64}([])

    game = Game(nStates, nPartitions, nLeaderActions, nFollowerActions,
        nObservations, nTransitions, nRewards, disc,
        states, leaderActions, followerActions, observations,
        transitions, rewards, partitions,
        0, 0, transitionMap)

    return game
end

function prepare(game::Game)
    game.minReward = minimum([r[4] for r in game.rewards])
    game.maxReward = maximum([r[4] for r in game.rewards])

    for transition in game.transitions
        game.transitionMap[transition[1:5]] = transition[6]
    end

    for transition in game.transitions
        s, a1, a2, o, sp, prob = transition
        partition = game.states[s].partition

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

        targetPartition = game.states[sp].partitionIndex
        if !haskey(partition.partitionTransitions, (a1, o))
            partition.partitionTransitions[(a1, o)] = targetPartition
        else
            @assert partition.partitionTransitions[(a1, o)] == targetPartition "Multipartition transition"
        end
    end

    for reward in game.rewards
        s, a1, a2, r = reward
        partition = game.states[s].partition

        partition.rewards[(s, a1, a2)] = r
    end

    for partition in game.partitions
        prepare(partition, game)
    end
end

Lmin(game::Game) = game.minReward / (1 - game.disc)

Umax(game::Game) = game.maxReward / (1 - game.disc)

lipschitzdelta(game::Game) = (Umax(game) - Lmin(game)) / 2
