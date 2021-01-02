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

    for partition in game.partitions
        prepare(partition, game)
    end

    return game
end

Lmin(game::Game) = game.minReward / (1 - game.disc)
Umax(game::Game) = game.maxReward / (1 - game.disc)
lipschitzdelta(game::Game) = (Umax(game) - Lmin(game)) / 2

function initBounds(game::Game)
    for partition in game.partitions
        initBounds(partition)
    end
end
