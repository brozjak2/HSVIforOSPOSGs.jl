mutable struct Game <: AbstractGame
    state_count::Int64
    partition_count::Int64
    leader_action_count::Int64
    follower_action_count::Int64
    observation_count::Int64
    transition_count::Int64
    reward_count::Int64

    disc::Float64

    states::Array{State,1}
    leader_actions::Array{String,1}
    follower_actions::Array{String,1}
    observations::Array{String,1}
    transitions::Array{Tuple{Int64,Int64,Int64,Int64,Int64,Float64},1}
    rewards::Array{Tuple{Int64,Int64,Int64,Float64},1}
    partitions::Array{Partition,1}

    minimal_reward::Float64
    maximal_reward::Float64

    transition_map::Dict{Tuple{Int64,Int64,Int64,Int64,Int64},Float64}
end

function Game(parameters)
    state_count = parse(Int64, parameters[1])
    partition_count = parse(Int64, parameters[2])
    leader_action_count = parse(Int64, parameters[3])
    follower_action_count = parse(Int64, parameters[4])
    observation_count = parse(Int64, parameters[5])
    transition_count = parse(Int64, parameters[6])
    reward_count = parse(Int64, parameters[7])
    disc = parse(Float64, parameters[8])

    states = Array{State,1}(undef, state_count)
    partitions = Array{Partition,1}(undef, partition_count)
    leader_actions = Array{String,1}(undef, leader_action_count)
    follower_actions = Array{String,1}(undef, follower_action_count)
    observations = Array{String,1}(undef, observation_count)
    transitions = Array{Tuple{Int64,Int64,Int64,Int64,Int64,Float64},1}(undef, transition_count)
    rewards = Array{Tuple{Int64,Int64,Int64,Float64},1}(undef, reward_count)

    return Game(state_count, partition_count, leader_action_count, follower_action_count,
        observation_count, transition_count, reward_count, disc,
        states, leader_actions, follower_actions, observations,
        transitions, rewards, partitions)
end

function Game(state_count, partition_count, leader_action_count, follower_action_count,
    observation_count, transition_count, reward_count, disc,
    states, leader_actions, follower_actions, observations,
    transitions, rewards, partitions)

    transition_map = Dict{Tuple{Int64,Int64,Int64,Int64,Int64},Float64}([])

    game = Game(state_count, partition_count, leader_action_count, follower_action_count,
        observation_count, transition_count, reward_count, disc,
        states, leader_actions, follower_actions, observations,
        transitions, rewards, partitions,
        0, 0, transition_map)

    return game
end

function prepare(game::Game)
    game.minimal_reward = minimum([r[4] for r in game.rewards])
    game.maximal_reward = maximum([r[4] for r in game.rewards])

    for transition in game.transitions
        game.transition_map[transition[1:5]] = transition[6]
    end

    for transition in game.transitions
        s, a1, a2, o, sp, prob = transition
        partition = game.states[s].partition

        if haskey(partition.ao_pair_transitions, (a1, o))
            push!(partition.ao_pair_transitions[(a1, o)], transition)
        else
            partition.ao_pair_transitions[(a1, o)] = [transition]
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

        target_partition = game.states[sp].partition_index
        if !haskey(partition.partition_transitions, (a1, o))
            partition.partition_transitions[(a1, o)] = target_partition
        elseif partition.partition_transitions[(a1, o)] != target_partition
            throw(MultiPartitionTransitionException())
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

LB_min(game::Game) = game.minimal_reward / (1 - game.disc)

UB_max(game::Game) = game.maximal_reward / (1 - game.disc)

lipschitz_delta(game::Game) = (UB_max(game) - LB_min(game)) / 2
