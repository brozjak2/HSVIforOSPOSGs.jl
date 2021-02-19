mutable struct Game <: AbstractGame
    discount_factor::Float64
    states::Vector{State}
    partitions::Vector{Partition}
    lipschitz_delta::Float64

    init_partition::Union{Partition,Nothing}
    init_belief::Vector{Float64}

    transitions::Vector{Transition}
    transition_map::Dict{Tuple{Int64,Int64,Int64,Int64,Int64},Float64}

    rewards::Vector{Reward}
    reward_map::Dict{Tuple{Int64,Int64,Int64},Float64}

    minimal_reward::Float64
    maximal_reward::Float64

    leader_actions_names::Vector{String}
    follower_actions_names::Vector{String}
    observations_names::Vector{String}
end

function Game(parsed_game_definition::ParsedGameDefinition)
    @unpack game_params, states_names, states_partitions, leader_actions_names,
        follower_actions_names, observations_names, follower_actions, leader_actions,
        transitions, rewards, init_partition_index, init_belief = parsed_game_definition
    @unpack state_count, partition_count, leader_action_count, follower_action_count,
        observation_count, transition_count, reward_count, discount_factor = game_params

    minimal_reward = minimum([r.r for r in parsed_game_definition.rewards])
    maximal_reward = maximum([r.r for r in parsed_game_definition.rewards])
    lipschitz_delta = (UB_max(maximal_reward, discount_factor) - LB_min(minimal_reward, discount_factor)) / 2

    reward_map = Dict([((r.s, r.a1, r.a2), r.r) for r in rewards])
    transition_map = Dict([((t.s, t.a1, t.a2, t.o, t.sp), t.p) for t in transitions])

    states = Vector{State}(undef, state_count)
    partitions = Vector{Partition}(undef, partition_count)
    game = Game(discount_factor, states, partitions, lipschitz_delta, nothing,
        init_belief, transitions, transition_map, rewards, reward_map, minimal_reward,
        maximal_reward, leader_actions_names, follower_actions_names, observations_names)

    partitions_states = [Vector{Int64}([]) for i = 1:partition_count]

    for s = 1:state_count
        p = states_partitions[s]
        push!(partitions_states[p], s)
        game.states[s] = State(s, p, length(partitions_states[p]), follower_actions[s], states_names[s])
    end

    for p = 1:partition_count
        game.partitions[p] = Partition(game, p, partitions_states[p], leader_actions[p])
    end
    game.init_partition = game.partitions[init_partition_index]

    for transition in transitions
        @unpack s, a1, a2, o, sp, p = transition
        partition = partitions[states[s].partition]

        dictarray_push_or_init(partition.ao_pair_transitions, (a1, o), transition)
        dictarray_push_or_init(partition.transitions, s, transition)
        dictarray_push_or_init(partition.observations, a1, o)

        target_partition = states[sp].partition
        if !haskey(partition.partition_transitions, (a1, o))
            partition.partition_transitions[(a1, o)] = target_partition
        elseif partition.partition_transitions[(a1, o)] != target_partition
            throw(MultiPartitionTransitionException())
        end
    end

    for reward in rewards
        @unpack s, a1, a2, r = reward
        partition = partitions[states[s].partition]

        partition.rewards[(s, a1, a2)] = r
    end

    for partition in partitions
        for (a1, os) in partition.observations
            partition.observations[a1] = unique(os)
        end
    end

    return game
end

LB_min(game::Game) = LB_min(game.minimal_reward, game.discount_factor)
LB_min(minimal_reward::Float64, discount_factor::Float64) = minimal_reward / (1 - discount_factor)

UB_max(game::Game) = UB_max(game.maximal_reward, game.discount_factor)
UB_max(maximal_reward::Float64, discount_factor::Float64) = maximal_reward / (1 - discount_factor)
