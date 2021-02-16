mutable struct Game
    discount_factor::Float64
    states::Vector{State}
    leader_actions::Vector{String}
    follower_actions::Vector{String}
    observations::Vector{String}
    transitions::Vector{Tuple{Int64,Int64,Int64,Int64,Int64,Float64}}
    rewards::Vector{Tuple{Int64,Int64,Int64,Float64}}
    partitions::Vector{Partition}

    minimal_reward::Float64
    maximal_reward::Float64

    reward_map::Dict{Tuple{Int64,Int64,Int64},Float64}
    transition_map::Dict{Tuple{Int64,Int64,Int64,Int64,Int64},Float64}
end

function Game(parsed_game_definition::ParsedGameDefinition)
    @unpack game_params, states_names, states_partitions, leader_actions_names,
        follower_actions_names, observations_names, follower_actions, leader_actions,
        transitions, rewards, init_partition, init_belief = parsed_game_definition

    minimal_reward = minimum([r.value for r in parsed_game_definition.rewards])
    maximal_reward = maximum([r.value for r in parsed_game_definition.rewards])

    reward_map = Dict([((r.s, r.a1, r.a2), r.v) for r in rewards])
    transition_map = Dict([((t.s, t.a1, t.a2, t.o, t.sp), t.p) for t in transitions])

    #TODO: Fix game initialization

    # for i = 1:length(game.partitions)
    #     game.partitions[i] = Partition(i)
    # end

    # for i = 1:length(game.states)
    #     name = readuntil(file, ' ')
    #     partition = parse(Int64, readuntil(file, '\n')) + 1
    #     partition = game.partitions[partition]
    #     push!(partition.states, i)
    #     game.states[i] = State(i, length(partition.states),
    #         name, partition, Vector{Int64}(undef, 0))
    # end

    # for i = 1:length(game.states)
    #     actions = [parse(Int64, x) + 1 for x in split(readline(file), ' ')]
    #     append!(game.states[i].follower_actions, actions)
    # end

    # for i = 1:length(game.partitions)
    #     actions = [parse(Int64, x) + 1 for x in split(readline(file), ' ')]
    #     append!(game.partitions[i].leader_actions, actions)
    # end

    # for i = 1:length(game.transitions)
    #     parsedLine = split(readline(file), ' ')
    #     parsedInts = [parse(Int64, x) + 1 for x in parsedLine[1:5]]
    #     prob = parse(Float64, parsedLine[6])

    #     game.transitions[i] = (parsedInts..., prob)
    # end

    # for i = 1:length(game.rewards)
    #     parsedLine = split(readline(file), ' ')
    #     parsedInts = [parse(Int64, x) + 1 for x in parsedLine[1:3]]
    #     reward = parse(Float64, parsedLine[4])

    #     game.rewards[i] = (parsedInts..., reward)
    # end

    # init_partition = parse(Int64, readuntil(file, ' ')) + 1
    # init_partition = game.partitions[init_partition]
    # init_belief = [parse(Float64, x) for x in split(readline(file), ' ')]

    # return game

    states = Vector{State}(undef, game_params.state_count)
    partitions = Vector{Partition}(undef, game_params.partition_count)
    leader_actions = Vector{String}(undef, game_params.leader_action_count)
    follower_actions = Vector{String}(undef, game_params.follower_action_count)
    observations = Vector{String}(undef, game_params.observation_count)
    transitions = Vector{Tuple{Int64,Int64,Int64,Int64,Int64,Float64}}(undef, game_params.transition_count)
    rewards = Vector{Tuple{Int64,Int64,Int64,Float64}}(undef, game_params.reward_count)

    game = Game(disc, states, leader_actions, follower_actions, observations, transitions,
        rewards, partitions, 0, 0, reward_map, transition_map)

    init!(game)

    return game
end

function init!(game::Game)
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

        target_partition = game.states[sp].partition
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
