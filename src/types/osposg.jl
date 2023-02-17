"""
    OSPOSG

Type for a one-sided partially observable stochastic game.
Can be loaded from `.osposg` files.
"""
struct OSPOSG
    discount::Float64

    states::Vector{State}
    partitions::Vector{Partition}

    transition_map::Dict{Tuple{Int,Int,Int,Int,Int},Float64}
    reward_map::Dict{Tuple{Int,Int,Int},Float64}

    initial_partition::Int
    initial_belief::Vector{Float64}

    state_labels::Vector{String}
    player1_action_labels::Vector{String}
    player2_action_labels::Vector{String}
    observation_labels::Vector{String}

    minimal_reward::Float64
    maximal_reward::Float64
end

function Base.show(io::IO, osposg::OSPOSG)
    println(io, "OSPOSG:")
    println(io, "  discount = $(osposg.discount)")
    println(io, "  state_count = $(length(osposg.states))")
    println(io, "  partition_count = $(length(osposg.partitions))")
    println(io, "  player1_action_count = $(length(osposg.player1_action_labels))")
    println(io, "  player2_action_count = $(length(osposg.player2_action_labels))")
    println(io, "  observation_count = $(length(osposg.observation_labels))")
    println(io, "  transition_count = $(length(osposg.transition_map))")
    println(io, "  reward_count = $(length(osposg.reward_map))")
    println(io, "  minimal_reward = $(osposg.minimal_reward)")
    println(io, "  maximal_reward = $(osposg.maximal_reward)")
    println(io, "  LB_min = $(LB_min(osposg))")
    println(io, "  UB_max = $(UB_max(osposg))")
    println(io, "  lipschitz_delta = $(lipschitz_delta(osposg))")
    println(io, "  initial_partition = $(osposg.initial_partition)")
    println(io, "  initial_belief = $(osposg.initial_belief)")
end

"""
    LB_min(osposg::OSPOSG)

Returns the minimal possible value of the game.
"""
LB_min(osposg::OSPOSG) = osposg.minimal_reward / (1.0 - osposg.discount)

"""
    UB_max(osposg::OSPOSG)

Returns the maximal possible value of the game.
"""
UB_max(osposg::OSPOSG) = osposg.maximal_reward / (1.0 - osposg.discount)

"""
    lipschitz_delta(osposg::OSPOSG)

Returns the Lipschitz delta of the game.
"""
lipschitz_delta(osposg::OSPOSG) = (UB_max(osposg) - LB_min(osposg)) / 2.0

"""
    OSPOSG(path::AbstractString)

Construct `OSPOSG` from `.osposg` file at `path`.
"""
function OSPOSG(path::AbstractString)
    return open(path, "r") do file
        description = split(readline(file), ' ')
        state_count, partition_count, player1_action_count, player2_action_count, observation_count, transition_count, reward_count = map(x -> parse(Int, x), description[1:7])
        discount = parse(Float64, description[8])

        state_labels = Vector{String}(undef, state_count)
        states = Vector{State}(undef, state_count)
        partitions = [Partition(p) for p in 1:partition_count]
        for s in 1:state_count
            state_labels[s] = readuntil(file, ' ')
            p = parse(Int, readuntil(file, '\n')) + 1
            belief_index = length(partitions[p].states) + 1

            states[s] = State(s, p, belief_index)
            push!(partitions[p].states, s)
        end

        player1_action_labels = [readline(file) for _ in 1:player1_action_count]
        player2_action_labels = [readline(file) for _ in 1:player2_action_count]
        observation_labels = [readline(file) for _ in 1:observation_count]

        for s in 1:state_count
            append!(states[s].player2_actions, [parse(Int, x) + 1 for x in split(readline(file), ' ')])
            for (i, a2) in enumerate(states[s].player2_actions)
                states[s].policy_index[a2] = i
            end
        end
        for p in 1:partition_count
            append!(partitions[p].player1_actions, [parse(Int, x) + 1 for x in split(readline(file), ' ')])
            for (i, a1) in enumerate(partitions[p].player1_actions)
                partitions[p].policy_index[a1] = i
            end
        end

        transition_map = Dict{Tuple{Int,Int,Int,Int,Int},Float64}()
        for _ in 1:transition_count
            transition = split(readline(file), ' ')
            s, a1, a2, o, sp = map(x -> parse(Int, x) + 1, transition[1:5])
            prob = parse(Float64, transition[6])
            p = states[s].partition
            tp = states[sp].partition

            if !haskey(partitions[p].target, (a1, o))
                partitions[p].target[a1, o] = tp
            elseif partitions[p].target[a1, o] != tp
                throw(MultiPartitionTransitionException())
            end

            push!(get!(partitions[p].observations, a1, Int[]), o)
            push!(get!(partitions[p].transitions, (s, a1, a2), Tuple{Int,Int,Float64}[]), (o, sp, prob))
            push!(get!(partitions[p].a1o_transitions, (a1, o), Tuple{Int,Int,Int,Float64}[]), (s, a2, sp, prob))

            transition_map[s, a1, a2, o, sp] = prob
        end

        for p in 1:partition_count
            unique!.(values(partitions[p].observations))
        end

        reward_map = Dict{Tuple{Int,Int,Int},Float64}()
        for _ in 1:reward_count
            reward = split(readline(file), ' ')
            s, a1, a2 = map(x -> parse(Int, x) + 1, reward[1:3])
            r = parse(Float64, reward[4])

            reward_map[s, a1, a2] = r
        end

        initial_partition = parse(Int, readuntil(file, ' ')) + 1
        initial_belief = [parse(Float64, x) for x in split(readline(file), ' ')]

        minimal_reward = minimum(values(reward_map))
        maximal_reward = maximum(values(reward_map))

        return OSPOSG(
            discount,
            states,
            partitions,
            transition_map,
            reward_map,
            initial_partition,
            initial_belief,
            state_labels,
            player1_action_labels,
            player2_action_labels,
            observation_labels,
            minimal_reward,
            maximal_reward
        )
    end
end

"""
    write(path::AbstractString, osposg::OSPOSG)

Writes game `osposg` in the `.osposg` format to `path`.
"""
function write(path::AbstractString, osposg::OSPOSG)
    open(path, "w") do file
        println(file, "$(length(osposg.states)) $(length(osposg.partitions)) $(length(osposg.player1_action_labels)) $(length(osposg.player2_action_labels)) $(length(osposg.observation_labels)) $(length(osposg.transition_map)) $(length(osposg.reward_map)) $(osposg.discount)")

        for (state, label) in zip(osposg.states, osposg.state_labels)
            println(file, "$label $(state.partition - 1)")
        end

        for label in osposg.player1_action_labels
            println(file, label)
        end

        for label in osposg.player2_action_labels
            println(file, label)
        end

        for label in osposg.observation_labels
            println(file, label)
        end

        for state in osposg.states
            println(file, join(state.player2_actions .- 1, ' '))
        end

        for partition in osposg.partitions
            println(file, join(partition.player1_actions .- 1, ' '))
        end

        for ((s, a1, a2, o, sp), prob) in osposg.transition_map
            println(file, "$(s - 1) $(a1 - 1) $(a2 - 1) $(o - 1) $(sp - 1) $prob")
        end

        for ((s, a1, a2), r) in osposg.reward_map
            println(file, "$(s - 1) $(a1 - 1) $(a2 - 1) $r")
        end

        println(file, "$(osposg.initial_partition - 1) $(join(osposg.initial_belief, ' '))")
    end
end
