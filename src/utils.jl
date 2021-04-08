function point_based_update(
    partition::Partition, belief::Vector{Float64}, alpha::Vector{Float64}, y::Float64
)
    push!(partition.gamma, alpha)
    # push!(partition.upsilon, (belief, y + (rand() * 0.02 - 0.01)))
    push!(partition.upsilon, (belief, y))
end

function select_ao_pair(
    partition::Partition, belief::Vector{Float64}, policy1::Vector{Float64},
    policy2::Vector{Vector{Float64}}, rho::Float64
)
    weighted_excess_gaps = Dict([])
    for a1 in partition.leader_actions, o in partition.observations[a1]
        weighted_excess_gaps[(a1, o)] = weighted_excess(partition, belief, policy1, policy2, a1, o, rho)
    end

    _, (a1, o) = findmax(weighted_excess_gaps)

    return a1, o
end

function weighted_excess(
    partition::Partition, belief::Vector{Float64}, policy1::Vector{Float64},
    policy2::Vector{Vector{Float64}}, a1::Int64, o::Int64, rho::Float64
)
    @unpack partitions = partition.game

    target_belief = get_target_belief(partition, belief, policy1, policy2, a1, o)
    target_partition = partitions[partition.partition_transitions[(a1, o)]]

    return (ao_pair_probability(partition, belief, policy1, policy2, a1, o)
           * excess(target_partition, target_belief, rho))
end

function get_target_belief(
    partition::Partition, belief::Vector{Float64}, policy1::Vector{Float64},
    policy2::Vector{Vector{Float64}}, a1::Int64, o::Int64
)
    @unpack game = partition
    @unpack states, partitions, state_index_table = game

    ao_inverse_prob = 1 / ao_pair_probability(partition, belief, policy1, policy2, a1, o)
    ao_inverse_prob = isinf(ao_inverse_prob) ? zero(ao_inverse_prob) : ao_inverse_prob # handle division by zero

    target_partition = partitions[partition.partition_transitions[(a1, o)]]
    target_belief = zeros(length(target_partition.states))
    for t in partition.ao_pair_transitions[(a1, o)]
        a1_index = partition.leader_action_index_table[a1]
        a2_index = states[t.s].follower_action_index_table[t.a2]

        s_index = state_index_table[t.s]
        sp_index = state_index_table[t.sp]
        target_belief[sp_index] += belief[s_index] * policy1[a1_index] * policy2[s_index][a2_index] * t.p
    end

    return ao_inverse_prob * target_belief
end

function ao_pair_probability(
    partition::Partition, belief::Vector{Float64}, policy1::Vector{Float64},
    policy2::Vector{Vector{Float64}}, a1::Int64, o::Int64
)
    @unpack game = partition
    @unpack states, state_index_table = game

    ao_pair_probability = 0
    for t in partition.ao_pair_transitions[(a1, o)]
        a1_index = partition.leader_action_index_table[a1]
        a2_index = states[t.s].follower_action_index_table[t.a2]

        s_index = state_index_table[t.s]
        sp_index = state_index_table[t.sp]
        ao_pair_probability += belief[s_index] * policy1[a1_index] * policy2[s_index][a2_index] * t.p
    end

    return ao_pair_probability
end

function excess(partition::Partition, belief::Vector{Float64}, rho::Float64)
    return width(partition, belief) - rho
end

function next_rho(prev_rho::Float64, game::Game, neigh_param_d::Float64)
    return (prev_rho - 2 * game.lipschitz_delta * neigh_param_d) / game.discount_factor
end
