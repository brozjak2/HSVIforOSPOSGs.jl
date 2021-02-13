function select_ao_pair(partition::Partition, belief::Array{Float64,1}, policy1::Dict{Int64,Float64}, policy2::Dict{Tuple{Int64,Int64},Float64}, rho::Float64)
    weighted_excess_gaps = Dict((a1, o) => weighted_excess(partition, belief, policy1, policy2, a1, o, rho) for a1 in partition.leader_actions for o in partition.observations[a1])

    _, (a1, o) = findmax(weighted_excess_gaps)

    return a1, o
end

function weighted_excess(partition::Partition, belief::Array{Float64,1}, policy1::Dict{Int64,Float64}, policy2::Dict{Tuple{Int64,Int64},Float64}, a1::Int64, o::Int64, rho::Float64)
    next_belief = get_next_belief(partition, belief, policy1, policy2, a1, o)
    next_partition = partition.game.partitions[partition.partition_transitions[(a1, o)]]

    return (ao_pair_probability(partition, belief, policy1, policy2, a1, o)
           * excess(next_partition, next_belief, rho))
end

function get_next_belief(partition::Partition, belief::Array{Float64,1},
    policy1::Dict{Int64,Float64}, policy2::Dict{Tuple{Int64,Int64},Float64}, a1::Int64, o::Int64)
    ao_inverse_prob = 1 / ao_pair_probability(partition, belief, policy1, policy2, a1, o)
    ao_inverse_prob = isinf(ao_inverse_prob) ? zero(ao_inverse_prob) : ao_inverse_prob # handle division by zero

    target_partition = partition.game.partitions[partition.partition_transitions[(a1, o)]]
    target_belief = zeros(length(target_partition.states))
    for t in partition.ao_pair_transitions[(a1, o)]
        target_belief[partition.game.states[t[5]].in_partition_index] += belief[partition.game.states[t[1]].in_partition_index] * policy1[a1] * policy2[(t[1], t[3])] * t[6]
    end

    return ao_inverse_prob * target_belief
end

function ao_pair_probability(partition::Partition, belief::Array{Float64,1},
    policy1::Dict{Int64,Float64}, policy2::Dict{Tuple{Int64,Int64},Float64}, a1::Int64, o::Int64)
    return sum(belief[partition.game.states[t[1]].in_partition_index] * policy1[a1] * policy2[(t[1], t[3])] * t[6]
               for t in partition.ao_pair_transitions[(a1, o)])
end

function excess(partition::Partition, belief::Array{Float64,1}, rho::Float64)
    return width(partition, belief) - rho
end

function next_rho(prev_rho::Float64, game::Game, neigh_param_d::Float64)
    return (prev_rho - 2 * lipschitz_delta(game) * neigh_param_d) / game.disc
end
