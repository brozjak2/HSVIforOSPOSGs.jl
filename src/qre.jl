function compute_LB_qre(partition::Partition, belief::Vector{Float64}, params::Params)
    @unpack qre_lambda, qre_epsilon = params
    game = partition.game

    reward_table = Dict((s, a1, a2) => partition.rewards[(s, a1, a2)] + game.discount_factor * sum(t.p * LB_value(game.partitions[game.states[t.sp].partition], state_belief(length(game.partitions[game.states[t.sp].partition].states), game.states[t.sp].belief_index)) for t in partition.transitions[(s, a1, a2)])
        for s in partition.states for a1 in partition.leader_actions for a2 in game.states[s].follower_actions)

    policy1 = Dict(a1 => 0 for a1 in partition.leader_actions)
    policy2 = Dict((s, a2) => 0 for s in partition.states for a2 in game.states[s].follower_actions)
    policy1_new = Dict(a1 => 1 / length(partition.leader_actions) for a1 in partition.leader_actions)
    policy2_new = Dict((s, a2) => 1 / length(game.states[s].follower_actions) for s in partition.states for a2 in game.states[s].follower_actions)

    t = 1
    while (any(abs.(policy1[a1] - policy1_new[a1] for a1 in partition.leader_actions) .> qre_epsilon) || any(abs.(policy2[(s, a2)] - policy2_new[(s, a2)] for s in partition.states for a2 in game.states[s].follower_actions) .> qre_epsilon)) && t <= 1000
        policy1 = copy(policy1_new)
        policy2 = copy(policy2_new)

        exp_parts1 = Dict(a1 => exp(qre_lambda * sum(policy2[(s, a2)] * belief[game.states[s].belief_index] * reward_table[(s, a1, a2)] for s in partition.states for a2 in game.states[s].follower_actions))
            for a1 in partition.leader_actions)
        exp_parts2 = Dict((s, a2) => exp(qre_lambda * sum(policy1[a1] * belief[game.states[s].belief_index] * - reward_table[(s, a1, a2)] for a1 in partition.leader_actions))
            for s in partition.states for a2 in game.states[s].follower_actions)

        exp_parts1_sum = sum(exp_parts1[a1] for a1 in partition.leader_actions)
        exp_parts2_sum = Dict(s => sum(exp_parts2[(s, a2)] for a2 in game.states[s].follower_actions) for s in partition.states)

        policy1_new = Dict(a1 => exp_parts1[a1] / exp_parts1_sum for a1 in partition.leader_actions)
        policy2_new = Dict((s, a2) => exp_parts2[(s, a2)] / exp_parts2_sum[s] for s in partition.states for a2 in game.states[s].follower_actions)

        policy1_new = Dict(a1 => (policy1[a1] * t + policy1_new[a1]) / (t + 1) for a1 in partition.leader_actions)
        policy2_new = Dict((s, a2) =>  (policy2[(s, a2)] * t + policy2_new[(s, a2)]) / (t + 1) for s in partition.states for a2 in game.states[s].follower_actions)

        t += 1
    end

    value = [sum(policy1_new[a1] * policy2_new[(s, a2)] * reward_table[(s, a1, a2)] for a1 in partition.leader_actions for a2 in game.states[s].follower_actions) for s in partition.states]

    return policy1_new, policy2_new, value
end

function state_belief(belief_size::Int64, belief_index::Int64)
    belief = zeros(belief_size)
    belief[belief_index] = 1.0
    return belief
end
