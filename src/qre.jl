compute_LB_qre(partition, belief, params) = compute_qre(partition, belief, params, LB_value)

function compute_UB_qre(partition, belief, params)
    policy1, policy2, states_values = compute_qre(partition, belief, params, UB_value)
    return policy1, policy2, sum(states_values .* belief)
end

function compute_qre(partition::Partition, belief::Vector{Float64}, params::Params, value_func::Function)
    @unpack game = partition
    @unpack qre_lambda, qre_epsilon, qre_iter_limit = params
    @unpack discount_factor, states, partitions = game

    leader_action_count = length(partition.leader_actions)

    reward_table = Dict((s, a1, a2) => partition.rewards[(s, a1, a2)] + discount_factor * transitions_reward(partition, s, a1, a2, value_func)
        for s in partition.states for a1 in partition.leader_actions for a2 in game.states[s].follower_actions)

    policy1 = zeros(leader_action_count)
    policy1_new = fill(1 / leader_action_count, leader_action_count)
    policy2 = [zeros(length(states[s].follower_actions)) for s in partition.states]
    policy2_new = [fill(1 / length(states[s].follower_actions), length(states[s].follower_actions)) for s in partition.states]

    t = 1
    while (any(.!isapprox.(policy1, policy1_new; atol=qre_epsilon)) || any(.!isapprox.(policy2, policy2_new, atol=qre_epsilon))) && t <= qre_iter_limit
        policy1 = copy(policy1_new)
        policy2 = copy(policy2_new)

        exp_parts1 = [
            exp(qre_lambda * sum(
                policy2[si][a2i] * belief[si] * reward_table[(s, a1, a2)]
                for (si, s) in enumerate(partition.states) for (a2i, a2) in enumerate(states[s].follower_actions)))
            for a1 in partition.leader_actions]
        exp_parts2 = [[
            exp(qre_lambda * sum(
                policy1[a1i] * belief[si] * - reward_table[(s, a1, a2)] for (a1i, a1) in enumerate(partition.leader_actions)))
            for a2 in states[s].follower_actions] for (si, s) in enumerate(partition.states)]

        policy1_new = exp_parts1 ./ sum(exp_parts1)
        policy2_new = [exp_parts2[si] ./ sum(exp_parts2[si]) for si in 1:length(partition.states)]

        policy1_new = (policy1 .* t .+ policy1_new) ./ (t + 1)
        policy2_new = [(policy2[si] .* t .+ policy2_new[si]) ./ (t + 1) for si in 1:length(partition.states)]

        t += 1
    end

    if t > qre_iter_limit
        @warn "QRE iteration limit $qre_iter_limit exceeded!"
    end

    states_values = [sum(policy1_new[a1i] * policy2_new[si][a2i] * reward_table[(s, a1, a2)]
        for (a1i, a1) in enumerate(partition.leader_actions) for (a2i, a2) in enumerate(states[s].follower_actions)) for (si, s) in enumerate(partition.states)]

    return policy1_new, policy2_new, states_values
end

function transitions_reward(partition, s, a1, a2, value_func)
    game = partition.game
    transitions = partition.transitions[(s, a1, a2)]
    weighted_transition_values = [t.p * transition_value(game, t, value_func) for t in transitions]
    return sum(weighted_transition_values)
end

function transition_value(game, t, value_func)
    @unpack states, partitions = game
    target_state = states[t.sp]
    target_partition = partitions[target_state.partition]

    target_belief = zeros(length(target_partition.states))
    target_belief[target_state.belief_index] = 1.0

    return value_func(target_partition, target_belief)
end
