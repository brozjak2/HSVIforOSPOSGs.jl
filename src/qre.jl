compute_LB_qre(partition, belief, context) = compute_qre(partition, belief, context, LB_value)

function compute_UB_qre(partition, belief, context)
    policy1, policy2, states_values = compute_qre(partition, belief, context, UB_value)
    return policy1, policy2, dot(states_values, belief)
end

function compute_qre(partition, belief, context, value_func)
    @unpack game, args = context
    @unpack qre_lambda, qre_epsilon, qre_iter_limit = args
    @unpack discount_factor, states, partitions, state_index_table = game

    value_cache = Vector{Tuple{Tuple{Int64,Vector{Float64}},Float64}}(undef, 0)

    leader_action_count = length(partition.leader_actions)

    policy1_old = zeros(leader_action_count)
    policy1 = fill(1 / leader_action_count, leader_action_count)
    policy2_old = [zeros(length(states[s].follower_actions)) for s in partition.states]
    policy2 = [fill(1 / length(states[s].follower_actions), length(states[s].follower_actions)) for s in partition.states]

    t = 1
    while t <= qre_iter_limit && (!isapprox(policy1_old, policy1; atol=qre_epsilon) || !isapprox(policy2_old, policy2, atol=qre_epsilon))
        policy1_old = copy(policy1)
        policy2_old = copy(policy2)

        a1_values = zeros(leader_action_count)
        for a1 in partition.leader_actions
            a1i = partition.leader_action_index_table[a1]

            for o in partition.observations[a1]
                target_partition = partitions[partition.partition_transitions[(a1, o)]]
                target_belief = zeros(length(target_partition.states))
                ao_prob = 0.0

                for t in partition.ao_pair_transitions[(a1, o)]
                    a2_index = states[t.s].follower_action_index_table[t.a2]
                    s_index = state_index_table[t.s]
                    sp_index = state_index_table[t.sp]

                    t_prob = belief[s_index] * policy2[s_index][a2_index] * t.p
                    ao_prob += t_prob
                    target_belief[sp_index] += t_prob

                    a1_values[a1i] += t_prob * partition.rewards[(t.s, t.a1, t.a2)]
                end

                if ao_prob == 0.0
                    continue
                end
                target_belief = target_belief ./ ao_prob

                a1_values[a1i] += discount_factor * ao_prob * value_func(target_partition, target_belief, context)
            end
        end

        a2_values = [zeros(length(states[s].follower_actions)) for s in partition.states]
        for s in partition.states
            state = states[s]
            si = game.state_index_table[s]

            for a2 in state.follower_actions
                a2i = state.follower_action_index_table[a2]

                for a1 in partition.leader_actions, o in partition.observations[a1]
                    target_partition = partitions[partition.partition_transitions[(a1, o)]]
                    target_belief = zeros(length(target_partition.states))
                    ao_prob = 0.0

                    for t in partition.ao_pair_transitions[(a1, o)]
                        if t.a2 != a2 || t.s != s
                            continue
                        end

                        a1_index = partition.leader_action_index_table[t.a1]
                        s_index = state_index_table[t.s]
                        sp_index = state_index_table[t.sp]

                        t_prob = policy1[a1_index] * t.p
                        ao_prob += t_prob
                        target_belief[sp_index] += t_prob

                        a2_values[si][a2i] -= t_prob * partition.rewards[(t.s, t.a1, t.a2)]
                    end

                    if ao_prob == 0.0
                        continue
                    end
                    target_belief = target_belief ./ ao_prob

                    a2_values[si][a2i] -= discount_factor * ao_prob * value_func(target_partition, target_belief, context)
                end
            end
        end

        exp_parts1 = [exp(qre_lambda * value) for value in a1_values]
        exp_parts2 = [[exp(qre_lambda * value) for value in a2_s_values] for a2_s_values in a2_values]

        policy1 = exp_parts1 ./ sum(exp_parts1)
        policy2 = [exp_parts2[si] ./ sum(exp_parts2[si]) for si in 1:length(partition.states)]

        policy1 = (policy1_old .* t .+ policy1) ./ (t + 1)
        policy2 = [(policy2_old[si] .* t .+ policy2[si]) ./ (t + 1) for si in 1:length(partition.states)]

        t += 1
    end

    if t > qre_iter_limit
        @warn "QRE iteration limit $qre_iter_limit exceeded!"
    end

    states_values = zeros(length(partition.states))
    for s in partition.states
        state = states[s]
        si = game.state_index_table[s]

        for a1 in partition.leader_actions, o in partition.observations[a1]
            target_partition = partitions[partition.partition_transitions[(a1, o)]]
            target_belief = zeros(length(target_partition.states))
            ao_prob = 0.0

            for t in partition.ao_pair_transitions[(a1, o)]
                if t.s != s
                    continue
                end

                a1_index = partition.leader_action_index_table[t.a1]
                a2_index = states[t.s].follower_action_index_table[t.a2]
                s_index = state_index_table[t.s]
                sp_index = state_index_table[t.sp]

                t_prob = policy1[a1_index] * policy2[s_index][a2_index] * t.p
                ao_prob += t_prob
                target_belief[sp_index] += t_prob

                states_values[si] += t_prob * partition.rewards[(t.s, t.a1, t.a2)]
            end

            if ao_prob == 0.0
                continue
            end
            target_belief = target_belief ./ ao_prob

            states_values[si] += discount_factor * ao_prob * get_cache_value(value_cache, value_func, target_partition, target_belief, context)
        end
    end

    return policy1, policy2, states_values
end
