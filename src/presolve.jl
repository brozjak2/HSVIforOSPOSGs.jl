"""
    presolve_LB(osposg::OSPOSG, hsvi::HSVI, recorder::Recorder)

Presolve LB by computing value of the game for fixed uniform policies.
"""
function presolve_LB(osposg::OSPOSG, hsvi::HSVI, recorder::Recorder)
    clock_start = time()
    L = LB_min(osposg)

    for partition in osposg.partitions
        push!(partition.gamma, fill(L, length(partition.states)))
    end

    policies = Vector{Vector{Float64}}(undef, length(osposg.partitions))
    for partition in osposg.partitions
        policy_length = length(partition.player1_actions)
        policies[partition.index] = ones(policy_length) ./ policy_length
    end

    delta = Inf
    while time() - clock_start < hsvi.presolve_time_limit && delta > hsvi.presolve_epsilon
        delta = 0.0

        for (p, partition) in enumerate(osposg.partitions)
            new_alpha = Vector{Float64}(undef, length(partition.states))

            for (si, s) in enumerate(partition.states)
                state = osposg.states[s]

                new_state_value = Inf
                for a2 in state.player2_actions

                    a2_value = 0.0
                    for (a1i, a1) in enumerate(partition.player1_actions)

                        a1_value = osposg.reward_map[s, a1, a2]
                        for (_, sp, prob) in partition.transitions[s, a1, a2]
                            target_partition = osposg.partitions[osposg.states[sp].partition]
                            target_belief_index = osposg.states[sp].belief_index
                            a1_value += osposg.discount * prob * target_partition.gamma[1][target_belief_index]
                        end

                        a2_value += policies[p][a1i] * a1_value
                    end

                    new_state_value = min(new_state_value, a2_value)
                end

                new_alpha[si] = new_state_value
            end

            delta = max(maximum(abs.(partition.gamma[1] - new_alpha)), delta)
            partition.gamma[1] = new_alpha
        end
    end

    log_presolveLB(osposg, hsvi, delta, time() - clock_start, recorder)
end

"""
    presolve_UB(osposg::OSPOSG, hsvi::HSVI, recorder::Recorder)

Presolve UB by value iteration of the perfect information variant of the game, which can be solved by linear program.
"""
function presolve_UB(osposg::OSPOSG, hsvi::HSVI, recorder::Recorder)
    clock_start = time()
    U = UB_max(osposg)

    presolve_UB_value = fill(U, length(osposg.states))

    delta = Inf
    while time() - clock_start < hsvi.presolve_time_limit && delta > hsvi.presolve_epsilon
        delta = 0.0

        for (s, state) in enumerate(osposg.states)
            prev_value = presolve_UB_value[s]

            partition = osposg.partitions[state.partition]

            model = Model(hsvi.optimizer_factory)
            set_silent(model)

            @variable(model, 1.0 >= policy1[a1=partition.player1_actions] >= 0.0)
            @variable(model, presolve_value)

            @objective(model, Max, presolve_value)

            @constraint(model, [a2 = state.player2_actions],
                presolve_value <= sum(
                    policy1[a1] * presolve_UB_utility(osposg, presolve_UB_value, partition, s, a1, a2)
                    for a1 in partition.player1_actions
                ))

            @constraint(model, sum(policy1[a1] for a1 in partition.player1_actions) == 1.0)

            optimize!(model)

            presolve_UB_value[s] = objective_value(model)

            delta = max(delta, abs(prev_value - presolve_UB_value[s]))
        end
    end

    for partition in osposg.partitions
        for s in partition.states
            state = osposg.states[s]

            belief = zeros(length(partition.states))
            belief[state.belief_index] = 1.0

            push!(partition.upsilon, (belief, presolve_UB_value[s]))
        end
    end

    log_presolveUB(osposg, hsvi, delta, time() - clock_start, recorder)
end

function presolve_UB_utility(osposg::OSPOSG, presolve_UB_value::Vector{Float64}, partition::Partition, s::Int, a1::Int, a2::Int)
    immediate_reward = osposg.reward_map[s, a1, a2]
    exp_transition_value = sum(prob * presolve_UB_value[sp] for (_, sp, prob) in partition.transitions[s, a1, a2])

    return immediate_reward + osposg.discount * exp_transition_value
end
