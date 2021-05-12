function presolve_UB(context)
    @unpack args, game = context
    @unpack presolve_min_delta, presolve_time_limit, ub_value_method = args
    @unpack partitions, states = game

    clock_start = time()
    U = UB_max(game)

    presolve_UB_value = zeros(length(states))
    for state in states
        presolve_UB_value[state.index] = U
    end

    delta = Inf
    while time() - clock_start < presolve_time_limit && delta > presolve_min_delta
        delta = 0.0

        for state in states
            s = state.index
            partition = partitions[state.partition]

            prev_value = presolve_UB_value[s]

            state_value_model = Model(GLPK.Optimizer)
            JuMP.set_optimizer_attribute(state_value_model, "msg_lev", GLPK.GLP_MSG_OFF)

            @variable(state_value_model, 1.0 >= policy1[a1=partition.leader_actions] >= 0.0)
            @variable(state_value_model, presolve_value)

            @objective(state_value_model, Max, presolve_value)

            @constraint(state_value_model, [a2=state.follower_actions],
                presolve_value <= sum(
                    policy1[a1] * presolve_UB_utility(context, presolve_UB_value, s, a1, a2)
                    for a1 in partition.leader_actions
                ))

            @constraint(state_value_model,
                sum(policy1[a1] for a1 in partition.leader_actions) == 1.0)

            optimize!(state_value_model)
            presolve_UB_value[s] = objective_value(state_value_model)

            delta = max(delta, abs(prev_value - presolve_UB_value[s]))
        end
    end
    presolve_time = time() - clock_start

    for partition in partitions
        for s in partition.states
            state = states[s]

            belief = zeros(length(partition.states))
            belief[state.belief_index] = 1.0

            push!(partition.upsilon, (belief, presolve_UB_value[s]))
        end
    end

    if ub_value_method == :nn
        initial_nn_train(context)
    end

    log_presolveUB(context, delta, presolve_time)
end

function presolve_UB_utility(context, presolve_UB_value, s, a1, a2)
    @unpack partitions, states, discount_factor = context.game

    partition = partitions[states[s].partition]
    immediate_reward = partition.rewards[(s, a1, a2)]
    exp_transition_value = sum(t.p * presolve_UB_value[t.sp] for t in partition.transitions[(s, a1, a2)])

    return immediate_reward + discount_factor * exp_transition_value
end

function presolve_LB(context)
    @unpack args, game = context
    @unpack presolve_min_delta, presolve_time_limit = args
    @unpack partitions, states, discount_factor = game

    clock_start = time()
    L = LB_min(game)

    for partition in partitions
        n = length(partition.states)
        push!(partition.gamma, fill(L, n))
    end

    strategies = Vector{Vector{Float64}}(undef, length(partitions))
    for partition in partitions
        leader_action_count = length(partition.leader_actions)
        strategies[partition.index] = fill(1 / leader_action_count, leader_action_count)
    end

    delta = Inf
    while time() - clock_start < presolve_time_limit && delta > presolve_min_delta
        delta = 0.0

        for partition in partitions
            new_alpha = zeros(length(partition.states))
            a1it = partition.leader_action_index_table

            for s in partition.states
                state = states[s]

                new_state_value = Inf
                for a2 in state.follower_actions

                    a2_value = 0
                    for a1 in partition.leader_actions

                        a1_value = partition.rewards[(s, a1, a2)]
                        for t in partition.transitions[(s, a1, a2)]
                            target_partition = partitions[states[t.sp].partition]
                            target_belief_index = states[t.sp].belief_index
                            a1_value += discount_factor * t.p * target_partition.gamma[1][target_belief_index]
                        end

                        a2_value += strategies[partition.index][a1it[a1]] * a1_value
                    end

                    new_state_value = min(new_state_value, a2_value)
                end

                new_alpha[state.belief_index] = new_state_value
            end

            delta = max(maximum(abs.(partition.gamma[1] - new_alpha)), delta)
            partition.gamma[1] = new_alpha
        end
    end
    presolve_time = time() - clock_start

    log_presolveLB(context, delta, presolve_time)
end
