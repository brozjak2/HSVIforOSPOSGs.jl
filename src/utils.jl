function point_based_update(partition, belief, alpha, y, context)
    @unpack args = context
    @unpack ub_value_method = args

    push!(partition.gamma, alpha)

    if ub_value_method == :nn
        prune_and_retrain(partition, belief, y, args)
    else
        push!(partition.upsilon, (belief, y))
    end
end

function prune_and_retrain(partition, belief, y, args)
    @unpack nn_retrain_epochs, ub_prunning_epsilon = args

    for (i, (beliefp, yp)) in enumerate(partition.upsilon)
        if isapprox(belief, beliefp; atol=ub_prunning_epsilon)
            if y < yp
                deleteat!(partition.upsilon, i)
                push!(partition.upsilon, (belief, y))
                train_nn(partition, nn_retrain_epochs, args)
            end

            return
        end
    end

    push!(partition.upsilon, (belief, y))
    train_nn(partition, nn_retrain_epochs, args)
end

function select_ao_pair(partition, belief, policy1, policy2, rho, context)
    weighted_excess_gaps = Dict([])
    for a1 in partition.leader_actions, o in partition.observations[a1]
        weighted_excess_gaps[(a1, o)] = weighted_excess(partition, belief, policy1, policy2, a1, o, rho, context)
    end

    _, (a1, o) = findmax(weighted_excess_gaps)

    return a1, o
end

function weighted_excess(partition, belief, policy1, policy2, a1, o, rho, context)
    @unpack partitions = context.game

    target_belief = get_target_belief(partition, belief, policy1, policy2, a1, o, context)
    target_partition = partitions[partition.partition_transitions[(a1, o)]]

    return (ao_pair_probability(partition, belief, policy1, policy2, a1, o, context)
           * excess(target_partition, target_belief, rho, context))
end

function get_target_belief(partition, belief, policy1, policy2, a1, o, context)
    @unpack game = context
    @unpack states, partitions, state_index_table = game

    ao_inverse_prob = 1 / ao_pair_probability(partition, belief, policy1, policy2, a1, o, context)
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

function ao_pair_probability(partition, belief, policy1, policy2, a1, o, context)
    @unpack game = context
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

function excess(partition, belief, rho, context)
    return width(partition, belief, context) - rho
end

function next_rho(prev_rho, game, neigh_param_d)
    return (prev_rho - 2 * game.lipschitz_delta * neigh_param_d) / game.discount_factor
end

function check_neigh_param_d(context)
    @unpack args, game = context
    @unpack epsilon, neigh_param_d = args
    @unpack discount_factor, lipschitz_delta = game

    upper_limit = (1 - discount_factor) * epsilon / (2 * lipschitz_delta)
    if !(0 <= neigh_param_d <= upper_limit)
        @warn @sprintf(
            "neighborhood parameter = %.5f is outside bounds (%.5f, %.5f)",
            neigh_param_d, 0, upper_limit
        )
    end
end

function dictarray_push_or_init!(dictarray::Dict{K,Array{V,N}}, key::K, value::V) where {K,V,N}
    if haskey(dictarray, key)
        push!(dictarray[key], value)
    else
        dictarray[key] = [value]
    end
end

function initial_nn_train(context)
    @unpack game, args = context
    @unpack nn_train_epochs = args

    for partition in game.partitions
        train_nn(partition, nn_train_epochs, args)

        @debug @sprintf(
            "%7.3fs\tpartition %i NN trained",
            time() - context.clock_start,
            partition.index
        )
    end
end

function compute_LB(partition, belief, context)
    @unpack stage_game_method = context.args

    if stage_game_method == :lp
        return compute_LB_primal(partition, belief, context)
    elseif stage_game_method == :qre
        return compute_LB_qre(partition, belief, context)
    else
        hrow(InvalidArgumentValue("stage_game_method", stage_game_method))
    end
end

function compute_UB(partition, belief, context)
    @unpack stage_game_method = context.args

    if stage_game_method == :lp
        return compute_UB_dual(partition, belief, context)
    elseif stage_game_method == :qre
        return compute_UB_qre(partition, belief, context)
    else
        hrow(InvalidArgumentValue("stage_game_method", stage_game_method))
    end
end
