"""
    function hsvi(
        game_file_path::String, epsilon::Float64;
        neighborhood::Float64 = 1e-6,
        presolve_epsilon::Float64 = 1e-4,
        presolve_time_limit::Float64 = 300.0,
        time_limit::Float64 = 3600.0
    )

Run the HSVI for One-Sided POSGs algorithm on game loaded from `game_file_path`
aiming for precision `epsilon`.

# Parameters
    - game_file_path: path to the file with game definition
    - epsilon: desired precision with which the algorithm tries to solve the value of the game
    - neighborhood: parameter that guarantees Lipschitz continuity and convergence of the algorithm
    - presolve_epsilon: when changes to the bounds during the presolve stage of the algorithm
        are smaller than this value the presolve is terminated
    - presolve_time_limit: time limit for the presolve stage of the algorithm in seconds
    - time_limit: time limit of the whole algorithm, after which it is killed, in seconds;
        set to Inf to turn off
"""
function hsvi(
    game_file_path::String, epsilon::Float64;
    neighborhood::Float64 = 1e-6,
    presolve_epsilon::Float64 = 1e-4,
    presolve_time_limit::Float64 = 300.0,
    time_limit::Float64 = 3600.0
)
    args = Args(
        game_file_path, epsilon, neighborhood, presolve_epsilon, presolve_time_limit
    )
    game = load(args)
    context = Context(args, game, time_limit)

    presolve_UB(context)

    presolve_LB(context)

    solve(context, time_limit)

    return context
end

function solve(context, time_limit)
    @unpack args, game, clock_start = context
    @unpack epsilon = args
    @unpack init_partition, init_belief = game

    save_exploration_data(context)
    log_progress(context)

    while excess(context, init_partition, init_belief, epsilon) > 0
        explore(context, init_partition, init_belief, epsilon, 0)

        context.exploration_count += 1
        save_exploration_data(context)

        log_depth(context)
        log_progress(context)

        if time() - clock_start >= time_limit
            @warn "reached time limit of $(Int(time_limit))s and did not converge, killed"
            break
        end
    end

    log_solve(context)
end

function explore(context, partition, belief, rho, depth)
    _, LB_follower_policy, alpha = compute_LB(context, partition, belief)
    UB_leader_policy, _ , y = compute_UB(context, partition, belief)

    point_based_update(context, partition, belief, alpha, y)

    weighted_excess_gap, target_partition, target_belief = select_ao(context, partition, belief, UB_leader_policy, LB_follower_policy, rho)

    if weighted_excess_gap > 0
        explore(context, target_partition, target_belief, next_rho(context, rho), depth + 1)

        _, _, alpha = compute_LB(context, partition, belief)
        _, _ , y = compute_UB(context, partition, belief)

        point_based_update(context, partition, belief, alpha, y)
    else
        push!(context.exploration_depths, depth)
    end
end

function select_ao(context, partition, belief, policy1, policy2, rho)
    @unpack game, args = context
    @unpack partitions, states, state_index_table = game

    max_weighted_excess_gap = -Inf
    max_target_partition = nothing
    max_target_belief = nothing

    for a1 in partition.leader_actions, o in partition.observations[a1]
        target_partition = partitions[partition.partition_transitions[(a1, o)]]
        target_belief = zeros(length(target_partition.states))
        ao_prob = 0.0

        for t in partition.ao_pair_transitions[(a1, o)]
            a1_index = partition.leader_action_index_table[t.a1]
            a2_index = states[t.s].follower_action_index_table[t.a2]
            s_index = state_index_table[t.s]
            sp_index = state_index_table[t.sp]

            t_prob = belief[s_index] * policy1[a1_index] * policy2[s_index][a2_index] * t.p

            ao_prob += t_prob
            target_belief[sp_index] += t_prob
        end

        if ao_prob != 0.0
            target_belief = (1.0 / ao_prob) .* target_belief
            excess_gap = excess(context, target_partition, target_belief, next_rho(context, rho))
            weighted_excess_gap = ao_prob * excess_gap

            if weighted_excess_gap > max_weighted_excess_gap
                max_weighted_excess_gap = weighted_excess_gap
                max_target_partition = target_partition
                max_target_belief = target_belief
            end
        end
    end

    return max_weighted_excess_gap, max_target_partition, max_target_belief
end

function point_based_update(context, partition, belief, alpha, y)
    push!(partition.gamma, alpha)
    push!(partition.upsilon, (belief, y))
end

function excess(context, partition, belief, rho)
    return width(context, partition, belief) - rho
end

function next_rho(context, rho)
    @unpack game, args = context
    @unpack lipschitz_delta, discount_factor = game
    @unpack neighborhood = args

    return (rho - 2 * lipschitz_delta * neighborhood) / discount_factor
end
