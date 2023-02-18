"""
    solve(osposg::OSPOSG, hsvi::HSVI, epsilon::Float64, time_limit::Float64)

Run `hsvi` solver on `osposg` game with time limit `time_limit` aiming for gap of at most `epsilon` in initial belief.
"""
function solve(osposg::OSPOSG, hsvi::HSVI, epsilon::Float64, time_limit::Float64)
    check_neighborhood(osposg, hsvi, epsilon)

    recorder = Recorder()

    log_initial(osposg, hsvi)

    presolve_LB(osposg, hsvi, recorder)
    presolve_UB(osposg, hsvi, recorder)

    update_recorder(osposg, hsvi, recorder) # Commit bounds from presolve phase to recorder
    push!(recorder.exploration_depths, 0) # For record alignment

    log_start(osposg, hsvi, recorder)

    # Main loop
    while excess(osposg, hsvi, osposg.partitions[osposg.initial_partition], osposg.initial_belief, epsilon) > 0.0
        explore(osposg, hsvi, recorder, osposg.partitions[osposg.initial_partition], osposg.initial_belief, epsilon, 0)

        update_recorder(osposg, hsvi, recorder)
        log_progress(recorder)

        if time() - recorder.clock_start >= time_limit
            @warn @sprintf("reached time limit of %8.3fs and did not converge, killed", time_limit)
            break
        end
    end

    log_end(recorder, epsilon)

    return recorder
end

function explore(osposg::OSPOSG, hsvi::HSVI, recorder::Recorder, partition::Partition, belief::Vector{Float64}, rho::Float64, depth::Int)
    # Solve stage game on the way down
    _, LB_player2_policy, alpha = compute_LB(osposg, hsvi, partition, belief)
    UB_player1_policy, _, y = compute_UB(osposg, hsvi, partition, belief)

    # Update bounds on the way down
    point_based_update(partition, belief, alpha, y)

    # Select exploration pair (a1, o) by heuristic
    weighted_excess_gap, target_partition, target_belief = select_a1o(osposg, hsvi, partition, belief, UB_player1_policy, LB_player2_policy, rho)

    if weighted_excess_gap > 0
        # Recurse deeper if termination condition has not yet been met
        explore(osposg, hsvi, recorder, target_partition, target_belief, next_rho(osposg, hsvi, rho), depth + 1)

        # Solve stage game on the way up
        _, _, alpha = compute_LB(osposg, hsvi, partition, belief)
        _, _, y = compute_UB(osposg, hsvi, partition, belief)

        # Update bounds on the way up
        point_based_update(partition, belief, alpha, y)
    else
        # Record reached depth before backtracking
        push!(recorder.exploration_depths, depth)
    end
end

function select_a1o(osposg::OSPOSG, hsvi::HSVI, partition::Partition, belief::Vector{Float64}, policy1::Vector{Float64}, policy2::Vector{Vector{Float64}}, rho::Float64)
    max_weighted_excess_gap = -Inf
    max_target_partition = nothing
    max_target_belief = nothing

    # Traverse all possible (a1, o) pairs
    for a1 in partition.player1_actions, o in partition.observations[a1]
        target_partition = osposg.partitions[partition.target[a1, o]]
        target_belief = zeros(length(target_partition.states))

        # Compute (a1, o) pair probability and belief in target_partition
        a1o_prob = 0.0
        for (s, a2, sp, prob) in partition.a1o_transitions[a1, o]
            a1i = partition.policy_index[a1]
            a2i = osposg.states[s].policy_index[a2]

            si = osposg.states[s].belief_index
            spi = osposg.states[sp].belief_index

            total_prob = belief[si] * policy1[a1i] * policy2[si][a2i] * prob

            a1o_prob += total_prob
            target_belief[spi] += total_prob
        end

        if a1o_prob != 0.0
            # If (a1, o) pair is possible under computed policies, compute its excess gap weighted by probability
            target_belief = (1.0 / a1o_prob) .* target_belief
            excess_gap = excess(osposg, hsvi, target_partition, target_belief, next_rho(osposg, hsvi, rho))
            weighted_excess_gap = a1o_prob * excess_gap

            # Look for largest weighted excess gap
            if weighted_excess_gap > max_weighted_excess_gap
                max_weighted_excess_gap = weighted_excess_gap
                max_target_partition = target_partition
                max_target_belief = target_belief
            end
        end
    end

    return max_weighted_excess_gap, max_target_partition, max_target_belief
end

function point_based_update(partition::Partition, belief::Vector{Float64}, alpha::Vector{Float64}, y::Float64)
    push!(partition.gamma, alpha)
    push!(partition.upsilon, (belief, y))
end

excess(osposg::OSPOSG, hsvi::HSVI, partition::Partition, belief::Vector{Float64}, rho::Float64) =
    return width(osposg, hsvi, partition, belief) - rho

next_rho(osposg::OSPOSG, hsvi::HSVI, rho::Float64) =
    (rho - 2.0 * lipschitz_delta(osposg) * hsvi.neighborhood) / osposg.discount

"""
    check_neighborhood(osposg::OSPOSG, hsvi::HSVI, epsilon::Float64)

Checks that the `hsvi.neighborhood` parameter is within bounds for `osposg` game and gap `epsilon`.
"""
function check_neighborhood(osposg::OSPOSG, hsvi::HSVI, epsilon::Float64)
    upper_limit = (1.0 - osposg.discount) * epsilon / (2.0 * lipschitz_delta(osposg))
    if !(0.0 < hsvi.neighborhood < upper_limit)
        @warn @sprintf(
            "neighborhood parameter = %.4e is outside bounds (%.4e, %.4e)",
            hsvi.neighborhood, 0.0, upper_limit
        )
    end
end
