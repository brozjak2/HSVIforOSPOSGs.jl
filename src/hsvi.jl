function hsvi(
    game_file_path::String,
    epsilon::Float64,
    neigh_param_d::Float64,
    presolve_min_delta::Float64,
    presolve_time_limit::Float64
)
    params = Params(epsilon, neigh_param_d, presolve_min_delta, presolve_time_limit)
    @debug repr(params)

    clock_start = time()
    game = load(game_file_path)
    @debug @sprintf("Game from %s loaded and initialized in %7.3fs", game_file_path, time() - clock_start)
    @debug repr(game)

    context = Context(params, game)

    check_neigh_param(context)

    presolve_UB(context)
    @debug @sprintf("%7.3fs\tpresolveUB\t%+9.3f",
        time() - context.clock_start,
        UB_value(game)
    )
    presolve_LB(context)
    @debug @sprintf("%7.3fs\tpresolveLB\t%+9.3f",
        time() - context.clock_start,
        LB_value(game)
    )

    solve(context)
    @info @sprintf("%7.3fs\t%+9.3f\t%+9.3f\t%+9.3f\tGame solved",
        time() - context.clock_start,
        LB_value(game),
        UB_value(game),
        width(game)
    )

    return context
end

function check_neigh_param(context::Context)
    @unpack params, game = context
    @unpack epsilon, neigh_param_d = params
    @unpack discount_factor, lipschitz_delta = game

    upper_limit = (1 - discount_factor) * epsilon / (2 * lipschitz_delta)
    if neigh_param_d <= 0 || neigh_param_d >= upper_limit
        @warn @sprintf(
            "neighborhood parameter = %.5f is outside bounds (%.5f, %.5f)",
            neigh_param_d, 0, upper_limit
        )
    end
end

function presolve_UB(context::Context)
    @unpack params, game = context
    @unpack presolve_min_delta, presolve_time_limit = params

    U = UB_max(game)

    for partition in game.partitions
        n = length(partition.states)
        for i = 1:n
            belief = zeros(n)
            belief[i] += 1
            push!(partition.upsilon, (belief, U))
        end
    end
end

function presolve_LB(context::Context)
    @unpack params, game = context
    @unpack presolve_min_delta, presolve_time_limit = params

    L = LB_min(game)

    for partition in game.partitions
        n = length(partition.states)
        push!(partition.gamma, fill(L, n))
    end
end

function solve(context::Context)
    @unpack params, game = context
    @unpack epsilon, neigh_param_d = params
    @unpack init_partition, init_belief = game

    while excess(init_partition, init_belief, epsilon) > 0
        log_progress(context)
        explore(init_partition, init_belief, epsilon, neigh_param_d)
    end

    return game
end

function explore(partition, belief, rho, neigh_param_d)
    _, LB_follower_policy, alpha = compute_LB_primal(partition, belief)
    UB_leader_policy, _, y = compute_UB_dual(partition, belief)

    point_based_update(partition, belief, alpha, y)

    a1, o = select_ao_pair(partition, belief, UB_leader_policy, LB_follower_policy, rho)
    next_partition = partition.game.partitions[partition.partition_transitions[(a1, o)]]

    if weighted_excess(partition, belief, UB_leader_policy, LB_follower_policy, a1, o, rho) > 0
        next_belief = get_next_belief(partition, belief, UB_leader_policy, LB_follower_policy, a1, o)
        explore(next_partition, next_belief, next_rho(rho, partition.game, neigh_param_d), neigh_param_d)

        _, _, alpha = compute_LB_primal(partition, belief)
        _, _, y = compute_UB_dual(partition, belief)
        point_based_update(partition, belief, alpha, y)
    end
end

function log_progress(context::Context)
    @unpack game, clock_start = context
    global_gamma_size = sum(length(p.gamma) for p in game.partitions)
    global_upsilon_size = sum(length(p.upsilon) for p in game.partitions)

    @info @sprintf("%7.3fs\t%+9.3f\t%+9.3f\t%+9.3f\t%5d\t%5d",
        time() - clock_start,
        LB_value(game),
        UB_value(game),
        width(game),
        global_gamma_size,
        global_upsilon_size
    )
end
