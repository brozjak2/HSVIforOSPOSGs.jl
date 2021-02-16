function hsvi(
    game_file_path::String,
    epsilon::Float64,
    neigh_param_d::Float64,
    presolve_min_delta::Float64,
    presolve_time_limit::Float64
)
    params = Params(epsilon, neigh_param_d, presolve_min_delta, presolve_time_limit)

    game = load(game_file_path)

    check_neigh_param(params, game)

    context = Context(params, game)

    presolve_UB(game, presolve_min_delta, presolve_time_limit)
    presolve_LB(game, presolve_min_delta, presolve_time_limit)

    solve(game, init_partition, init_belief, epsilon, neigh_param_d, start)
    @info @sprintf("%7.3fs\t%+9.3f\t%+9.3f\t%+9.3f\tGame solved",
        time() - context.clock_start,
        LB_value(init_partition, init_belief),
        UB_value(init_partition, init_belief),
        width(init_partition, init_belief)
    )

    return game, init_partition, init_belief
end

function check_neigh_param(params::Params, game::Game)
    @unpack epsilon, neigh_param_d = params
    @unpack disc, lipschitz_delta = game

    upper_limit = (1 - disc) * epsilon / (2 * lipschitz_delta)
    if neigh_param_d <= 0 || neigh_param_d >= upper_limit
        @warn @sprintf(
            "neighborhood parameter = %.5f is outside bounds (%.5f, %.5f)",
            neigh_param_d, 0, upper_limit
        )
    end
end

function presolve_UB(game::Game, presolve_min_delta::Float64, presolve_time_limit::Float64)
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

function presolve_LB(game::Game, presolve_min_delta::Float64, presolve_time_limit::Float64)
    L = LB_min(game)

    for partition in game.partitions
        n = length(partition.states)
        push!(partition.gamma, fill(L, n))
    end
end

function solve(
    game::Game,
    init_partition::Partition,
    init_belief::Vector{Float64},
    epsilon::Float64,
    neigh_param_d::Float64,
    start::Float64
)
    while excess(init_partition, init_belief, epsilon) > 0
        log_progress(game, init_partition, init_belief, start)
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

function log_progress(game::Game, init_partition::Partition,
    init_belief::Vector{Float64}, start::Float64
)
    global_gamma_size = sum(length(p.gamma) for p in game.partitions)
    global_upsilon_size = sum(length(p.upsilon) for p in game.partitions)

    @debug @sprintf("%7.3fs\t%+9.3f\t%+9.3f\t%+9.3f\t%5d\t%5d",
        time() - start,
        LB_value(init_partition, init_belief),
        UB_value(init_partition, init_belief),
        width(init_partition, init_belief),
        global_gamma_size,
        global_upsilon_size
    )
end
