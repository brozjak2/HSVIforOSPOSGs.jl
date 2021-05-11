"""
    function hsvi(
        game_file_path::String, epsilon::Float64;
        ub_value_method::Symbol = :lp,
        stage_game_method::Symbol = :lp,
        normalize_rewards::Bool = true,
        neigh_param_d::Float64 = 1e-6,
        presolve_min_delta::Float64 = 1e-6,
        presolve_time_limit::Float64 = 300.0,
        qre_lambda::Float64 = 100.0,
        qre_epsilon::Float64 = 1e-3,
        qre_iter_limit::Int64 = 1000,
        nn_target_loss::Float64 = 1e-6,
        nn_batch_size::Int64 = 64,
        nn_learning_rate::Float64 = 1e-2,
        nn_neurons::String = "32-16-8",
        ub_prunning_epsilon::Float64 = 1e-4,
        time_limit::Float64 = 3600.0,
        output_dir::String = ""
    )

Run the HSVI for One-Sided POSGs algorithm on game loaded from `game_file_path`
aiming for precision `epsilon`.

# Parameters
    - game_file_path: path to the file with game definition
    - epsilon: desired precision with which the algorithm tries to solve the value of the game
    - ub_value_method: implemetation for computing the value of the UB; either `:nn` for the
        approximative method using Neural networks or `:lp` for the exact method using Linear Programming
    - stage_game_method: implemetation for computing the value of stage game; either `:qre` for the
        approximative iterative method of Quantal response equilibrium or `:lp` for the exact
        method using Linear programming for min-max/max-min optimization
    - normalize_rewards: normalize rewards (utilities) so that the minimal and maximal
        rewards are equal to 0 and 1 respectively
    - neigh_param_d: parameter that guarantees Lipschitz continuity and convergence of the algorithm
    - presolve_min_delta: when changes to the bounds during the presolve stage of the algorithm
        are smaller than this value the presolve is terminated
    - presolve_time_limit: time limit for the presolve stage of the algorithm in seconds
    - qre_lambda: constant of the QRE algorithm which affects the convergence
    - qre_epsilon: when policies of both players do not change between consecutive iterations of
        QRE by value larger than this, the QRE algorithm terminates
    - qre_iter_limit: iteration limit for the QRE algorithm solving stage game
    - nn_target_loss: UB NNs target loss after which to stop training
    - nn_batch_size: UB NNs batch size
    - nn_learning_rate: learning rate for ADAM optimizer used in UB NNs
    - nn_neurons: number of neurons in individual hidden layers of UB NNs separated by dash
    - ub_prunning_epsilon: neighborhood of this size is searched when prunning in UB update
    - time_limit: time limit of the whole algorithm, after which it is killed, in seconds; set to Inf to turn off
    - output_dir: directory to which results are written; if empty no results are written
"""
function hsvi(
    game_file_path::String, epsilon::Float64;
    ub_value_method::Symbol = :lp,
    stage_game_method::Symbol = :lp,
    normalize_rewards::Bool = true,
    neigh_param_d::Float64 = 1e-6,
    presolve_min_delta::Float64 = 1e-6,
    presolve_time_limit::Float64 = 300.0,
    qre_lambda::Float64 = 100.0,
    qre_epsilon::Float64 = 1e-3,
    qre_iter_limit::Int64 = 1000,
    nn_target_loss::Float64 = 1e-6,
    nn_batch_size::Int64 = 64,
    nn_learning_rate::Float64 = 1e-2,
    nn_neurons::String = "32-16-8",
    ub_prunning_epsilon::Float64 = 1e-4,
    time_limit::Float64 = 3600.0,
    output_dir::String = ""
)
    args = Args(
        game_file_path, epsilon, ub_value_method, stage_game_method, normalize_rewards,
        neigh_param_d, presolve_min_delta, presolve_time_limit, qre_lambda, qre_epsilon,
        qre_iter_limit, nn_target_loss, nn_batch_size, nn_learning_rate, nn_neurons,
        ub_prunning_epsilon
    )
    @debug args

    game = load(args)
    @debug "Game loaded from '$(args.game_file_path)'"
    @debug game

    context = Context(args, game)
    check_neigh_param_d(context)
    flush_logs()

    presolve_UB(context)
    if ub_value_method == :nn
        initial_nn_train(context)
    end
    @info @sprintf(
        "%7.3fs\tpresolveUB\t%+7.5f",
        time() - context.clock_start,
        UB_value(context)
    )
    flush_logs()

    presolve_LB(context)
    @info @sprintf(
        "%7.3fs\tpresolveLB\t%+7.5f",
        time() - context.clock_start,
        LB_value(context)
    )
    flush_logs()

    success = solve(context, time_limit)
    @info @sprintf(
        "%7.3fs\t%+7.5f\t%+7.5f\t%+7.5f\t%s",
        time() - context.clock_start,
        LB_value(context),
        UB_value(context),
        width(context),
        success ? "Converged" : "Did not converge"
    )
    flush_logs()

    if output_dir != ""
        save_results(output_dir, context)
    end

    return context
end

function solve(context, time_limit)
    @unpack args, game, exploration_depths, clock_start = context
    @unpack epsilon, neigh_param_d = args
    @unpack init_partition, init_belief = game

    while excess(init_partition, init_belief, epsilon, context) > 0
        log_progress(context)
        flush_logs()

        context.exploration_count += 1
        explore(init_partition, init_belief, epsilon, context, 0)
        @debug "max_depth = $(exploration_depths[end])"
        flush_logs()

        if time() - clock_start >= time_limit
            @warn "reached 1h time limit and did not converge, killed"
            log_progress(context)
            return false
        end
    end

    log_progress(context)
    return true
end

function explore(partition, belief, rho, context, depth)
    @unpack game, args = context
    @unpack neigh_param_d = args

    _, LB_follower_policy, alpha = compute_LB(partition, belief, context)
    UB_leader_policy, _ , y = compute_UB(partition, belief, context)

    point_based_update(partition, belief, alpha, y, context)

    weighted_excess_gap, (a1, o) = select_ao_pair(partition, belief, UB_leader_policy, LB_follower_policy, rho, context)

    if weighted_excess_gap > 0
        target_partition = game.partitions[partition.partition_transitions[(a1, o)]]
        target_belief = get_target_belief(partition, belief, UB_leader_policy, LB_follower_policy, a1, o, context)

        explore(target_partition, target_belief, next_rho(rho, game, neigh_param_d), context, depth + 1)

        _, _, alpha = compute_LB(partition, belief, context)
        _, _ , y = compute_UB(partition, belief, context)

        point_based_update(partition, belief, alpha, y, context)
    else
        push!(context.exploration_depths, depth)
    end
end

function log_progress(context)
    @unpack game, exploration_count, clock_start = context
    global_gamma_size = sum(length(p.gamma) for p in game.partitions)
    global_upsilon_size = sum(length(p.upsilon) for p in game.partitions)

    @info @sprintf(
        "%4d:\t%7.3fs\t%+7.5f\t%+7.5f\t%+7.5f\t%5d\t%5d",
        exploration_count,
        time() - clock_start,
        LB_value(context),
        UB_value(context),
        width(context),
        global_gamma_size,
        global_upsilon_size
    )
end

function save_results(output_dir, context)
    @unpack args, game, exploration_count, exploration_depths, clock_start = context
    global_gamma_size = sum(length(p.gamma) for p in game.partitions)
    global_upsilon_size = sum(length(p.upsilon) for p in game.partitions)

    mkpath(output_dir)
    args_fields = [getfield(args, field) for field in fieldnames(Args)]
    filename = join(splitpath(args.game_file_path), "-")[1:end-5] * "-" * join(args_fields[2:end], "-") * ".csv"
    output_file = joinpath(output_dir, filename)

    open(output_file, "w") do file
        args_heading_string = join(fieldnames(Args), ",")
        result_heading_string = join(
            [
                "time",
                "lb_value",
                "ub_value",
                "width",
                "gamma_size",
                "upsilon_size",
                "exploration_count",
                "average_depth"
            ],
            ","
        )

        args_string = join([getfield(args, field) for field in fieldnames(Args)], ",")
        result_string = join(
            Any[
                time() - clock_start,
                LB_value(context),
                UB_value(context),
                width(context),
                global_gamma_size,
                global_upsilon_size,
                exploration_count,
                sum(exploration_depths) / length(exploration_depths)
            ],
            ","
        )

        write(file, args_heading_string * "," * result_heading_string * "\n")
        write(file, args_string * "," * result_string * "\n")
    end
end
