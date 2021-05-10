"""
    function hsvi(
        game_file_path::String, epsilon::Float64;
        ub_value_method::Symbol = :nn,
        stage_game_method::Symbol = :qre,
        normalize_rewards::Bool = true,
        neigh_param_d::Float64 = 1e-6,
        presolve_min_delta::Float64 = 1e-3,
        presolve_time_limit::Float64 = 60.0,
        qre_lambda::Float64 = 100.0,
        qre_epsilon::Float64 = 1e-2,
        qre_iter_limit::Int64 = 100,
        nn_target_loss::Float64 = 1e-6,
        nn_batch_size::Int64 = 128,
        nn_learning_rate::Float64 = 1e-2,
        nn_neurons::String = "32-16",
        ub_prunning_epsilon::Float64 = 1e-2,
        output_file::String = ""
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
    - ub_prunning_epsilon: neighborhood of this size is searched when prunning after UB update
    - output_file: path to which results are written; if empty no results are written
"""
function hsvi(
    game_file_path::String, epsilon::Float64;
    ub_value_method::Symbol = :nn,
    stage_game_method::Symbol = :qre,
    normalize_rewards::Bool = true,
    neigh_param_d::Float64 = 1e-6,
    presolve_min_delta::Float64 = 1e-3,
    presolve_time_limit::Float64 = 60.0,
    qre_lambda::Float64 = 100.0,
    qre_epsilon::Float64 = 1e-2,
    qre_iter_limit::Int64 = 100,
    nn_target_loss::Float64 = 1e-6,
    nn_batch_size::Int64 = 128,
    nn_learning_rate::Float64 = 1e-2,
    nn_neurons::String = "32-16",
    ub_prunning_epsilon::Float64 = 1e-3,
    output_file::String = ""
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

    presolve_UB(context)
    if ub_value_method == :nn
        initial_nn_train(context)
    end
    @info @sprintf(
        "%7.3fs\tpresolveUB\t%+7.5f",
        time() - context.clock_start,
        UB_value(context)
    )

    presolve_LB(context)
    @info @sprintf(
        "%7.3fs\tpresolveLB\t%+7.5f",
        time() - context.clock_start,
        LB_value(context)
    )

    solve(context)
    @info @sprintf(
        "%7.3fs\t%+7.5f\t%+7.5f\t%+7.5f\tGame solved",
        time() - context.clock_start,
        LB_value(context),
        UB_value(context),
        width(context)
    )

    if output_file != ""
        save_results(output_file, context)
    end

    return context
end

function presolve_UB(context)
    @unpack args, game = context
    @unpack presolve_min_delta, presolve_time_limit = args
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
                presolve_value <= sum(policy1[a1] * next_value(game, presolve_UB_value, s, a1, a2) for a1 in partition.leader_actions))

            @constraint(state_value_model,
                sum(policy1[a1] for a1 in partition.leader_actions) == 1.0)

            optimize!(state_value_model)
            presolve_UB_value[s] = objective_value(state_value_model)

            delta = max(delta, abs(prev_value - presolve_UB_value[s]))
        end
    end
    if delta <= presolve_min_delta
        @debug @sprintf(
            "presolve_UB reached desired precision %s in %7.3fs",
            presolve_min_delta, time() - clock_start
        )
    else
        @debug @sprintf("presolve_UB reached time limit %7.3fs", presolve_time_limit)
    end

    for partition in partitions
        for s in partition.states
            state = states[s]

            belief = zeros(length(partition.states))
            belief[state.belief_index] = 1.0

            push!(partition.upsilon, (belief, presolve_UB_value[s]))
        end
    end
end

function next_value(game, presolve_UB_value, s, a1, a2)
    @unpack partitions, states, discount_factor = game

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
    if delta <= presolve_min_delta
        @debug @sprintf(
            "presolve_LB reached desired precision %s in %7.3fs",
            presolve_min_delta, time() - clock_start
        )
    else
        @debug @sprintf("presolve_LB reached time limit %7.3fs", presolve_time_limit)
    end
end

function solve(context)
    @unpack args, game = context
    @unpack epsilon, neigh_param_d = args
    @unpack init_partition, init_belief = game

    while excess(init_partition, init_belief, epsilon, context) > 0
        log_progress(context)
        explore(init_partition, init_belief, epsilon, context, 0)
        @debug "max_depth = $(context.exploration_depths[end])"

        context.exploration_count += 1
    end
end

function explore(partition, belief, rho, context, depth)
    @unpack game, args = context
    @unpack neigh_param_d = args

    _, LB_follower_policy, alpha = compute_LB(partition, belief, context)
    UB_leader_policy, _ , y = compute_UB(partition, belief, context)

    point_based_update(partition, belief, alpha, y, context)

    a1, o = select_ao_pair(partition, belief, UB_leader_policy, LB_follower_policy, rho, context)
    target_partition = game.partitions[partition.partition_transitions[(a1, o)]]

    if weighted_excess(partition, belief, UB_leader_policy, LB_follower_policy, a1, o, rho, context) > 0
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

function save_results(output_file, context)
    @unpack args, game, exploration_count, exploration_depths, clock_start = context
    global_gamma_size = sum(length(p.gamma) for p in game.partitions)
    global_upsilon_size = sum(length(p.upsilon) for p in game.partitions)

    mkpath(splitdir(output_file)[1])
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
