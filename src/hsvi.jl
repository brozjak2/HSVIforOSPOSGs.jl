function hsvi(
    game_file_path::String, epsilon::Float64;
    neigh_param_d::Float64 = 1.0e-6,
    presolve_min_delta::Float64 = 0.001,
    presolve_time_limit::Float64 = 60.0,
    qre_lambda::Float64 = 100.0,
    qre_epsilon::Float64 = 0.001,
    qre_iter_limit::Int64 = 1000
)
    params = Params(
        epsilon, neigh_param_d, presolve_min_delta, presolve_time_limit, qre_lambda,
        qre_epsilon, qre_iter_limit
    )
    @debug params

    game = load(game_file_path)
    @debug "Game loaded from $game_file_path"
    @debug game

    context = Context(params, game, time())
    check_neigh_param(context)

    presolve_UB(context)
    @info @sprintf(
        "%7.3fs\tpresolveUB\t%+7.5f",
        time() - context.clock_start,
        UB_value(game)
    )
    presolve_LB(context)
    @info @sprintf(
        "%7.3fs\tpresolveLB\t%+7.5f",
        time() - context.clock_start,
        LB_value(game)
    )

    solve(context)
    @info @sprintf(
        "%7.3fs\t%+7.5f\t%+7.5f\t%+7.5f\tGame solved",
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
    if !(0 <= neigh_param_d <= upper_limit)
        msg =  @sprintf(
            "neighborhood parameter = %.5f is outside bounds (%.5f, %.5f)",
            neigh_param_d, 0, upper_limit
        )
        @warn msg
    end
end

function presolve_UB(context::Context)
    @unpack params, game = context
    @unpack presolve_min_delta, presolve_time_limit = params
    @unpack partitions, states = game

    clock_start = time()
    U = UB_max(game)

    for state in states
        state.presolve_UB_value = U
    end

    delta = Inf
    while time() - clock_start < presolve_time_limit && delta > presolve_min_delta
        delta = 0.0

        for state in states
            prev_value = state.presolve_UB_value

            s = state.index
            partition = partitions[state.partition]

            state_value_model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
            JuMP.set_optimizer_attribute(state_value_model, "OutputFlag", 0)

            @variable(state_value_model, 1.0 >= policy1[a1=partition.leader_actions] >= 0.0)
            @variable(state_value_model, presolve_value)

            @objective(state_value_model, Max, presolve_value)

            @constraint(state_value_model, [a2=state.follower_actions],
                presolve_value <= sum(policy1[a1] * state_value(game, s, a1, a2) for a1 in partition.leader_actions))

            @constraint(state_value_model,
                sum(policy1[a1] for a1 in partition.leader_actions) == 1)

            optimize!(state_value_model)
            state.presolve_UB_value = objective_value(state_value_model)

            delta = max(delta, abs(prev_value - state.presolve_UB_value))
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
        partition_state_count = length(partition.states)
        for s in partition.states
            state = states[s]
            belief = zeros(partition_state_count)
            belief[state.belief_index] = 1.0
            push!(partition.upsilon, (belief, state.presolve_UB_value))
            # push!(partition.upsilon, (belief, state.presolve_UB_value + (rand() * 0.02 - 0.01))) # UB NN noise simulation
        end
    end
end

function state_value(game::Game, s::Int64, a1::Int64, a2::Int64)
    @unpack partitions, states, discount_factor = game

    partition = partitions[states[s].partition]
    immediate_reward = partition.rewards[(s, a1, a2)]
    exp_transition_value = sum(t.p * states[t.sp].presolve_UB_value for t in partition.transitions[(s, a1, a2)])

    return immediate_reward + discount_factor * exp_transition_value
end

function presolve_LB(context::Context)
    @unpack params, game = context
    @unpack presolve_min_delta, presolve_time_limit = params
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

function solve(context::Context)
    @unpack params, game = context
    @unpack epsilon, neigh_param_d = params
    @unpack init_partition, init_belief = game

    iteration = 0
    while excess(init_partition, init_belief, epsilon) > 0
        log_progress(context, iteration)
        explore(init_partition, init_belief, epsilon, params, 0)

        iteration += 1;
    end

    return game
end

function explore(partition::Partition, belief::Vector{Float64}, rho::Float64, params::Params, depth::Int64)
    @unpack neigh_param_d = params
    @unpack game = partition

    # _, LB_follower_policy, alpha = compute_LB_primal(partition, belief)
    # UB_leader_policy, _ , y = compute_UB_dual(partition, belief)
    _, LB_follower_policy, alpha = compute_LB_qre(partition, belief, params)
    UB_leader_policy, _ , y = compute_UB_qre(partition, belief, params)

    ##### COMPARE LP vs. QRE #####
    # _, qre_LB_follower_policy, qre_alpha = compute_LB_qre(partition, belief, params)
    # qre_UB_leader_policy, _ , qre_y = compute_UB_qre(partition, belief, params)

    # @debug "LP vs. QRE(λ=$(params.qre_lambda))"
    # @debug "belief:\t$belief)"
    # @debug "y:"
    # @debug "LP:\t$y"
    # @debug "QRE:\t$qre_y"
    # @debug "alpha:"
    # @debug "LP:\t$alpha"
    # @debug "QRE:\t$qre_alpha"
    # @debug "UB_leader_policy:"
    # @debug "LP:\t$UB_leader_policy"
    # @debug "QRE:\t$qre_UB_leader_policy"
    # @debug "LB_follower_policy:"
    # @debug "LP:\t$LB_follower_policy"
    # @debug "QRE:\t$qre_LB_follower_policy"
    ##############################

    ##### Plot QRE values with respect to lambda #####
    # xlim = (0.01, 500)
    # from, to = xlim
    # discretization = 20
    # lambdas = 10 .^ range(log10(from), log10(to); length=discretization)

    # data = nothing
    # @showprogress 1 "QRE: " for lambda in lambdas
    #     qre_params = Params(
    #         params.epsilon,
    #         params.neigh_param_d,
    #         params.presolve_min_delta,
    #         params.presolve_time_limit,
    #         lambda,
    #         params.qre_epsilon,
    #         params.qre_iter_limit
    #     )

    #     _, qre_LB_follower_policy, qre_alpha = compute_LB_qre(partition, belief, qre_params)
    #     qre_UB_leader_policy, _ , qre_y = compute_UB_qre(partition, belief, qre_params)

    #     y_diff = abs(y - qre_y)
    #     alpha_belief_diff = sum(abs.(alpha - qre_alpha) .* belief)

    #     if data === nothing
    #         data = [qre_y sum(qre_alpha .* belief) y_diff alpha_belief_diff]
    #     else
    #         data = vcat(data, [qre_y sum(qre_alpha .* belief) y_diff alpha_belief_diff])
    #     end
    # end

    # value_plot = plot(lambdas, data[:, 1:2];
    #     label=["QRE y" "QRE alpha ⋅ belief"],
    #     xscale=:log10,
    #     xlim,
    #     ylim=(0, 1),
    #     xlabel="λ",
    #     ylabel="V"
    # )
    # plot!([y sum(alpha .* belief)];
    #     seriestype=:hline,
    #     label=["LP y" "LP alpha ⋅ belief"]
    # )

    # diff_plot = plot(lambdas, data[:, 3:4];
    #     label=["y diff" "alpha ⋅ belief diff" "alpha diff" "policy1 diff" "policy2 diff"],
    #     xscale=:log10,
    #     xlim,
    #     xlabel="λ",
    #     ylabel="Δ"
    # )
    # plot!([params.epsilon];
    #     seriestype=:hline,
    #     label="epsilon",
    #     linestyle=:dash,
    #     color=:black,
    #     alpha=0.5
    # )

    # qre_plot = plot(value_plot, diff_plot;
    #     layout=(2, 1),
    #     size=(800, 1000),
    #     legend=:topleft,
    # )
    # display(qre_plot)

    # print("Press Enter to continue...")
    # readline()
    ##################################################

    point_based_update(partition, belief, alpha, y)

    a1, o = select_ao_pair(partition, belief, UB_leader_policy, LB_follower_policy, rho)
    target_partition = game.partitions[partition.partition_transitions[(a1, o)]]

    if weighted_excess(partition, belief, UB_leader_policy, LB_follower_policy, a1, o, rho) > 0
        target_belief = get_target_belief(partition, belief, UB_leader_policy, LB_follower_policy, a1, o)
        explore(target_partition, target_belief, next_rho(rho, game, neigh_param_d), params, depth + 1)

        # _, _, alpha = compute_LB_primal(partition, belief)
        # _, _, y = compute_UB_dual(partition, belief)
        _, _, alpha = compute_LB_qre(partition, belief, params)
        _, _ , y = compute_UB_qre(partition, belief, params)

        point_based_update(partition, belief, alpha, y)
    else
        @debug "max depth: $depth"
    end
end

function log_progress(context::Context, iteration::Int64)
    @unpack game, clock_start = context
    global_gamma_size = sum(length(p.gamma) for p in game.partitions)
    global_upsilon_size = sum(length(p.upsilon) for p in game.partitions)

    @info @sprintf(
        "%4d:\t%7.3fs\t%+7.5f\t%+7.5f\t%+7.5f\t%5d\t%5d",
        iteration,
        time() - clock_start,
        LB_value(game),
        UB_value(game),
        width(game),
        global_gamma_size,
        global_upsilon_size
    )
end
