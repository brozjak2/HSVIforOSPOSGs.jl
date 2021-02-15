module HSVIforOneSidedPOSGs

using Printf
using JuMP
using Gurobi
using Logging

export main

include("exceptions.jl")
include("abstract_types.jl")
include("state.jl")
include("partition.jl")
include("game.jl")
include("load.jl")
include("linear_programs.jl")
include("utils.jl")

const global GRB_ENV = Ref{Gurobi.Env}()

function __init__()
    GRB_ENV[] = Gurobi.Env()
end

function main(
    game_file_path::String,
    epsilon::Float64,
    neigh_param_d::Float64,
    presolve_min_delta::Float64,
    presolve_time_limit::Float64
)
    start = time()

    game, init_partition, init_belief = load(game_file_path)
    @debug @sprintf("%7.3fs\tGame loaded", time() - start)

    prepare(game)
    @debug @sprintf("%7.3fs\tGame prepared", time() - start)

    if neigh_param_d <= 0 || neigh_param_d >= (1 - game.disc) * epsilon / (2 * lipschitz_delta(game))
        @warn @sprintf("neighborhood parameter neigh_param_d = %.5f is outside bounds (%.5f, %.5f)",
            neigh_param_d, 0, (1 - game.disc) * epsilon / (2 * lipschitz_delta(game)))
    end

    begin
        @debug "GAME:"
        @debug "state_count: $(game.state_count)"
        @debug "partition_count: $(game.partition_count)"
        @debug "leader_action_count: $(game.leader_action_count)"
        @debug "follower_action_count: $(game.follower_action_count)"
        @debug "observation_count: $(game.observation_count)"
        @debug "transition_count: $(game.transition_count)"
        @debug "reward_count: $(game.reward_count)"
        @debug "minimal_reward: $(game.minimal_reward)"
        @debug "maximal_reward: $(game.maximal_reward)"
        @debug "LB_min: $(LB_min(game))"
        @debug "UB_max: $(UB_max(game))"
        @debug "lipschitz_delta: $(lipschitz_delta(game))"
        @debug "neigh_param_d: $neigh_param_d"
        @debug "epsilon: $epsilon"
        @debug "init_partition: $(init_partition.index)"
        @debug "init_belief: $init_belief"
    end

    presolve_UB(game, presolve_min_delta, presolve_time_limit)
    @debug @sprintf("%7.3fs\tpresolveUB\t%+9.3f",
        time() - start,
        UB_value(init_partition, init_belief),
    )

    presolve_LB(game, presolve_min_delta, presolve_time_limit)
    @debug @sprintf("%7.3fs\tpresolveLB\t%+9.3f",
        time() - start,
        LB_value(init_partition, init_belief),
    )

    solve(game, init_partition, init_belief, epsilon, neigh_param_d, start)
    @info @sprintf("%7.3fs\t%+9.3f\t%+9.3f\t%+9.3f\tGame solved",
            time() - start,
            LB_value(init_partition, init_belief),
            UB_value(init_partition, init_belief),
            width(init_partition, init_belief),
        )

    return game, init_partition, init_belief
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

end
