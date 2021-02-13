module HSVIforOneSidedPOSGs

using Printf
using JuMP
using Gurobi
using Logging

export main

include("abstract_types.jl")
include("state.jl")
include("partition.jl")
include("game.jl")
include("load.jl")
include("linearPrograms.jl")
include("utils.jl")

const global GRB_ENV = Ref{Gurobi.Env}()

function __init__()
    GRB_ENV[] = Gurobi.Env()
end

function main(
    gameFilePath::String,
    epsilon::Float64,
    D::Float64,
    presolveDelta::Float64,
    presolveLimit::Float64
)
    start = time()

    game, initPartition, initBelief = load(gameFilePath)
    @info @sprintf("%7.3fs\tGame loaded", time() - start)

    prepare(game)
    @info @sprintf("%7.3fs\tGame prepared", time() - start)

    if D <= 0 || D >= (1 - game.disc) * epsilon / (2 * lipschitzdelta(game))
        @warn @sprintf("neighborhood parameter D = %.5f is outside bounds (%.5f, %.5f)",
            D, 0, (1 - game.disc) * epsilon / (2 * lipschitzdelta(game)))
    end

    begin
        @info "GAME:"
        @info "nStates: $(game.nStates)"
        @info "nPartitions: $(game.nPartitions)"
        @info "nLeaderActions: $(game.nLeaderActions)"
        @info "nFollowerActions: $(game.nFollowerActions)"
        @info "nObservations: $(game.nObservations)"
        @info "nTransitions: $(game.nTransitions)"
        @info "nRewards: $(game.nRewards)"
        @info "minReward: $(game.minReward)"
        @info "maxReward: $(game.maxReward)"
        @info "Lmin: $(Lmin(game))"
        @info "Umax: $(Umax(game))"
        @info "lipschitzdelta: $(lipschitzdelta(game))"
        @info "D: $D"
        @info "epsilon: $epsilon"
        @info "initPartition: $(initPartition.index)"
        @info "initBelief: $initBelief"
    end

    presolveUB(game, presolveDelta, presolveLimit)
    @info @sprintf("%7.3fs\tpresolveUB\t%+9.3f",
        time() - start,
        UBValue(initPartition, initBelief),
    )

    presolveLB(game, presolveDelta, presolveLimit)
    @info @sprintf("%7.3fs\tpresolveLB\t%+9.3f",
        time() - start,
        LBValue(initPartition, initBelief),
    )

    solve(game, initPartition, initBelief, epsilon, D, start)
    @info @sprintf("%7.3fs\tGame solved\t%+9.3f",
        time() - start,
        width(initPartition, initBelief),
    )

    return game, initPartition, initBelief
end

function presolveUB(game::Game, presolveDelta::Float64, presolveLimit::Float64)
    U = Umax(game)

    for partition in game.partitions
        n = length(partition.states)
        for i = 1:n
            belief = zeros(n)
            belief[i] += 1
            push!(partition.upsilon, (belief, U))
        end
    end
end

function presolveLB(game::Game, presolveDelta::Float64, presolveLimit::Float64)
    L = Lmin(game)

    for partition in game.partitions
        n = length(partition.states)
        push!(partition.gamma, fill(L, n))
    end
end

function solve(
    game::Game,
    initPartition::Partition,
    initBelief::Array{Float64,1},
    epsilon::Float64,
    D::Float64,
    start::Float64
)
    while excess(initPartition, initBelief, epsilon) > 0
        globalAlphaCount = sum(length(p.gamma) for p in game.partitions)
        globalUpsilonCount = sum(length(p.upsilon) for p in game.partitions)

        @info @sprintf("%7.3fs\t%+9.3f\t%+9.3f\t%+9.3f\t%5d\t%5d",
            time() - start,
            LBValue(initPartition, initBelief),
            UBValue(initPartition, initBelief),
            width(initPartition, initBelief),
            globalAlphaCount,
            globalUpsilonCount
        )

        explore(initPartition, initBelief, epsilon, D)
    end

    return game
end

function explore(partition, belief, rho, D)
    _, policy2lb, alpha = computeLBprimal(partition, belief)
    policy1ub, _, y = computeUBdual(partition, belief)

    pointBasedUpdate(partition, belief, alpha, y)

    a1, o = selectAOPair(partition, belief, policy1ub, policy2lb, rho)
    nextPartition = partition.game.partitions[partition.partitionTransitions[(a1, o)]]

    if weightedExcess(partition, belief, policy1ub, policy2lb, a1, o, rho) > 0
        nextBelief = beliefTransform(partition, belief, policy1ub, policy2lb, a1, o)
        explore(nextPartition, nextBelief, nextRho(rho, partition.game, D), D)

        _, _, alpha = computeLBprimal(partition, belief)
        _, _, y = computeUBdual(partition, belief)
        pointBasedUpdate(partition, belief, alpha, y)
    end
end

end
