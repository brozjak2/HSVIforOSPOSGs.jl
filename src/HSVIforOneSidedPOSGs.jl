module HSVIforOneSidedPOSGs

using Printf
using JuMP
using Gurobi

export main

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

function solve(game::Game, initPartition::Partition, initBelief::Array{Float64, 1}, epsilon::Float64, D::Float64, start::Float64
    while excess(game, initPartition, initBelief, epsilon) > 0
        globalAlphaCount = sum(length(p.gamma) for p in game.partitions)
        globalUpsilonCount = sum(length(p.upsilon) for p in game.partitions

        println(@sprintf(
            "%3.5fs\t%+3.5f\t%+3.5f\t%+3.5f\t%4d\t%4d",
            time() - start,
            LBValue(initPartition, initBelief),
            UBValue(game, initPartition, initBelief),
            width(game, initPartition, initBelief),
            globalAlphaCount,
            globalUpsilonCount
        ))

        explore(game, initPartition, initBelief, epsilon, D)
    end

    return game
end

function explore(game, partition, belief, rho, D)
    _, policy2lb, alpha = computeLBprimal(gameData, belief)
    policy1ub, _, y = computeUBdual(gameData, belief)

    pointBasedUpdate(partition, belief, alpha, y)

    # NOTE: Rework ended here
    a1, o = selectAOPair(gameData, belief, policy1ub, policy2lb, rho)

    if weightedExcess(gameData, belief, policy1ub, policy2lb, a1, o, rho) > 0
        nextBelief = beliefUpdate(gameData.game, belief, policy1ub, policy2lb, a1, o)
        gameData = explore(gameData, nextBelief, nextRho(rho, gameData, D), D)

        _, _, alpha = computeLBprimal(gameData, belief)
        _, _, y = computeUBdual(gameData, belief)
        pointBasedUpdate(gameData, alpha, belief, y)
    end

    return gameData
end

function main(gameFilePath::String, epsilon::Float64, D::Float64)
    start = time()
    game, initPartition, initBelief = load(gameFilePath)
    println(@sprintf("%3.5fs\tGame loaded...", time() - start))

    @assert 0 < D < (1 - game.disc) * epsilon / (2 * lipschitzdelta(game)) @sprintf(
        "neighborhood parameter D = %.5f is outside bounds (%.5f, %.5f)",
        D,
        0,
        (1 - game.disc) * epsilon / (2 * lipschitzdelta(game))
    )

    println("nStates: $(game.nStates)")
    println("nPartitions: $(game.nPartitions)")
    println("nLeaderActions: $(game.nLeaderActions)")
    println("nFollowerActions: $(game.nFollowerActions)")
    println("nObservations: $(game.nObservations)")
    println("nTransitions: $(game.nTransitions)")
    println("nRewards: $(game.nRewards)")
    println("minReward: $(game.minReward)")
    println("maxReward: $(game.maxReward)")
    println("LB: $(game.minReward / (1 - game.disc))")
    println("UB: $(game.maxReward / (1 - game.disc))")
    println("lipschitzdelta: $(lipschitzdelta(game))")
    println("D: $D")
    println("epsilon: $epsilon")
    println("initPartition: $(initPartitionIndex.index)")
    println("initBelief: $initBelief")

    # TODO: replace with presolveLB and presolveUB
    initBounds(game)
    println(@sprintf("%3.5fs\tBounds initialized...", time() - start))

    solve(game, initPartition, initBelief, epsilon, D, start)
end

end
