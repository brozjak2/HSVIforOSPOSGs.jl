module HSVIforOneSidedPOSGs

using Printf
using JuMP
using Gurobi

export main

include("state.jl")
include("partition.jl")
include("game.jl")
include("linearPrograms.jl")
include("utils.jl")

const global GRB_ENV = Ref{Gurobi.Env}()

function __init__()
    GRB_ENV[] = Gurobi.Env()
end

function solve(game, initPartition, initBelief, epsilon, D)
    while excess(gameData, initBelief, epsilon) > 0
        # TODO: print progress
        explore(gameData, initBelief, epsilon, D)
    end

    return gameData
end

function explore(gameData, belief, rho, D)
    _, policy2lb, alpha = computeLBprimal(gameData, belief)
    policy1ub, _, y = computeUBdual(gameData, belief)

    pointBasedUpdate(gameData, alpha, belief, y)

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
    game, initPartition, initBelief = load(gameFilePath)

    @assert 0 < D < (1 - game.disc) * epsilon / (2 * lipschitzdelta(game)) @sprintf(
        "neighborhood parameter D = %.5f is outside bounds (%.5f, %.5f)",
        D, 0, (1 - game.disc) * epsilon / (2 * lipschitzdelta(game)))

    # TODO: replace with presolveLB and presolveUB
    initBounds(game)

    println("nStates: $(game.nStates)")
    println("nPartitions: $(game.nPartitions)")
    println("nLeaderActions: $(game.nLeaderActions)")
    println("nFollowerActions: $(game.nFollowerActions)")
    println("nObservations: $(game.nObservations)")
    println("nTransitions: $(game.nTransitions)")
    println("nRewards: $(game.nRewards)")
    println("lipschitzdelta: $ldelta")
    println("minReward: $(game.minReward)")
    println("maxReward: $(game.maxReward)")
    println("LB: $(game.minReward / (1 - game.disc))")
    println("UB: $(game.maxReward / (1 - game.disc))")
    println("lipschitzdelta: $(lipschitzdelta(game))")
    println("D: $D")
    println("epsilon: $epsilon")
    println("initPartition: $initPartition")
    println("initBelief: $initBelief")

    solve(game, initPartition, initBelief, epsilon, D)
end
