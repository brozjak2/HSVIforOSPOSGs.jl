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
    start::Float64,
)
    while excess(initPartition, initBelief, epsilon) > 0
        globalAlphaCount = sum(length(p.gamma) for p in game.partitions)
        globalUpsilonCount = sum(length(p.upsilon) for p in game.partitions)

        println(@sprintf(
            "%7.3fs\t%+9.3f\t%+9.3f\t%+9.3f\t%5d\t%5d",
            time() - start,
            LBValue(initPartition, initBelief),
            UBValue(initPartition, initBelief),
            width(initPartition, initBelief),
            globalAlphaCount,
            globalUpsilonCount
        ))

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

function main(
    gameFilePath::String,
    epsilon::Float64,
    D::Float64,
    presolveDelta::Float64,
    presolveLimit::Float64,
)
    start = time()
    game, initPartition, initBelief = load(gameFilePath)
    println(@sprintf("%7.3fs\tGame loaded", time() - start))
    prepare(game)
    println(@sprintf("%7.3fs\tGame prepared", time() - start))

    @assert 0 < D < (1 - game.disc) * epsilon / (2 * lipschitzdelta(game)) @sprintf(
        "neighborhood parameter D = %.5f is outside bounds (%.5f, %.5f)",
        D,
        0,
        (1 - game.disc) * epsilon / (2 * lipschitzdelta(game))
    )

    println("GAME:")
    println("nStates: $(game.nStates)")
    println("nPartitions: $(game.nPartitions)")
    println("nLeaderActions: $(game.nLeaderActions)")
    println("nFollowerActions: $(game.nFollowerActions)")
    println("nObservations: $(game.nObservations)")
    println("nTransitions: $(game.nTransitions)")
    println("nRewards: $(game.nRewards)")
    println("minReward: $(game.minReward)")
    println("maxReward: $(game.maxReward)")
    println("Lmin: $(Lmin(game))")
    println("Umax: $(Umax(game))")
    println("lipschitzdelta: $(lipschitzdelta(game))")
    println("D: $D")
    println("epsilon: $epsilon")
    println("initPartition: $(initPartition.index)")
    println("initBelief: $initBelief")
    println("--------------------")

    presolveUB(game, presolveDelta, presolveLimit)
    println(@sprintf(
        "%7.3fs\tpresolveUB\t%+9.3f",
        time() - start,
        UBValue(initPartition, initBelief),
    ))
    presolveLB(game, presolveDelta, presolveLimit)
    println(@sprintf(
        "%7.3fs\tpresolveLB\t%+9.3f",
        time() - start,
        LBValue(initPartition, initBelief),
    ))

    solve(game, initPartition, initBelief, epsilon, D, start)
    println(@sprintf(
        "%7.3fs\tGame solved\t%+9.3f",
        time() - start,
        width(initPartition, initBelief),
    ))
end

end
