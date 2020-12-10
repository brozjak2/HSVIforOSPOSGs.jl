using Printf
using LinearAlgebra

include("gameDefinition.jl")
include("linearPrograms.jl")
include("utils.jl")

mutable struct GameData
    game::Game
    disc::Float64
    lipschitzdelta::Float64
    gamma::Array{Array{Float64,1},1}
    upsilon::Array{Tuple{Array{Float64,1},Float64},1}
end

function main(game, initBelief, disc, epsilon, D)
    gameData = initGameData(game, disc, epsilon, D)

    while excess(gameData, initBelief, epsilon) > 0
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

##

# initBelief = [.25; 0.25; 0.25; .25]
initBelief = [0; 0; 0; 1.]
disc = 0.9
epsilon = 0.1
D = 0.0001

##

gameData = main(game, initBelief, disc, epsilon, D)
