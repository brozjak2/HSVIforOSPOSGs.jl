module HSVIforOneSidedPOSGs

using Printf
using LinearAlgebra: I
using JuMP
using Gurobi

export solve, getGame, initGameData

include("gameDefinition.jl")
include("linearPrograms.jl")
include("utils.jl")

const global GRB_ENV = Ref{Gurobi.Env}()

function __init__()
    GRB_ENV[] = Gurobi.Env()
end

mutable struct GameData
    game::Game
    disc::Float64
    lipschitzdelta::Float64
    gamma::Array{Array{Float64,1},1}
    upsilon::Array{Tuple{Array{Float64,1},Float64},1}
end

function solve(game, initBelief, disc, epsilon, D)
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

end
