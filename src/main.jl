using Printf

include("gameDefinition.jl")
include("linearPrograms.jl")
include("utils.jl")

mutable struct GameData
    game::Game
    disc::Float64
    lipschitzdelta::Float64
    gamma::Array{Float64,2}
    upsilon::Array{Tuple{Array{Float64,1},Float64},1}
end

function main(game, initBelief, disc, epsilon, D)
    # TODO: move gameData initialization into own function
    minr = minimum(game.reward)
    maxr = maximum(game.reward)
    L = minr / (1 - disc)
    U = maxr / (1 - disc)
    lipschitzdelta = (U - L) / 2

    @assert 0 < D < (1 - disc) * epsilon / (2 * lipschitzdelta) @sprintf(
        "neighborhood parameter D = %.5f is outside bounds (%.5f, %.5f)",
        D, 0, (1 - disc) * epsilon / (2 * lipschitzdelta))

    # TODO: move to bound initialization
    gamma = repeat([minr], 1, 4)
    upsilon = [([1.; 0.; 0.; 0.], maxr);
               ([0.; 1.; 0.; 0.], maxr);
               ([0.; 0.; 1.; 0.], maxr);
               ([0.; 0.; 0.; 1.], maxr)]

    gameData = GameData(game, disc, lipschitzdelta, gamma, upsilon)

    while excess(gameData, initBelief, epsilon) > 0
        explore(gameData, initBelief, epsilon)
    end

    # TODO : return
end

function explore(gameData, initBelief, epsilon)
    # TODO

    # _, policy2lb, alpha = computeLBprimal(game, b)
    # policy1ub, _, y = computeUBdual(game, b)
    #
    # game.gamma = [game.gamma; reshape(alpha, (1, :))]
    # game.upsilon = [game.upsilon; (b, y)]
end

##

initBelief = [.25; 0.25; 0.25; .25]
disc = 0.9
epsilon = 0.1
D = 0.0001

##

main(game, initBelief, disc, epsilon, D)
