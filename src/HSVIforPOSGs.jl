include("gameDefinition.jl")
include("linearPrograms.jl")
include("utils.jl")

##

# while true
for i in 1:10
    b = game.b0
    _, policy2lb, alpha = computeLBprimal(game, b)
    policy1ub, _, y = computeUBdual(game, b)

    game.gamma = [game.gamma; reshape(alpha, (1, :))]
    game.upsilon = [game.upsilon; (b, y)]
end
