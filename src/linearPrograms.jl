using JuMP
using Gurobi

function computeLBprimal(game, belief)
    gammarange = 1:size(game.gamma, 1)
    LBprimal = JuMP.Model(Gurobi.Optimizer)
    JuMP.set_optimizer_attribute(LBprimal, "OutputFlag", 0)

    @variable(LBprimal, policy1[game.actions1] >= 0) # 27f
    @variable(LBprimal, lambda[game.actions1, game.observations, gammarange] >= 0) # 27g
    @variable(LBprimal, alphavec[game.actions1, game.observations, game.states])
    @variable(LBprimal, statevalue[game.states])

    # 27a
    @objective(LBprimal, Max, sum(belief[s] * statevalue[s] for s in game.states))

    # 27b
    @constraint(LBprimal, con27b[s=game.states, a2=game.actions2],
        statevalue[s] <= sum(policy1[a1] * game.reward[s, a1, a2] for a1 in game.actions1)
                         + game.disc * sum(game.transition(s, a1, a2, o, sp) * alphavec[a1, o, sp]
                                           for a1 in game.actions1 for o in game.observations for sp in game.states))

    # 27c
    @constraint(LBprimal, con27c[a1=game.actions1, o=game.observations, sp=game.states],
           alphavec[a1, o, sp] == sum(lambda[a1, o, i] * game.gamma[i, sp] for i in gammarange))

    # 27d
    @constraint(LBprimal, con27d[a1=game.actions1, o=game.observations],
        sum(lambda[a1, o, i] for i in gammarange) == policy1[a1])

    # 27e
    @constraint(LBprimal, con27e,
        sum(policy1[a1] for a1 in game.actions1) == 1)

    optimize!(LBprimal)

    # policy of 2nd player is represented as joint probability in the LPs
    policy2conditional = - dual.(LBprimal[:con27b]).data ./ belief
    policy2conditional = map(x -> isinf(x) | isnan(x) ? zero(x) : x, policy2conditional)

    return value.(LBprimal[:policy1]).data, policy2conditional, value.(LBprimal[:statevalue]).data
end

function computeUBdual(game, belief)
    upsilonrange = 1:size(game.upsilon, 1)
    UBdual = JuMP.Model(Gurobi.Optimizer)
    JuMP.set_optimizer_attribute(UBdual, "OutputFlag", 0)

    @variable(UBdual, gamevalue)
    @variable(UBdual, policy2[game.states, game.actions2] >= 0) # 28f
    @variable(UBdual, beliefupdate[game.actions1, game.observations, game.states])
    @variable(UBdual, subgamevalue[game.actions1, game.observations])
    @variable(UBdual, lambda[game.actions1, game.observations, upsilonrange] >= 0) # 36f
    @variable(UBdual, delta[game.actions1, game.observations, game.states])
    @variable(UBdual, beliefp[game.actions1, game.observations, game.states])

    # 28a
    @objective(UBdual, Min, gamevalue)

    # 28b
    @constraint(UBdual, con28b[a1=game.actions1],
        gamevalue >= sum(policy2[s, a2] * game.reward[s, a1, a2] for s in game.states for a2 in game.actions2)
                     + game.disc * sum(subgamevalue[a1, o] for o in game.observations))

    # 28d
    @constraint(UBdual, con28d[a1=game.actions1, o=game.observations, sp=game.states],
        beliefupdate[a1, o, sp] >= sum(game.transition(s, a1, a2, o, sp) * policy2[s, a2]
                                       for s in game.states for a2 in game.actions2))

    # 28e
    @constraint(UBdual, con28e[s=game.states],
        sum(policy2[s, a2] for a2 in game.actions2) == belief[s])

    # 36a
    @constraint(UBdual, con36a[a1=game.actions1, o=game.observations],
        subgamevalue[a1, o] == sum(lambda[a1, o, i] * game.upsilon[i][2] for i in upsilonrange)
                               + game.lipschitzdelta * sum(delta[a1, o, sp] for sp in game.states))

    # 36b
    @constraint(UBdual, con36b[a1=game.actions1, o=game.observations, sp=game.states],
        sum(lambda[a1, o, i] * game.upsilon[i][1][sp] for i in upsilonrange) == beliefp[a1, o, sp])

    # 36c
    @constraint(UBdual, con36c[a1=game.actions1, o=game.observations, sp=game.states],
        delta[a1, o, sp] >= beliefp[a1, o, sp] - beliefupdate[a1, o, sp])

    # 36d
    @constraint(UBdual, con36d[a1=game.actions1, o=game.observations, sp=game.states],
        delta[a1, o, sp] >= beliefupdate[a1, o, sp] - beliefp[a1, o, sp])

    # 36e
    @constraint(UBdual, con36e[a1=game.actions1, o=game.observations],
        sum(lambda[a1, o, i] for i in upsilonrange) == sum(beliefupdate[a1, o, sp] for sp in game.states))

    optimize!(UBdual)

    # policy of 2nd player is represented as joint probability in the LPs
    policy2conditional = value.(UBdual[:policy2]).data ./ belief
    policy2conditional = map(x -> isinf(x) | isnan(x) ? zero(x) : x, policy2conditional)

    return dual.(UBdual[:con28b]).data, policy2conditional, value(UBdual[:gamevalue])
end
