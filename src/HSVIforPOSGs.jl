using JuMP
using Gurobi

##

mutable struct Game
    states
    actions1
    actions2
    observations
    reward
    transition
    gamma
    upsilon
    b0
    disc
    lipschitzdelta
end

##

function computeLBprimal(game, b)
    gammarange = 1:size(game.gamma, 1)
    LBprimal = JuMP.Model(Gurobi.Optimizer)
    JuMP.set_optimizer_attribute(LBprimal, "OutputFlag", 0)

    @variable(LBprimal, policy1[game.actions1] >= 0) # 27f
    @variable(LBprimal, lambda[game.actions1, game.observations, gammarange] >= 0) # 27g
    @variable(LBprimal, alphavec[game.actions1, game.observations, game.states])
    @variable(LBprimal, statevalue[game.states])

    # 27a
    @objective(LBprimal, Max, sum(b[s] * statevalue[s] for s in game.states))

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

    return value.(LBprimal[:policy1]).data, dual.(LBprimal[:con27b]), value.(LBprimal[:statevalue]).data
end

function computeLBdual(game, b)
    gammarange = 1:size(game.gamma, 1)
    LBdual = JuMP.Model(Gurobi.Optimizer)
    JuMP.set_optimizer_attribute(LBdual, "OutputFlag", 0)

    @variable(LBdual, gamevalue)
    @variable(LBdual, policy2[game.states, game.actions2] >= 0) # 28f
    @variable(LBdual, beliefupdate[game.actions1, game.observations, game.states])
    @variable(LBdual, subgamevalue[game.actions1, game.observations])

    # 28a
    @objective(LBdual, Min, gamevalue)

    # 28b
    @constraint(LBdual, con28b[a1=game.actions1],
        gamevalue >= sum(policy2[s, a2] * game.reward[s, a1, a2] for s in game.states for a2 in game.actions2)
                     + game.disc * sum(subgamevalue[a1, o] for o in game.observations))

    # 28c
    @constraint(LBdual, con28c[a1=game.actions1, o=game.observations, i=gammarange],
        subgamevalue[a1, o] >= sum(beliefupdate[a1, o, sp] * game.gamma[i, sp] for sp in game.states))

    # 28d
    @constraint(LBdual, con28d[a1=game.actions1, o=game.observations, sp=game.states],
        beliefupdate[a1, o, sp] >= sum(game.transition(s, a1, a2, o, sp) * policy2[s, a2]
                                       for s in game.states for a2 in game.actions2))

    # 28e
    @constraint(LBdual, con28e[s=game.states],
        sum(policy2[s, a2] for a2 in game.actions2) == b[s])

    optimize!(LBdual)

    return value.(LBdual[:policy2]).data
end

function computeUBdual(game, b)
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
        sum(policy2[s, a2] for a2 in game.actions2) == b[s])

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

    return dual.(UBdual[:con28b]).data, value.(UBdual[:policy2]).data, value(UBdual[:gamevalue])
end

## GAME DEFINITION

function transition(s, a1, a2, o, sp)
    @assert 1 <= s <= 4
    @assert 1 <= a1 <= 2
    @assert 1 <= a2 <= 2
    @assert 1 <= o <= 4
    @assert 1 <= sp <= 4

    if (s != 4)
        if (sp == s && o == s)
            return 1
        end
    elseif (a1 == 2)
        if (sp == 3 && o == 3)
            return 1
        end
    elseif (a2 == 1)
        if (sp == 1 && o == 1)
            return 1
        end
    else
        if (sp == 2 && o == 2)
            return 1
        end
    end

    return 0
end

nstates = 4
nactions1 = 2
nactions2 = 2
nobservations = 4
reward = cat([0. 0.; 0. 0.; 0. 0.; 1. 3.], [0. 0.; 0. 0.; 0. 0.; 4. 3.], dims=3)
minr = minimum(reward)
maxr = maximum(reward)
gamma = repeat([minr], 1, 4)
upsilon = [([1.; 0.; 0.; 0.], maxr); ([0.; 1.; 0.; 0.], maxr); ([0.; 0.; 1.; 0.], maxr); ([0.; 0.; 0.; 1.], maxr)]
binit = [.25; 0.25; 0.25; .25]
disc = 0.9
L = minr / (1 - disc)
U = maxr / (1 - disc)
lipschitzdelta = (U - L) / 2

game = Game(1:nstates, 1:nactions1, 1:nactions2, 1:nobservations,
            reward, transition, gamma, upsilon, binit, disc, lipschitzdelta)

##

# while true
for i in 1:10
    b = game.b0
    policy1lb, policy2lb, alpha = computeLBprimal(game, b)
    policy1ub, policy2ub, y = computeUBdual(game, b)

    game.gamma = [game.gamma; reshape(alpha, (1, :))]
    game.upsilon = [game.upsilon; (b, y)]
end

##
function transitionProbability(game, belief, policy1, policy2, s, a1, a2, o, sp)
    return belief[s] * policy1[a1] * policy2[s, a2] * game.transition(s, a1, a2, o, sp)
end

function A1OPairProbability(game, belief, policy1, policy2, a1, o)
    return sum(transitionProbability(game, belief, policy1, policy2, s, a1, a2, o, sp)
               for s in game.states for a2 in game.actions2 for sp in game.states)
end

function beliefUpdate(game, belief, policy1, policy2, a1, o)
    invA1OP = 1 / A1OPairProbability(game, belief, policy1, policy2, a1, o)
    invA1OP = map(x -> isinf(x) ? zero(x) : x, invA1OP) # handle division by zero

    pSums = [sum(transitionProbability(game, belief, policy1, policy2, s, a1, a2, o, sp)
                 for s in game.states for a2 in game.actions2)
             for sp in game.states]

    return  invA1OP * pSums
end

"""
    Compute rho(t + 1) given rho(t)
    rho(0) = epsilon
"""
function nextRho(prevRho, game, D)
    return (prevRho - 2 * game.lipschitzdelta * D) / game.disc
end

"""
    Excess of the gap between V_LB(belief) and V_UB(belief) after substracting rho(t)
"""
function excess(game, belief, rho)
    return UBvalue(game, belief) - LBvalue(game, belief) - rho
end

function weightedExcess(game, belief, policy1, policy2, a1, o, rho)
    updatedBelief = beliefUpdate(game, belief, policy1, policy2, a1, o)

    return A1OPairProbability(game, belief, policy1, policy2, a1, o)
           * excess(game, updatedBelief, rho)
"""
    Select optimal (player 1 action, observation) pair according to maximal
    weighted excess gap heuristic
"""
function selectAOPair(game, belief, policy1, policy2, rho)
    weightedExcessGaps = [
        [weightedExcess(game, belief, policy1, policy2, a1, o, rho)
         for o in game.observations]
        for a1 in game.actions1]

    _, indices = findmax(weightedExcessGaps)
    a1 = indices[1]
    o = indices[2]

    return a1, o
