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
function nextRho(prevRho, gameData, D)
    return (prevRho - 2 * gameData.lipschitzdelta * D) / gameData.disc
end

"""
    Excess of the gap between V_LB(belief) and V_UB(belief) after substracting rho(t)
"""
function excess(gameData, belief, rho)
    return UBvalue(gameData, belief) - LBvalue(gameData, belief) - rho
end

function weightedExcess(gameData, belief, policy1, policy2, a1, o, rho)
    game = gameData.game
    updatedBelief = beliefUpdate(game, belief, policy1, policy2, a1, o)

    return (A1OPairProbability(game, belief, policy1, policy2, a1, o)
           * excess(gameData, updatedBelief, rho))
end

"""
    Select optimal (player 1 action, observation) pair according to maximal
    weighted excess gap heuristic
"""
function selectAOPair(gameData, belief, policy1, policy2, rho)
    game = gameData.game
    weightedExcessGaps = [
        [weightedExcess(gameData, belief, policy1, policy2, a1, o, rho)
         for o in game.observations]
        for a1 in game.actions1]

    _, indices = findmax(weightedExcessGaps)
    a1 = indices[1]
    o = indices[2]

    return a1, o
end

function LBvalue(gameData, belief)
    return maximum(sum(alpha .* belief) for alpha in eachrow(gameData.gamma))
end
