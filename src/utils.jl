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

    return (A1OPairProbability(game, belief, policy1, policy2, a1, o)
           * excess(game, updatedBelief, rho))
end

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
end

function LBvalue(game, belief)
    return maximum(sum(alpha .* belief) for alpha in eachrow(game.gamma))
end
