function aoPairProbability(partition::Partition, belief::Array{Float64,1},
    policy1::Array{Float64,1}, policy2::Array{Float64,2}, a1::Int64, o::Int64)
    return sum(belief[t[1]] * policy1[a1] * policy2[t[1], t[3]] * t[6]
               for t in partition.aoTransitions[(a1, o)])
end

function beliefUpdate(partition::Partition, belief::Array{Float64,1},
    policy1::Array{Float64,1}, policy2::Array{Float64,2}, a1::Int64, o::Int64)
    aoProbInv = 1 / aoPairProbability(partition, belief, policy1, policy2, a1, o)
    aoProbInv = isinf(aoProbInv) ? zero(aoProbInv) : aoProbInv # handle division by zero

    targetPartition = partition.game.partitions[partition.partitionTransitions[(a1, o)]]
    targetBelief = zeros(length(targetPartition.states))
    for t in partition.aoTransitions[(a1, o)]
        targetBelief[t[5]] += belief[t[1]] * policy1[a1] * policy2[t[1], t[3]] * t[6]
    end

    return aoProbInv * pSums
end

function nextRho(prevRho::Float64, game::Game, D::Float64)
    return (prevRho - 2 * lipschitzdelta(game) * D) / game.disc
end

function excess(partition::Partition, belief::Array{Float64,1}, rho::Float64)
    return width(partition, belief) - rho
end

function weightedExcess(partition::Partition, belief::Array{Float64,1}, policy1::Array{Float64,1}, policy2::Array{Float64,2}, a1::Int64, o::Int64, rho::Float64)
    updatedBelief = beliefUpdate(partition, belief, policy1, policy2, a1, o)

    return (aoPairProbability(partition, belief, policy1, policy2, a1, o)
           * excess(partition, updatedBelief, rho))
end

function selectAOPair(partition::Partition, belief::Array{Float64,1}, policy1::Array{Float64,1}, policy2::Array{Float64,2}, rho::Float64)
    weightedExcessGaps = [weightedExcess(partition, belief, policy1, policy2, a1, o, rho)
        for a1 in partition.leaderActions, o in partition.observations[a1]]

    _, indices = findmax(weightedExcessGaps)
    a1 = indices[1]
    o = indices[2]

    return a1, o
end
