function selectAOPair(partition::Partition, belief::Array{Float64,1}, policy1::Dict{Int64,Float64}, policy2::Dict{Tuple{Int64,Int64},Float64}, rho::Float64)
    weightedExcessGaps = Dict((a1, o) => weightedExcess(partition, belief, policy1, policy2, a1, o, rho) for a1 in partition.leaderActions for o in partition.observations[a1])

    _, (a1, o) = findmax(weightedExcessGaps)

    return a1, o
end

function weightedExcess(partition::Partition, belief::Array{Float64,1}, policy1::Dict{Int64,Float64}, policy2::Dict{Tuple{Int64,Int64},Float64}, a1::Int64, o::Int64, rho::Float64)
    transformedBelief = beliefTransform(partition, belief, policy1, policy2, a1, o)
    nextPartition = partition.game.partitions[partition.partitionTransitions[(a1, o)]]

    return (aoPairProbability(partition, belief, policy1, policy2, a1, o)
           * excess(nextPartition, transformedBelief, rho))
end

function beliefTransform(partition::Partition, belief::Array{Float64,1},
    policy1::Dict{Int64,Float64}, policy2::Dict{Tuple{Int64,Int64},Float64}, a1::Int64, o::Int64)
    aoProbInv = 1 / aoPairProbability(partition, belief, policy1, policy2, a1, o)
    aoProbInv = isinf(aoProbInv) ? zero(aoProbInv) : aoProbInv # handle division by zero

    targetPartition = partition.game.partitions[partition.partitionTransitions[(a1, o)]]
    targetBelief = zeros(length(targetPartition.states))
    for t in partition.aoTransitions[(a1, o)]
        targetBelief[partition.game.states[t[5]].inPartitionIndex] += belief[partition.game.states[t[1]].inPartitionIndex] * policy1[a1] * policy2[(t[1], t[3])] * t[6]
    end

    return aoProbInv * targetBelief
end

function aoPairProbability(partition::Partition, belief::Array{Float64,1},
    policy1::Dict{Int64,Float64}, policy2::Dict{Tuple{Int64,Int64},Float64}, a1::Int64, o::Int64)
    return sum(belief[partition.game.states[t[1]].inPartitionIndex] * policy1[a1] * policy2[(t[1], t[3])] * t[6]
               for t in partition.aoTransitions[(a1, o)])
end

function excess(partition::Partition, belief::Array{Float64,1}, rho::Float64)
    return width(partition, belief) - rho
end

function nextRho(prevRho::Float64, game::Game, D::Float64)
    return (prevRho - 2 * lipschitzdelta(game) * D) / game.disc
end
