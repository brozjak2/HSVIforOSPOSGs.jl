struct Partition
    index::Int64
    states::Array{Int64,1}
    leaderActions::Array{Int64,1}
    observations::Dict{Int64,Array{Int64,1}}

    gamma::Array{Array{Float64,1},1}
    upsilon::Array{Tuple{Array{Float64,1},Float64},1}

    # TODO: compute partition transitions
end

function initBounds(partition::Partition, game)
    L = L(game)
    U = U(game)
    n = length(partition.states)

    append!(partition.gamma, [repeat([L], n)])

    for i = 1:n
        belief = zeros(n)
        belief[i] += 1
        append!(partition.upsilon, [(belief, U)])
    end
end
