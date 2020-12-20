struct Partition
    index::Int64
    states::Array{Int64,1}
    leaderActions::Array{Int64,1}
    # states::Array{State,1}
    observations::Dict{Int64,Array{Int64,1}}
end
