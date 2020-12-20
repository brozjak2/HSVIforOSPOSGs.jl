struct Game
    # nStates::Int64
    # nPartitions::Int64
    # nLeaderActions::Int64
    # nFollowerActions::Int64
    # nObservations::Int64
    # nTransitions::Int64
    # nRewards::Int64
    disc::Float64
    states::Array{State,1}
    leaderActions::Array{String,1}
    followerActions::Array{String,1}
    observations::Array{String,1}
    # transitions::Array{Tuple{State,Int64,Int64,Int64,State,Float64},1}
    transitions::Array{Tuple{Int64,Int64,Int64,Int64,Int64,Float64},1}
    # rewards::Array{Tuple{State,Int64,Int64,Float64},1}
    rewards::Array{Tuple{Int64,Int64,Int64,Float64},1}
    partitions::Array{Partition,1}
end
