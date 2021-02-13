mutable struct State
    game::AbstractGame
    index::Int64
    inPartitionIndex::Int64
    name::String
    partition::AbstractPartition
    partitionIndex::Int64
    followerActions::Array{Int64,1}
end
