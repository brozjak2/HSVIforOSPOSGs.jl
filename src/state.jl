struct State
    index::Int64
    inPartitionIndex::Int64
    name::String
    partition::Int64
    followerActions::Array{Int64,1}
end
