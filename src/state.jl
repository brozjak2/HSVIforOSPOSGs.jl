struct State
    index::Int64
    name::String
    # partition::Partition
    partition::Int64
    followerActions::Array{Int64,1}
end
