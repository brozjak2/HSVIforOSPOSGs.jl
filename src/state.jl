mutable struct State
    game::AbstractGame
    index::Int64
    in_partition_index::Int64
    name::String
    partition::AbstractPartition
    partition_index::Int64
    follower_actions::Vector{Int64}
end
