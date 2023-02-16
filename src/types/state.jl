mutable struct State
    index::Int64
    partition::Int64
    belief_index::Int64
    follower_actions::Vector{Int64}
    follower_action_index_table::Dict{Int64,Int64}
    name::String
end

function Base.show(io::IO, state::State)
    print(io, "State($(state.index))")
end
