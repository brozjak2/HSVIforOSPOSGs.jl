mutable struct State
    index::Int64
    partition::Int64
    belief_index::Int64
    follower_actions::Vector{Int64}
    follower_action_index_table::Dict{Int64,Int64}

    presolve_UB_value::Float64

    name::String
end

function State(
    index::Int64, partition::Int64, belief_index::Int64, follower_actions::Vector{Int64},
    follower_action_index_table::Dict{Int64,Int64}, name::String
)
    return State(
        index,
        partition,
        belief_index,
        follower_actions,
        follower_action_index_table,
        0,
        name
    )
end

function Base.show(io::IO, state::State)
    print(io, "State: $state.index")
end
