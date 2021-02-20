mutable struct State
    index::Int64
    partition::Int64
    belief_index::Int64
    follower_actions::Vector{Int64}

    presolve_UB_value::Float64

    name::String
end

function Base.show(io::IO, state::State)
    print(io, "State: $state.index")
end
