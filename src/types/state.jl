"""
    State

Type for a state of one-sided partially observable stochastic game.
"""
struct State
    index::Int
    partition::Int
    belief_index::Int
    player2_actions::Vector{Int}
    policy_index::Dict{Int,Int}

    State(index::Int, partition::Int, belief_index::Int) = new(
        index,
        partition,
        belief_index,
        Int[],
        Dict{Int,Int}()
    )
end

function Base.show(io::IO, state::State)
    print(io, "State($(state.index))")
end
