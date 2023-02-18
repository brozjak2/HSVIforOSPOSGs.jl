"""
    State

Type for a state of one-sided partially observable stochastic game.

Fields:

- `index::Int` - global index of the state
- `partition::Int` - partition index to which the state belongs
- `belief_index::Int` - in-partition index of the state
- `player2_actions::Vector{Int}` - indexes of player2 actions available in this state
- `policy_index::Dict{Int,Int}` - mapping from player2 action index to in-policy index
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
