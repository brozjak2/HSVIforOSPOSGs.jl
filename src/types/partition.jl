"""
    Partition

Type for a partition of one-sided partially observable stochastic game.

Fields:

- `index::Int` - global index of the partition
- `states::Vector{Int}` - indexes of states that belong to this partition
- `player1_actions::Vector{Int}` - indexes of player1 actions available in this partition
- `policy_index::Dict{Int,Int}` - mapping from player1 action index to in-policy index
- `target::Dict{Tuple{Int,Int},Int}` - mapping from player1 action index and observation index pair to target partition index
- `observations::Dict{Int,Vector{Int}}` - mapping from player1 action index to indexes of possible observations
- `transitions::Dict{Tuple{Int,Int,Int},Vector{Tuple{Int,Int,Float64}}}` - mapping from state index, player1 action index and player2 action index triple to possible observation index, next state index and probability triples
- `a1o_transitions::Dict{Tuple{Int,Int},Vector{Tuple{Int,Int,Int,Float64}}}` - mapping from player1 action index and observation index tuple to possible state index, player2 action index, next state index and probability quadruples
- `gamma::Vector{Vector{Float64}}` - vector of α-vectors representing the lower bound LB
- `upsilon::Vector{Tuple{Vector{Float64},Float64}}` - vector of belief-value pairs representing the upper bound UB
"""
struct Partition
    index::Int
    states::Vector{Int}
    player1_actions::Vector{Int}
    policy_index::Dict{Int,Int}

    target::Dict{Tuple{Int,Int},Int}
    observations::Dict{Int,Vector{Int}}
    transitions::Dict{Tuple{Int,Int,Int},Vector{Tuple{Int,Int,Float64}}}
    a1o_transitions::Dict{Tuple{Int,Int},Vector{Tuple{Int,Int,Int,Float64}}}

    gamma::Vector{Vector{Float64}}
    upsilon::Vector{Tuple{Vector{Float64},Float64}}

    Partition(index::Int) = new(
        index,
        Int[],
        Int[],
        Dict{Int,Int}(),
        Dict{Tuple{Int,Int},Int}(),
        Dict{Int,Vector{Int}}(),
        Dict{Tuple{Int,Int,Int},Vector{Tuple{Int,Int,Float64}}}(),
        Dict{Tuple{Int,Int},Vector{Tuple{Int,Int,Int,Float64}}}(),
        Vector{Float64}[],
        Tuple{Vector{Float64},Float64}[]
    )
end

function Base.show(io::IO, partition::Partition)
    print(io, "Partition($(partition.index))")
end
