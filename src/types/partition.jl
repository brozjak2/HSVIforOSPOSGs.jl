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
    index::Int # `p` usually refers to current partition index, `tp` is being used for target partition
    states::Vector{Int}
    player1_actions::Vector{Int} # `a1` usually refers to global player1 action index
    policy_index::Dict{Int,Int} # `a1i` usually refers to in-policy player1 action index

    target::Dict{Tuple{Int,Int},Int} # (a1, o) -> tp
    observations::Dict{Int,Vector{Int}} # (a1) -> o[]
    transitions::Dict{Tuple{Int,Int,Int},Vector{Tuple{Int,Int,Float64}}} # (s, a1, a2) -> (o, sp, prob)[]
    a1o_transitions::Dict{Tuple{Int,Int},Vector{Tuple{Int,Int,Int,Float64}}} # (a1, o) -> (s, a2, sp, prob)[]

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
