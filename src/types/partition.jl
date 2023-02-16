mutable struct Partition
    index::Int64
    states::Vector{Int64}
    leader_actions::Vector{Int64}
    leader_action_index_table::Dict{Int64,Int64}

    observations::Dict{Int64,Vector{Int64}}
    rewards::Dict{Tuple{Int64,Int64,Int64},Float64}

    gamma::Vector{Vector{Float64}}
    upsilon::Vector{Tuple{Vector{Float64},Float64}}

    transitions::Dict{Tuple{Int64,Int64,Int64},Vector{Transition}}
    ao_pair_transitions::Dict{Tuple{Int64,Int64},Vector{Transition}}
    partition_transitions::Dict{Tuple{Int64,Int64},Int64}

    nn::Chain
end

function Partition(index, states, leader_actions, leader_action_index_table, nn_neurons)
    return Partition(
        index,
        states,
        leader_actions,
        leader_action_index_table,
        Dict{Int64,Vector{Int64}}([]),
        Dict{Tuple{Int64,Int64,Int64},Float64}([]),
        Vector{Vector{Float64}}(undef, 0),
        Vector{Tuple{Vector{Float64},Float64}}(undef, 0),
        Dict{Tuple{Int64,Int64,Int64},Vector{Transition}}([]),
        Dict{Tuple{Int64,Int64},Vector{Transition}}([]),
        Dict{Tuple{Int64,Int64},Int64}([]),
        create_partition_nn(length(states), nn_neurons)
    )
end

function Base.show(io::IO, partition::Partition)
    print(io, "Partition($(partition.index))")
end
