"""
    Partition

Type for a partition of one-sided partially observable stochastic game.
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
