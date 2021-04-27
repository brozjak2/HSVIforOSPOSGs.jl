mutable struct Partition
    game::Union{Nothing,AbstractGame}
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

function Partition(
    index::Int64, states::Vector{Int64}, leader_actions::Vector{Int64},
    leader_action_index_table::Dict{Int64,Int64}
)
    return Partition(
        nothing,
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
        Chain(Dense(length(states), 12, σ), Dense(12, 6, σ), Dense(6, 1))
    )
end

function Base.show(io::IO, partition::Partition)
    print(io, "Partition: $partition.index")
end

function train_nn(partition::Partition, epochs::Int64)
    BATCH_SIZE = 32
    LR = 1e-2

    inputs = hcat(getfield.(partition.upsilon, 1)...)
    labels = hcat(getfield.(partition.upsilon, 2)...)

    data = Flux.Data.DataLoader((inputs, labels), batchsize=min(BATCH_SIZE, length(partition.upsilon)), shuffle=true)
    opt = ADAM(LR)
    ps = params(partition.nn)

    loss(x, y) = Flux.Losses.mse(partition.nn(x), y)

    for i in 1:epochs
        Flux.train!(loss, ps, data, opt)
    end
end
