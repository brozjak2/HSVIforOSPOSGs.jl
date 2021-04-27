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

function Partition(index, states, leader_actions, leader_action_index_table, args)
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
        create_partition_nn(length(states), args)
    )
end

function Base.show(io::IO, partition::Partition)
    print(io, "Partition($(partition.index))")
end

function train_nn(partition, epochs, args)
    @unpack nn_learning_rate = args

    inputs = hcat(getfield.(partition.upsilon, 1)...)
    labels = hcat(getfield.(partition.upsilon, 2)...)

    data = Flux.Data.DataLoader((inputs, labels), batchsize=length(partition.upsilon), shuffle=true)
    opt = ADAM(nn_learning_rate)
    ps = params(partition.nn)

    loss(x, y) = Flux.Losses.mse(partition.nn(x), y)

    for i in 1:epochs
        Flux.train!(loss, ps, data, opt)
    end
end

function create_partition_nn(input_neurons, args)
    @unpack nn_neurons = args

    in_neurons = input_neurons
    dense_layers = []

    for layer_neurons in nn_neurons
        push!(dense_layers, Dense(in_neurons, layer_neurons, σ))
        in_neurons = layer_neurons
    end

    push!(dense_layers, Dense(in_neurons, 1))

    return Chain(dense_layers...)
end
