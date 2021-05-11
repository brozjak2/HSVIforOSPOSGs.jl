function create_partition_nn(input_neurons, nn_neurons)
    in_neurons = input_neurons
    dense_layers = []

    for layer_neurons in [parse(Int64, x) for x in split(nn_neurons, "-")]
        push!(dense_layers, Dense(in_neurons, layer_neurons, σ))
        in_neurons = layer_neurons
    end

    push!(dense_layers, Dense(in_neurons, 1))

    return Chain(dense_layers...)
end

function initial_nn_train(context)
    for partition in context.game.partitions
        train_nn(context, partition)

        log_initial_nn_train(context, partition)
    end
end

function train_nn(context, partition)
    @unpack nn_target_loss, nn_batch_size, nn_learning_rate = context.args

    inputs = hcat(getfield.(partition.upsilon, 1)...)
    labels = hcat(getfield.(partition.upsilon, 2)...)

    opt = ADAM(nn_learning_rate)
    ps = Flux.params(partition.nn)

    loss(x, y) = Flux.Losses.mse(partition.nn(x), y)

    epoch = 1
    while loss(inputs, labels) > nn_target_loss || epoch == 1
        indexes = rand(1:length(partition.upsilon), nn_batch_size)
        data = [(inputs[:, indexes], labels[:, indexes])]

        Flux.train!(loss, ps, data, opt)
        epoch += 1
    end
end

function prune_and_retrain(context, partition, belief, y)
    shares_neighborhood = false
    delete_inds = []

    for (i, (beliefp, yp)) in enumerate(partition.upsilon)
        if isapprox(belief, beliefp; atol=context.args.ub_prunning_epsilon)
            shares_neighborhood = true

            if y < yp
                push!(delete_inds, i)
            end
        end
    end

    if !shares_neighborhood || (shares_neighborhood && !isempty(delete_inds))
        deleteat!(partition.upsilon, delete_inds)
        push!(partition.upsilon, (belief, y))
        train_nn(context, partition)
    end
end
