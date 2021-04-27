struct Args
    game_file_path::String
    epsilon::Float64
    lp_solver::Symbol
    ub_value_method::Symbol
    stage_game_method::Symbol
    normalize_rewards::Bool
    neigh_param_d::Float64
    presolve_min_delta::Float64
    presolve_time_limit::Float64
    qre_lambda::Float64
    qre_epsilon::Float64
    qre_iter_limit::Int64
    qre_cache_epsilon::Float64
    nn_train_epochs::Int64
    nn_retrain_epochs::Int64
    nn_learning_rate::Float64
    nn_neurons::Vector{Int64}
    ub_prunning_epsilon::Float64
end

function Base.show(io::IO, args::Args)
    println(io, "Args:")
    for field in fieldnames(Args)
        println(io, "$field = $(getfield(args, field))")
    end
end
