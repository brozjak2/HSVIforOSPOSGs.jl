struct Params
    epsilon::Float64
    neigh_param_d::Float64
    presolve_min_delta::Float64
    presolve_time_limit::Float64
    qre_lambda::Float64
    qre_epsilon::Float64
    qre_iter_limit::Int64
end

function Base.show(io::IO, params::Params)
    println(io, "Params:")
    for field in fieldnames(Params)
        println(io, "  $field = $(getfield(params, field))")
    end
end
