struct Args
    game_file_path::String
    epsilon::Float64
    neigh_param_d::Float64
    presolve_min_delta::Float64
    presolve_time_limit::Float64
end

function Base.show(io::IO, args::Args)
    println(io, "Args:")
    for field in fieldnames(Args)
        println(io, "$field = $(getfield(args, field))")
    end
end
