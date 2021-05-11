flush_logs() = flush(global_logger().stream)

function log_initial(context)
    @debug context.args
    @debug "Game loaded from '$(context.args.game_file_path)'"
    @debug context.game
    flush_logs()
end

function log_presolveUB(context)
    @info @sprintf(
        "%7.3fs\tpresolveUB\t%+7.5f",
        time() - context.clock_start,
        UB_value(context)
    )
    flush_logs()
end

function log_presolveLB(context)
    @info @sprintf(
        "%7.3fs\tpresolveLB\t%+7.5f",
        time() - context.clock_start,
        LB_value(context)
    )
    flush_logs()
end

function log_solve(context)
    success = width(context) <= context.args.epsilon

    @info @sprintf(
        "%7.3fs\t%+7.5f\t%+7.5f\t%+7.5f\t%s",
        time() - context.clock_start,
        LB_value(context),
        UB_value(context),
        width(context),
        success ? "Converged" : "Did not converge"
    )
    flush_logs()
end

function log_progress(context)
    @unpack game, exploration_count, clock_start = context
    global_gamma_size = sum(length(p.gamma) for p in game.partitions)
    global_upsilon_size = sum(length(p.upsilon) for p in game.partitions)

    @info @sprintf(
        "%4d:\t%7.3fs\t%+7.5f\t%+7.5f\t%+7.5f\t%5d\t%5d",
        exploration_count,
        time() - clock_start,
        LB_value(context),
        UB_value(context),
        width(context),
        global_gamma_size,
        global_upsilon_size
    )
    flush_logs()
end

function log_depth(context)
    @info "\texploration_depth = $(context.exploration_depths[end])"
    flush_logs()
end

function log_initial_nn_train(context, partition)
    @debug @sprintf(
        "%7.3fs\tpartition %i NN trained",
        time() - context.clock_start,
        partition.index
    )
end

function log_results(context, output_dir)
    if output_dir == ""
        return
    end

    @unpack args, game, exploration_count, exploration_depths, clock_start = context
    global_gamma_size = sum(length(p.gamma) for p in game.partitions)
    global_upsilon_size = sum(length(p.upsilon) for p in game.partitions)

    mkpath(output_dir)

    # Create output filename using args
    args_fields = [getfield(args, field) for field in fieldnames(Args)]
    filename = join(splitpath(args.game_file_path), "-")[1:end-5] * "-" * join(args_fields[2:end], "-") * ".csv"
    output_file = joinpath(output_dir, filename)

    open(output_file, "w") do file
        args_heading_string = join(fieldnames(Args), ",")
        result_heading_string = join([
            "time",
            "lb_value",
            "ub_value",
            "width",
            "gamma_size",
            "upsilon_size",
            "exploration_count",
            "average_depth"
        ], ",")

        args_string = join([getfield(args, field) for field in fieldnames(Args)], ",")
        result_string = join(Any[
            time() - clock_start,
            LB_value(context),
            UB_value(context),
            width(context),
            global_gamma_size,
            global_upsilon_size,
            exploration_count,
            sum(exploration_depths) / length(exploration_depths)
        ], ",")

        write(file, args_heading_string * "," * result_heading_string * "\n")
        write(file, args_string * "," * result_string * "\n")
    end
end
