flush_logs() = flush(global_logger().stream)

function log_initial(context)
    @debug context.args
    @debug context.game
    flush_logs()
end

function log_presolveUB(context, delta, presolve_time)
    @unpack presolve_epsilon, presolve_time_limit = context.args

    if delta <= presolve_epsilon
        @debug @sprintf(
                "presolve_LB reached desired precision %s in %7.3fs",
                presolve_epsilon, presolve_time
            )
    else
        @debug @sprintf("presolve_LB reached time limit %7.3fs", presolve_time_limit)
    end

    @info @sprintf(
            "%7.3fs\tpresolveUB\t%+7.5f",
            time() - context.clock_start,
            UB_value(context)
        )
    flush_logs()
end

function log_presolveLB(context, delta, presolve_time)
    @unpack presolve_epsilon, presolve_time_limit = context.args

    if delta <= presolve_epsilon
        @debug @sprintf(
                "presolve_LB reached desired precision %s in %7.3fs",
                presolve_epsilon, presolve_time
            )
    else
        @debug @sprintf("presolve_LB reached time limit %7.3fs", presolve_time_limit)
    end

    @info @sprintf(
            "%7.3fs\tpresolveLB\t%+7.5f",
            time() - context.clock_start,
            LB_value(context)
        )
    flush_logs()
end

function log_solve(context)
    success = context.gaps[end] <= context.args.epsilon

    @info @sprintf(
        "%7.3fs\t%+7.5f\t%+7.5f\t%+7.5f\t%s",
        context.timestamps[end],
        context.lb_values[end],
        context.ub_values[end],
        context.gaps[end],
        success ? "Converged" : "Did not converge"
    )
    flush_logs()
end

function log_progress(context)
    @info @sprintf(
        "%4d:\t%7.3fs\t%+7.5f\t%+7.5f\t%+7.5f\t%5d\t%5d",
        context.exploration_count,
        context.timestamps[end],
        context.lb_values[end],
        context.ub_values[end],
        context.gaps[end],
        context.gamma_sizes[end],
        context.upsilon_sizes[end]
    )
    flush_logs()
end

function log_depth(context)
    @info "\texploration_depth = $(context.exploration_depths[end])"
    flush_logs()
end
