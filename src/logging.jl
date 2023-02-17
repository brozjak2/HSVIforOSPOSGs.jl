flush_logs() = flush(global_logger().stream)

function update_recorder(osposg::OSPOSG, hsvi::HSVI, recorder::Recorder)
    recorder.exploration_count += 1
    timestamp = time() - recorder.clock_start
    lb_value = LB_value(osposg, hsvi)
    ub_value = UB_value(osposg, hsvi)
    global_gamma_size = sum(length(partition.gamma) for partition in osposg.partitions)
    global_upsilon_size = sum(length(partition.upsilon) for partition in osposg.partitions)

    push!(recorder.timestamps, timestamp)
    push!(recorder.lb_values, lb_value)
    push!(recorder.ub_values, ub_value)
    push!(recorder.gaps, ub_value - lb_value)
    push!(recorder.gamma_sizes, global_gamma_size)
    push!(recorder.upsilon_sizes, global_upsilon_size)
end

function log_presolveLB(osposg::OSPOSG, hsvi::HSVI, delta::Float64, presolve_time::Float64, recorder::Recorder)
    if delta <= hsvi.presolve_epsilon
        @debug @sprintf(
            "presolve_LB achieved desired precision %f in %.2fs",
            hsvi.presolve_epsilon, presolve_time
        )
    else
        @debug @sprintf(
            "presolve_LB reached time limit %.2fs and achieved delta %f",
            hsvi.presolve_time_limit, delta
        )
    end

    @info @sprintf(
        "%7.2fs presolve_LB %+10.5f",
        time() - recorder.clock_start,
        LB_value(osposg, hsvi)
    )

    flush_logs()
end

function log_presolveUB(osposg::OSPOSG, hsvi::HSVI, delta::Float64, presolve_time::Float64, recorder::Recorder)
    if delta <= hsvi.presolve_epsilon
        @debug @sprintf(
            "presolve_UB achieved desired precision %f in %.2fs",
            hsvi.presolve_epsilon, presolve_time
        )
    else
        @debug @sprintf(
            "presolve_UB reached time limit %.2fs and achieved delta %f",
            hsvi.presolve_time_limit, delta
        )
    end

    @info @sprintf(
        "%7.2fs presolve_UB %+10.5f",
        time() - recorder.clock_start,
        UB_value(osposg, hsvi)
    )

    flush_logs()
end

function log_initial(osposg::OSPOSG, hsvi::HSVI)
    @info osposg
    @info hsvi

    flush_logs()
end

function log_start(osposg::OSPOSG, hsvi::HSVI, recorder::Recorder)
    timestamp = time() - recorder.clock_start
    lb_value = LB_value(osposg, hsvi)
    ub_value = UB_value(osposg, hsvi)
    global_gamma_size = sum(length(partition.gamma) for partition in osposg.partitions)
    global_upsilon_size = sum(length(partition.upsilon) for partition in osposg.partitions)

    @info @sprintf(
        "%5s %9s %10s %10s %10s %6s %6s %6s",
        "exp",
        "time",
        "LB",
        "UB",
        "gap",
        "Γ",
        "Υ",
        "depth"
    )

    @info @sprintf(
        "%5d: %7.2fs %+10.5f %+10.5f %+10.5f %6d %6d",
        0,
        timestamp,
        lb_value,
        ub_value,
        ub_value - lb_value,
        global_gamma_size,
        global_upsilon_size
    )

    flush_logs()
end

function log_progress(recorder::Recorder)
    @info @sprintf(
        "%5d: %7.2fs %+10.5f %+10.5f %+10.5f %6d %6d %6d",
        recorder.exploration_count,
        recorder.timestamps[end],
        recorder.lb_values[end],
        recorder.ub_values[end],
        recorder.gaps[end],
        recorder.gamma_sizes[end],
        recorder.upsilon_sizes[end],
        recorder.exploration_depths[end]
    )

    flush_logs()
end

function log_end(recorder::Recorder, epsilon::Float64)
    if recorder.exploration_count > 0
        @info @sprintf(
            "%5s: %8.3fs %+10.5f %+10.5f %+10.5f",
            recorder.gaps[end] <= epsilon ? "OK" : "FAIL",
            recorder.timestamps[end],
            recorder.lb_values[end],
            recorder.ub_values[end],
            recorder.gaps[end]
        )
    else
        @info "Game was solved during presolve"
    end

    flush_logs()
end
