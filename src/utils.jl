isapprox_max_norm(a, b; atol) = isapprox(a, b; atol=atol, norm=vec -> norm(vec, Inf))

function check_neighborhood(context)
    @unpack args, game = context
    @unpack epsilon, neighborhood = args
    @unpack discount_factor, lipschitz_delta = game

    upper_limit = (1 - discount_factor) * epsilon / (2 * lipschitz_delta)
    if !(0 <= neighborhood <= upper_limit)
        @warn @sprintf(
            "neighborhood parameter = %.4e is outside bounds (%d, %.4e)",
            neighborhood, 0, upper_limit
        )
    end
end

function dictarray_push_or_init!(dictarray::Dict{K,Array{V,N}}, key::K, value::V) where {K,V,N}
    if haskey(dictarray, key)
        push!(dictarray[key], value)
    else
        dictarray[key] = [value]
    end
end

function save_exploration_data(context)
    @unpack game, clock_start = context

    lb_value = LB_value(context)
    ub_value = UB_value(context)
    global_gamma_size = sum(length(p.gamma) for p in game.partitions)
    global_upsilon_size = sum(length(p.upsilon) for p in game.partitions)

    push!(context.timestamps, time() - clock_start)
    push!(context.lb_values, lb_value)
    push!(context.ub_values, ub_value)
    push!(context.gaps, ub_value - lb_value)
    push!(context.gamma_sizes, global_gamma_size)
    push!(context.upsilon_sizes, global_upsilon_size)
end
