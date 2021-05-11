function check_neigh_param_d(context)
    @unpack args, game = context
    @unpack epsilon, neigh_param_d = args
    @unpack discount_factor, lipschitz_delta = game

    upper_limit = (1 - discount_factor) * epsilon / (2 * lipschitz_delta)
    if !(0 <= neigh_param_d <= upper_limit)
        @warn @sprintf(
            "neighborhood parameter = %.5f is outside bounds (%.5f, %.5f)",
            neigh_param_d, 0, upper_limit
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

function compute_LB(context, partition, belief)
    if context.args.stage_game_method == :lp
        return compute_LB_primal(context, partition, belief)
    elseif context.args.stage_game_method == :qre
        return compute_LB_qre(context, partition, belief)
    end
end

function compute_UB(context, partition, belief)
    if context.args.stage_game_method == :lp
        return compute_UB_dual(context, partition, belief)
    elseif context.args.stage_game_method == :qre
        return compute_UB_qre(context, partition, belief)
    end
end
