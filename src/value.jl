LB_value(context) = LB_value(context.game.init_partition, context.game.init_belief, context)

function LB_value(partition, belief, context)
    return maximum(dot(alpha, belief) for alpha in partition.gamma)
end

UB_value(context) = UB_value(context.game.init_partition, context.game.init_belief, context)

function UB_value(partition, belief, context)
    @unpack ub_value_method = context.args

    if ub_value_method == :lp
        return UB_value_lp(partition, belief, context)
    elseif ub_value_method == :nn
        return UB_value_nn(partition, belief)
    else
        throw(InvalidArgumentValue("ub_value_method", ub_value_method))
    end
end

function UB_value_nn(partition, belief)
    return partition.nn(belief)[1]
end

function UB_value_lp(partition, belief, context)
    @unpack lipschitz_delta = context.game

    upsilon_size = length(partition.upsilon)
    state_count = length(partition.states)

    UB_value_model = Model(GLPK.Optimizer)
    JuMP.set_optimizer_attribute(UB_value_model, "msg_lev", GLPK.GLP_MSG_OFF)

    @variable(UB_value_model, lambda[i=1:upsilon_size] >= 0.0) # 33f
    @variable(UB_value_model, delta[si=1:state_count])
    @variable(UB_value_model, beliefp[si=1:state_count])

    # 33a
    @objective(UB_value_model, Min,
        sum(lambda[i] * partition.upsilon[i][2] for i in 1:upsilon_size)
        + lipschitz_delta * sum(delta[si] for si in 1:state_count)
    )

    # 33b
    @constraint(UB_value_model, con33b[si=1:state_count],
        sum(lambda[i] * partition.upsilon[i][1][si] for i in 1:upsilon_size) == beliefp[si])

    # 33c
    @constraint(UB_value_model, con33c[si=1:state_count],
        delta[si] >= beliefp[si] - belief[si])

    # 33d
    @constraint(UB_value_model, con33d[si=1:state_count],
        delta[si] >= belief[si] - beliefp[si])

    # 33e
    @constraint(UB_value_model, con33e,
        sum(lambda[i] for i in 1:upsilon_size) == 1.0)

    optimize!(UB_value_model)

    return objective_value(UB_value_model)
end

width(context) = width(context.game.init_partition, context.game.init_belief, context)

function width(partition, belief, context)
    return UB_value(partition, belief, context) - LB_value(partition, belief, context)
end
