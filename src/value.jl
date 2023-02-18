"""
    LB_value(osposg::OSPOSG, hsvi::HSVI)
    LB_value(::OSPOSG, ::HSVI, partition::Partition, belief::Vector{Float64})

Compute LB value in initial partition and belief of `osposg` or in specified `partition` and `belief` using dot products of α-vectors with given belief.
"""
LB_value(osposg::OSPOSG, hsvi::HSVI) = LB_value(osposg, hsvi, osposg.partitions[osposg.initial_partition], osposg.initial_belief)
LB_value(::OSPOSG, ::HSVI, partition::Partition, belief::Vector{Float64}) = maximum(dot(alpha, belief) for alpha in partition.gamma)

"""
    UB_value(osposg::OSPOSG, hsvi::HSVI)
    UB_value(::OSPOSG, ::HSVI, partition::Partition, belief::Vector{Float64})

Compute UB value in initial partition and belief of `osposg` or in specified `partition` and `belief` using convex hull linear program.
"""
UB_value(osposg::OSPOSG, hsvi::HSVI) = UB_value(osposg, hsvi, osposg.partitions[osposg.initial_partition], osposg.initial_belief)
function UB_value(osposg::OSPOSG, hsvi::HSVI, partition::Partition, belief::Vector{Float64})
    upsilon_size = length(partition.upsilon)
    state_count = length(partition.states)

    # The comment labels of variables, constraints and objective correspond to the equation labels in https://doi.org/10.1016/j.artint.2022.103838
    model = Model(hsvi.optimizer_factory)
    set_silent(model)

    # 33f
    @variable(model, lambda[i=1:upsilon_size] >= 0.0)
    @variable(model, delta[si=1:state_count])
    @variable(model, beliefp[si=1:state_count])

    # 33a
    @objective(model, Min,
        sum(lambda[i] * partition.upsilon[i][2] for i in 1:upsilon_size) +
        lipschitz_delta(osposg) * sum(delta[si] for si in 1:state_count)
    )

    # 33b
    @constraint(model, con33b[si=1:state_count],
        sum(lambda[i] * partition.upsilon[i][1][si] for i in 1:upsilon_size) == beliefp[si])

    # 33c
    @constraint(model, con33c[si=1:state_count], delta[si] >= beliefp[si] - belief[si])

    # 33d
    @constraint(model, con33d[si=1:state_count], delta[si] >= belief[si] - beliefp[si])

    # 33e
    @constraint(model, con33e, sum(lambda[i] for i in 1:upsilon_size) == 1.0)

    optimize!(model)

    return objective_value(model)
end

width(osposg::OSPOSG, hsvi::HSVI) = UB_value(osposg, hsvi) - LB_value(osposg, hsvi)

function width(osposg::OSPOSG, hsvi::HSVI, partition::Partition, belief::Vector{Float64})
    return UB_value(osposg, hsvi, partition, belief) - LB_value(osposg, hsvi, partition, belief)
end
