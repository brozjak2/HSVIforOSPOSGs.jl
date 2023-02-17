function compute_LB(osposg::OSPOSG, hsvi::HSVI, partition::Partition, belief::Vector{Float64})
    # The comment labels of variables, constraints and objective correspond to the equation labels in https://doi.org/10.1016/j.artint.2022.103838
    model = Model(hsvi.optimizer_factory)
    set_silent(model)

    # 27f
    @variable(model, 1.0 >= policy1[a1=partition.player1_actions] >= 0.0)
    # 27g
    @variable(model, lambda[a1=partition.player1_actions, o=partition.observations[a1], i=1:length(osposg.partitions[partition.target[a1, o]].gamma)] >= 0.0)
    @variable(model, alphavec[a1=partition.player1_actions, o=partition.observations[a1], spi=1:length(osposg.partitions[partition.target[a1, o]].states)])
    @variable(model, statevalue[s=partition.states])

    # 27a
    @objective(model, Max, sum(belief[si] * statevalue[s] for (si, s) in enumerate(partition.states)))

    # 27b
    @constraint(model, con27b[s=partition.states, a2=osposg.states[s].player2_actions],
        statevalue[s] <= sum(policy1[a1] * osposg.reward_map[s, a1, a2] for a1 in partition.player1_actions) +
                         osposg.discount * sum(get(osposg.transition_map, (s, a1, a2, o, sp), 0.0) * alphavec[a1, o, spi]
                                               for a1 in partition.player1_actions for o in partition.observations[a1] for (spi, sp) in enumerate(osposg.partitions[partition.target[a1, o]].states)
        ))

    # 27c
    @constraint(model, con27c[a1=partition.player1_actions, o=partition.observations[a1], spi=1:length(osposg.partitions[partition.target[a1, o]].states)],
        alphavec[a1, o, spi] == sum(lambda[a1, o, i] * osposg.partitions[partition.target[a1, o]].gamma[i][spi] for i in 1:length(osposg.partitions[partition.target[a1, o]].gamma)))

    # 27d
    @constraint(model, con27d[a1=partition.player1_actions, o=partition.observations[a1]],
        sum(lambda[a1, o, i] for i in 1:length(osposg.partitions[partition.target[a1, o]].gamma)) == policy1[a1])

    # 27e
    @constraint(model, con27e, sum(policy1[a1] for a1 in partition.player1_actions) == 1.0)

    optimize!(model)

    # policy of 2nd player is represented as joint probability in the LPs
    policy2_conditional = [
        [-dual.(model[:con27b]).data[s, a2] / belief[si] for a2 in osposg.states[s].player2_actions]
        for (si, s) in enumerate(partition.states)
    ]
    policy2_conditional = map.(x -> isinf(x) | isnan(x) ? zero(x) : x, policy2_conditional)

    policy1 = value.(model[:policy1]).data

    return policy1, policy2_conditional, value.(model[:statevalue]).data
end

function compute_UB(osposg::OSPOSG, hsvi::HSVI, partition::Partition, belief::Vector{Float64})
    # The comment labels of variables, constraints and objective correspond to the equation labels in https://doi.org/10.1016/j.artint.2022.103838
    model = Model(hsvi.optimizer_factory)
    set_silent(model)

    @variable(model, gamevalue)
    # 28f
    @variable(model, 1.0 >= policy2[s=partition.states, a2=osposg.states[s].player2_actions] >= 0.0)
    @variable(model, belieftransform[a1=partition.player1_actions, o=partition.observations[a1], spi=1:length(osposg.partitions[partition.target[a1, o]].states)])
    @variable(model, subgamevalue[a1=partition.player1_actions, o=partition.observations[a1]])
    # 36f
    @variable(model, lambda[a1=partition.player1_actions, o=partition.observations[a1], i=1:length(osposg.partitions[partition.target[a1, o]].upsilon)] >= 0.0)
    @variable(model, delta[a1=partition.player1_actions, o=partition.observations[a1], spi=1:length(osposg.partitions[partition.target[a1, o]].states)])
    @variable(model, beliefp[a1=partition.player1_actions, o=partition.observations[a1], spi=1:length(osposg.partitions[partition.target[a1, o]].states)])

    # 28a
    @objective(model, Min, gamevalue)

    # 28b
    @constraint(model, con28b[a1=partition.player1_actions],
        gamevalue >= sum(policy2[s, a2] * osposg.reward_map[s, a1, a2] for s in partition.states for a2 in osposg.states[s].player2_actions) +
                     osposg.discount * sum(subgamevalue[a1, o] for o in partition.observations[a1]))

    # 28d
    @constraint(model, con28d[a1=partition.player1_actions, o=partition.observations[a1], sp=osposg.partitions[partition.target[a1, o]].states],
        belieftransform[a1, o, osposg.states[sp].belief_index] >= sum(get(osposg.transition_map, (s, a1, a2, o, sp), 0.0) * policy2[s, a2]
                                                               for s in partition.states for a2 in osposg.states[s].player2_actions))

    # 28e
    @constraint(model, con28e[s=partition.states],
        sum(policy2[s, a2] for a2 in osposg.states[s].player2_actions) == belief[osposg.states[s].belief_index])

    # 36a
    @constraint(model, con36a[a1=partition.player1_actions, o=partition.observations[a1]],
        subgamevalue[a1, o] == sum(lambda[a1, o, i] * osposg.partitions[partition.target[a1, o]].upsilon[i][2] for i in 1:length(osposg.partitions[partition.target[a1, o]].upsilon)) +
                               lipschitz_delta(osposg) * sum(delta[a1, o, spi] for spi in 1:length(osposg.partitions[partition.target[a1, o]].states)))

    # 36b
    @constraint(model, con36b[a1=partition.player1_actions, o=partition.observations[a1], spi=1:length(osposg.partitions[partition.target[a1, o]].states)],
        sum(lambda[a1, o, i] * osposg.partitions[partition.target[a1, o]].upsilon[i][1][spi] for i in 1:length(osposg.partitions[partition.target[a1, o]].upsilon)) == beliefp[a1, o, spi])

    # 36c
    @constraint(model, con36c[a1=partition.player1_actions, o=partition.observations[a1], spi=1:length(osposg.partitions[partition.target[a1, o]].states)],
        delta[a1, o, spi] >= beliefp[a1, o, spi] - belieftransform[a1, o, spi])

    # 36d
    @constraint(model, con36d[a1=partition.player1_actions, o=partition.observations[a1], spi=1:length(osposg.partitions[partition.target[a1, o]].states)],
        delta[a1, o, spi] >= belieftransform[a1, o, spi] - beliefp[a1, o, spi])

    # 36e
    @constraint(model, con36e[a1=partition.player1_actions, o=partition.observations[a1]],
        sum(lambda[a1, o, i] for i in 1:length(osposg.partitions[partition.target[a1, o]].upsilon)) == sum(belieftransform[a1, o, spi] for spi in 1:length(osposg.partitions[partition.target[a1, o]].states)))

    optimize!(model)

    # policy of 2nd player is represented as joint probability in the LPs
    policy2_conditional = [
        [value.(model[:policy2]).data[s, a2] / belief[si] for a2 in osposg.states[s].player2_actions]
        for (si, s) in enumerate(partition.states)
    ]
    policy2_conditional = map.(x -> isinf(x) | isnan(x) ? zero(x) : x, policy2_conditional)

    policy1 = dual.(model[:con28b]).data

    return policy1, policy2_conditional, value(model[:gamevalue])
end
