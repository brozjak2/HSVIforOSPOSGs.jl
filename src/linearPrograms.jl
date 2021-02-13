function compute_LB_primal(partition::Partition, belief::Array{Float64,1})
    game = partition.game

    LB_primal = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    JuMP.set_optimizer_attribute(LB_primal, "OutputFlag", 0)

    @variable(LB_primal, 1.0 >= policy1[a1=partition.leader_actions] >= 0.0) # 27f
    @variable(LB_primal, lambda[a1=partition.leader_actions, o=partition.observations[a1], i=1:length(game.partitions[partition.partition_transitions[(a1, o)]].gamma)] >= 0.0) # 27g
    @variable(LB_primal, alphavec[a1=partition.leader_actions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partition_transitions[(a1, o)]].states)])
    @variable(LB_primal, statevalue[s=partition.states])

    # 27a
    @objective(LB_primal, Max, sum(belief[si] * statevalue[s] for (si, s) in enumerate(partition.states)))

    # 27b
    @constraint(LB_primal, con27b[s=partition.states, a2=game.states[s].follower_actions],
        statevalue[s] <= sum(policy1[a1] * partition.rewards[s, a1, a2] for a1 in partition.leader_actions)
                         + game.disc * sum(get(game.transition_map, (s, a1, a2, o, sp), 0.0) * alphavec[a1, o, spi]
                                           for a1 in partition.leader_actions for o in partition.observations[a1] for (spi, sp) in enumerate(game.partitions[partition.partition_transitions[(a1, o)]].states)))

    # 27c
    @constraint(LB_primal, con27c[a1=partition.leader_actions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partition_transitions[(a1, o)]].states)],
           alphavec[a1, o, spi] == sum(lambda[a1, o, i] * game.partitions[partition.partition_transitions[(a1, o)]].gamma[i][spi] for i in 1:length(game.partitions[partition.partition_transitions[(a1, o)]].gamma)))

    # 27d
    @constraint(LB_primal, con27d[a1=partition.leader_actions, o=partition.observations[a1]],
        sum(lambda[a1, o, i] for i in 1:length(game.partitions[partition.partition_transitions[(a1, o)]].gamma)) == policy1[a1])

    # 27e
    @constraint(LB_primal, con27e,
        sum(policy1[a1] for a1 in partition.leader_actions) == 1)

    optimize!(LB_primal)

    # policy of 2nd player is represented as joint probability in the LPs
    policy2_conditional = Dict((s, a2) => - val / belief[game.states[s].in_partition_index] for ((s, a2), val) in dual.(LB_primal[:con27b]).data)
    policy2_conditional = Dict((s, a2) => isinf(val) | isnan(val) ? zero(val) : val for ((s, a2), val) in policy2_conditional)

    policy1 = Dict(a1 => value.(LB_primal[:policy1]).data[i] for (i, a1) in enumerate(partition.leader_actions))

    return policy1, policy2_conditional, value.(LB_primal[:statevalue]).data
end

function compute_UB_dual(partition::Partition, belief::Array{Float64,1})
    game = partition.game

    UB_dual = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    JuMP.set_optimizer_attribute(UB_dual, "OutputFlag", 0)

    @variable(UB_dual, gamevalue)
    @variable(UB_dual, 1.0 >= policy2[s=partition.states, a2=game.states[s].follower_actions] >= 0.0) # 28f
    @variable(UB_dual, belieftransform[a1=partition.leader_actions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partition_transitions[(a1, o)]].states)])
    @variable(UB_dual, subgamevalue[a1=partition.leader_actions, o=partition.observations[a1]])
    @variable(UB_dual, lambda[a1=partition.leader_actions, o=partition.observations[a1], i=1:length(game.partitions[partition.partition_transitions[(a1, o)]].upsilon)] >= 0.0) # 36f
    @variable(UB_dual, delta[a1=partition.leader_actions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partition_transitions[(a1, o)]].states)])
    @variable(UB_dual, beliefp[a1=partition.leader_actions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partition_transitions[(a1, o)]].states)])

    # 28a
    @objective(UB_dual, Min, gamevalue)

    # 28b
    @constraint(UB_dual, con28b[a1=partition.leader_actions],
        gamevalue >= sum(policy2[s, a2] * partition.rewards[s, a1, a2] for s in partition.states for a2 in game.states[s].follower_actions)
                     + game.disc * sum(subgamevalue[a1, o] for o in partition.observations[a1]))

    # 28d
    @constraint(UB_dual, con28d[a1=partition.leader_actions, o=partition.observations[a1], sp=game.partitions[partition.partition_transitions[(a1, o)]].states],
        belieftransform[a1, o, game.states[sp].in_partition_index] >= sum(get(game.transition_map, (s, a1, a2, o, sp), 0.0) * policy2[s, a2]
                                       for s in partition.states for a2 in game.states[s].follower_actions))

    # 28e
    @constraint(UB_dual, con28e[s=partition.states],
        sum(policy2[s, a2] for a2 in game.states[s].follower_actions) == belief[game.states[s].in_partition_index])

    # 36a
    @constraint(UB_dual, con36a[a1=partition.leader_actions, o=partition.observations[a1]],
        subgamevalue[a1, o] == sum(lambda[a1, o, i] * game.partitions[partition.partition_transitions[(a1, o)]].upsilon[i][2] for i in 1:length(game.partitions[partition.partition_transitions[(a1, o)]].upsilon))
                               + lipschitz_delta(game) * sum(delta[a1, o, spi] for spi in 1:length(game.partitions[partition.partition_transitions[(a1, o)]].states)))

    # 36b
    @constraint(UB_dual, con36b[a1=partition.leader_actions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partition_transitions[(a1, o)]].states)],
        sum(lambda[a1, o, i] * game.partitions[partition.partition_transitions[(a1, o)]].upsilon[i][1][spi] for i in 1:length(game.partitions[partition.partition_transitions[(a1, o)]].upsilon)) == beliefp[a1, o, spi])

    # 36c
    @constraint(UB_dual, con36c[a1=partition.leader_actions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partition_transitions[(a1, o)]].states)],
        delta[a1, o, spi] >= beliefp[a1, o, spi] - belieftransform[a1, o, spi])

    # 36d
    @constraint(UB_dual, con36d[a1=partition.leader_actions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partition_transitions[(a1, o)]].states)],
        delta[a1, o, spi] >= belieftransform[a1, o, spi] - beliefp[a1, o, spi])

    # 36e
    @constraint(UB_dual, con36e[a1=partition.leader_actions, o=partition.observations[a1]],
        sum(lambda[a1, o, i] for i in 1:length(game.partitions[partition.partition_transitions[(a1, o)]].upsilon)) == sum(belieftransform[a1, o, spi] for spi in 1:length(game.partitions[partition.partition_transitions[(a1, o)]].states)))

    optimize!(UB_dual)

    # policy of 2nd player is represented as joint probability in the LPs
    policy2_conditional = Dict((s, a2) => val / belief[game.states[s].in_partition_index] for ((s, a2), val) in value.(UB_dual[:policy2]).data)
    policy2_conditional = Dict((s, a2) => isinf(val) | isnan(val) ? zero(val) : val for ((s, a2), val) in policy2_conditional)

    policy1 = Dict(a1 => dual.(UB_dual[:con28b]).data[i] for (i, a1) in enumerate(partition.leader_actions))

    return policy1, policy2_conditional, value(UB_dual[:gamevalue])
end
