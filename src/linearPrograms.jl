function computeLBprimal(partition::Partition, belief::Array{Float64,1})
    game = partition.game

    LBprimal = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    JuMP.set_optimizer_attribute(LBprimal, "OutputFlag", 0)

    @variable(LBprimal, 1.0 >= policy1[a1=partition.leaderActions] >= 0.0) # 27f
    @variable(LBprimal, lambda[a1=partition.leaderActions, o=partition.observations[a1], i=1:length(game.partitions[partition.partitionTransitions[(a1, o)]].gamma)] >= 0.0) # 27g
    @variable(LBprimal, alphavec[a1=partition.leaderActions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partitionTransitions[(a1, o)]].states)])
    @variable(LBprimal, statevalue[s=partition.states])

    # 27a
    @objective(LBprimal, Max, sum(belief[si] * statevalue[s] for (si, s) in enumerate(partition.states)))

    # 27b
    @constraint(LBprimal, con27b[s=partition.states, a2=game.states[s].followerActions],
        statevalue[s] <= sum(policy1[a1] * partition.rewards[s, a1, a2] for a1 in partition.leaderActions)
                         + game.disc * sum(get(game.transitionMap, (s, a1, a2, o, sp), 0.0) * alphavec[a1, o, spi]
                                           for a1 in partition.leaderActions for o in partition.observations[a1] for (spi, sp) in enumerate(game.partitions[partition.partitionTransitions[(a1, o)]].states)))

    # 27c
    @constraint(LBprimal, con27c[a1=partition.leaderActions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partitionTransitions[(a1, o)]].states)],
           alphavec[a1, o, spi] == sum(lambda[a1, o, i] * game.partitions[partition.partitionTransitions[(a1, o)]].gamma[i][spi] for i in 1:length(game.partitions[partition.partitionTransitions[(a1, o)]].gamma)))

    # 27d
    @constraint(LBprimal, con27d[a1=partition.leaderActions, o=partition.observations[a1]],
        sum(lambda[a1, o, i] for i in 1:length(game.partitions[partition.partitionTransitions[(a1, o)]].gamma)) == policy1[a1])

    # 27e
    @constraint(LBprimal, con27e,
        sum(policy1[a1] for a1 in partition.leaderActions) == 1)

    optimize!(LBprimal)

    # policy of 2nd player is represented as joint probability in the LPs
    policy2conditional = Dict((s, a2) => - val / belief[game.states[s].inPartitionIndex] for ((s, a2), val) in dual.(LBprimal[:con27b]).data)
    policy2conditional = Dict((s, a2) => isinf(val) | isnan(val) ? zero(val) : val for ((s, a2), val) in policy2conditional)

    policy1 = Dict(a1 => value.(LBprimal[:policy1]).data[i] for (i, a1) in enumerate(partition.leaderActions))

    return policy1, policy2conditional, value.(LBprimal[:statevalue]).data
end

function computeUBdual(partition::Partition, belief::Array{Float64,1})
    game = partition.game

    UBdual = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    JuMP.set_optimizer_attribute(UBdual, "OutputFlag", 0)

    @variable(UBdual, gamevalue)
    @variable(UBdual, 1.0 >= policy2[s=partition.states, a2=game.states[s].followerActions] >= 0.0) # 28f
    @variable(UBdual, belieftransform[a1=partition.leaderActions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partitionTransitions[(a1, o)]].states)])
    @variable(UBdual, subgamevalue[a1=partition.leaderActions, o=partition.observations[a1]])
    @variable(UBdual, lambda[a1=partition.leaderActions, o=partition.observations[a1], i=1:length(game.partitions[partition.partitionTransitions[(a1, o)]].upsilon)] >= 0.0) # 36f
    @variable(UBdual, delta[a1=partition.leaderActions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partitionTransitions[(a1, o)]].states)])
    @variable(UBdual, beliefp[a1=partition.leaderActions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partitionTransitions[(a1, o)]].states)])

    # 28a
    @objective(UBdual, Min, gamevalue)

    # 28b
    @constraint(UBdual, con28b[a1=partition.leaderActions],
        gamevalue >= sum(policy2[s, a2] * partition.rewards[s, a1, a2] for s in partition.states for a2 in game.states[s].followerActions)
                     + game.disc * sum(subgamevalue[a1, o] for o in partition.observations[a1]))

    # 28d
    @constraint(UBdual, con28d[a1=partition.leaderActions, o=partition.observations[a1], sp=game.partitions[partition.partitionTransitions[(a1, o)]].states],
        belieftransform[a1, o, game.states[sp].inPartitionIndex] >= sum(get(game.transitionMap, (s, a1, a2, o, sp), 0.0) * policy2[s, a2]
                                       for s in partition.states for a2 in game.states[s].followerActions))

    # 28e
    @constraint(UBdual, con28e[s=partition.states],
        sum(policy2[s, a2] for a2 in game.states[s].followerActions) == belief[game.states[s].inPartitionIndex])

    # 36a
    @constraint(UBdual, con36a[a1=partition.leaderActions, o=partition.observations[a1]],
        subgamevalue[a1, o] == sum(lambda[a1, o, i] * game.partitions[partition.partitionTransitions[(a1, o)]].upsilon[i][2] for i in 1:length(game.partitions[partition.partitionTransitions[(a1, o)]].upsilon))
                               + lipschitzdelta(game) * sum(delta[a1, o, spi] for spi in 1:length(game.partitions[partition.partitionTransitions[(a1, o)]].states)))

    # 36b
    @constraint(UBdual, con36b[a1=partition.leaderActions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partitionTransitions[(a1, o)]].states)],
        sum(lambda[a1, o, i] * game.partitions[partition.partitionTransitions[(a1, o)]].upsilon[i][1][spi] for i in 1:length(game.partitions[partition.partitionTransitions[(a1, o)]].upsilon)) == beliefp[a1, o, spi])

    # 36c
    @constraint(UBdual, con36c[a1=partition.leaderActions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partitionTransitions[(a1, o)]].states)],
        delta[a1, o, spi] >= beliefp[a1, o, spi] - belieftransform[a1, o, spi])

    # 36d
    @constraint(UBdual, con36d[a1=partition.leaderActions, o=partition.observations[a1], spi=1:length(game.partitions[partition.partitionTransitions[(a1, o)]].states)],
        delta[a1, o, spi] >= belieftransform[a1, o, spi] - beliefp[a1, o, spi])

    # 36e
    @constraint(UBdual, con36e[a1=partition.leaderActions, o=partition.observations[a1]],
        sum(lambda[a1, o, i] for i in 1:length(game.partitions[partition.partitionTransitions[(a1, o)]].upsilon)) == sum(belieftransform[a1, o, spi] for spi in 1:length(game.partitions[partition.partitionTransitions[(a1, o)]].states)))

    optimize!(UBdual)

    # policy of 2nd player is represented as joint probability in the LPs
    policy2conditional = Dict((s, a2) => val / belief[game.states[s].inPartitionIndex] for ((s, a2), val) in value.(UBdual[:policy2]).data)
    policy2conditional = Dict((s, a2) => isinf(val) | isnan(val) ? zero(val) : val for ((s, a2), val) in policy2conditional)

    policy1 = Dict(a1 => dual.(UBdual[:con28b]).data[i] for (i, a1) in enumerate(partition.leaderActions))

    return policy1, policy2conditional, value(UBdual[:gamevalue])
end
