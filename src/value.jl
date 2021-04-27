LB_value(game::Game) = LB_value(game.init_partition, game.init_belief)

function LB_value(partition::Partition, belief::Vector{Float64})
    return maximum(sum(alpha .* belief) for alpha in partition.gamma)
end

UB_value(game::Game) = UB_value(game.init_partition, game.init_belief)

function UB_value(partition::Partition, belief::Vector{Float64})
    return partition.nn(belief)[1]
end

# function UB_value(partition::Partition, belief::Vector{Float64})
#     @unpack lipschitz_delta = partition.game

#     upsilon_size = length(partition.upsilon)
#     state_count = length(partition.states)

#     UB_value_model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
#     JuMP.set_optimizer_attribute(UB_value_model, "OutputFlag", 0)

#     @variable(UB_value_model, lambda[i=1:upsilon_size] >= 0) # 33f
#     @variable(UB_value_model, delta[si=1:state_count])
#     @variable(UB_value_model, beliefp[si=1:state_count])

#     # 33a
#     @objective(UB_value_model, Min,
#         sum(lambda[i] * partition.upsilon[i][2] for i in 1:upsilon_size)
#         + lipschitz_delta * sum(delta[si] for si in 1:state_count)
#     )

#     # 33b
#     @constraint(UB_value_model, con33b[si=1:state_count],
#         sum(lambda[i] * partition.upsilon[i][1][si] for i in 1:upsilon_size) == beliefp[si])

#     # 33c
#     @constraint(UB_value_model, con33c[si=1:state_count],
#         delta[si] >= beliefp[si] - belief[si])

#     # 33d
#     @constraint(UB_value_model, con33d[si=1:state_count],
#         delta[si] >= belief[si] - beliefp[si])

#     # 33e
#     @constraint(UB_value_model, con33e,
#         sum(lambda[i] for i in 1:upsilon_size) == 1)

#     optimize!(UB_value_model)

#     return objective_value(UB_value_model)
# end

width(game::Game) = width(game.init_partition, game.init_belief)

function width(partition::Partition, belief::Vector{Float64})
    return UB_value(partition, belief) - LB_value(partition, belief)
end
