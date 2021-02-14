mutable struct Partition <: AbstractPartition
    game::AbstractGame
    index::Int64
    states::Vector{Int64}
    states_set::Set{Int64}
    leader_actions::Vector{Int64}
    observations::Dict{Int64,Vector{Int64}}
    rewards::Dict{Tuple{Int64,Int64,Int64},Float64}

    gamma::Vector{Vector{Float64}}
    upsilon::Vector{Tuple{Vector{Float64},Float64}}

    transitions::Dict{Int64,Vector{Tuple{Int64,Int64,Int64,Int64,Int64,Float64}}}
    ao_pair_transitions::Dict{Tuple{Int64,Int64},Vector{Tuple{Int64,Int64,Int64,Int64,Int64,Float64}}}
    partition_transitions::Dict{Tuple{Int64,Int64},Int64}
end

Partition(game::AbstractGame, index::Int64) = Partition(
    game,
    index,
    Vector{Int64}(undef, 0),
    Set{Int64}([]),
    Vector{Int64}(undef, 0),
    Dict{Int64,Vector{Int64}}([]),
    Dict{Tuple{Int64,Int64,Int64},Float64}([]),
    Vector{Vector{Float64}}(undef, 0),
    Vector{Tuple{Vector{Float64},Float64}}(undef, 0),
    Dict{Int64,Vector{Tuple{Int64,Int64,Int64,Int64,Int64,Float64}}}([]),
    Dict{Tuple{Int64,Int64},Vector{Tuple{Int64,Int64,Int64,Int64,Int64,Float64}}}([]),
    Dict{Tuple{Int64,Int64},Int64}([])
)

function prepare(partition::Partition, game)
    partition.states_set = Set(partition.states)

    for (a1, obs) in partition.observations
        partition.observations[a1] = unique(obs)
    end
end

function LB_value(partition::Partition, belief::Vector{Float64})
    return maximum(sum(alpha .* belief) for alpha in partition.gamma)
end

function UB_value(partition::Partition, belief::Vector{Float64})
    upsilon_size = length(partition.upsilon)
    state_count = length(partition.states)

    UB_value_model = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    JuMP.set_optimizer_attribute(UB_value_model, "OutputFlag", 0)

    @variable(UB_value_model, lambda[i=1:upsilon_size] >= 0) # 33f
    @variable(UB_value_model, delta[si=1:state_count])
    @variable(UB_value_model, beliefp[si=1:state_count])

    # 33a
    @objective(UB_value_model, Min, sum(lambda[i] * partition.upsilon[i][2] for i in 1:upsilon_size)
                               + lipschitz_delta(partition.game) * sum(delta[si] for si in 1:state_count))

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
        sum(lambda[i] for i in 1:upsilon_size) == 1)

    optimize!(UB_value_model)

    return objective_value(UB_value_model)
end

function width(partition::Partition, belief::Vector{Float64})
    return UB_value(partition, belief) - LB_value(partition, belief)
end

function point_based_update(partition::Partition, belief::Vector{Float64}, alpha::Vector{Float64}, y::Float64)
    push!(partition.gamma, alpha)
    push!(partition.upsilon, (belief, y))
end
