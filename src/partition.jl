struct Partition
    index::Int64
    states::Array{Int64,1}
    leaderActions::Array{Int64,1}
    observations::Dict{Int64,Array{Int64,1}}

    gamma::Array{Array{Float64,1},1}
    upsilon::Array{Tuple{Array{Float64,1},Float64},1}

    # TODO: compute partition transitions
end

function initBounds(partition::Partition, game)
    L = Lmin(game)
    U = Umax(game)
    n = length(partition.states)

    append!(partition.gamma, [repeat([L], n)])

    for i = 1:n
        belief = zeros(n)
        belief[i] += 1
        append!(partition.upsilon, [(belief, U)])
    end
end

function LBValue(partition::Partition, belief::Array{Float64,1})
    return maximum(sum(alpha .* belief) for alpha in partition.gamma)
end

function UBValue(game, partition::Partition, belief::Array{Float64,1})
    nUpsilon = length(partition.upsilon)
    nStates = length(partition.states)
    UBvalueLP = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    JuMP.set_optimizer_attribute(UBvalueLP, "OutputFlag", 0)

    @variable(UBvalueLP, lambda[1:nUpsilon] >= 0) # 33f
    @variable(UBvalueLP, delta[1:nStates])
    @variable(UBvalueLP, beliefp[1:nStates])

    # 33a
    @objective(UBvalueLP, Min, sum(lambda[i] * partition.upsilon[i][2] for i in 1:nUpsilon)
                               + lipschitzdelta(game) * sum(delta[sp] for sp in 1:nStates))

    # 33b
    @constraint(UBvalueLP, con33b[s=1:nStates],
        sum(lambda[i] * partition.upsilon[i][1][s] for i in 1:nUpsilon) == beliefp[s])

    # 33c
    @constraint(UBvalueLP, con33c[s=1:nStates],
        delta[s] >= beliefp[s] - belief[s])

    # 33d
    @constraint(UBvalueLP, con33d[s=1:nStates],
        delta[s] >= belief[s] - beliefp[s])

    # 33e
    @constraint(UBvalueLP, con33e,
        sum(lambda[i] for i in 1:nUpsilon) == 1)

    optimize!(UBvalueLP)

    return objective_value(UBvalueLP)
end

function width(game, partition::Partition, belief::Array{Float64,1})
    return UBValue(game, partition, belief) - LBValue(partition, belief)
end

function pointBasedUpdate(partition::Partition, belief::Array{Float64,1}, alpha::Array{Float64,1}, y::Float64)
    append!(partition.gamma, [alpha])
    append!(partition.upsilon, [(belief, y)])
end
