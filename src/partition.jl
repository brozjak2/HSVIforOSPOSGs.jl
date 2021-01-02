mutable struct Partition
    game
    index::Int64
    states::Array{Int64,1}
    setStates::Set{Int64}
    leaderActions::Array{Int64,1}
    observations::Dict{Int64,Array{Int64,1}}
    rewards::Dict{Tuple{Int64,Int64,Int64},Float64}

    gamma::Array{Array{Float64,1},1}
    upsilon::Array{Tuple{Array{Float64,1},Float64},1}

    transitions::Dict{Int64,Array{Tuple{Int64,Int64,Int64,Int64,Int64,Float64},1}}
    aoTransitions::Dict{Tuple{Int64,Int64},Array{Tuple{Int64,Int64,Int64,Int64,Int64,Float64},1}}
    partitionTransitions::Dict{Tuple{Int64,Int64},Int64}

    Partition(index) = new(
        nothing,
        index,
        Array{Int64,1}(undef, 0),
        Set{Int64}([]),
        Array{Int64,1}(undef, 0),
        Dict{Int64,Array{Int64,1}}([]),
        Dict{Tuple{Int64,Int64,Int64},Float64}([]),
        Array{Array{Float64,1},1}(undef, 0),
        Array{Tuple{Array{Float64,1},Float64},1}(undef, 0),
        Dict{Int64,Array{Tuple{Int64,Int64,Int64,Int64,Int64,Float64},1}}([]),
        Dict{Tuple{Int64,Int64},Array{Tuple{Int64,Int64,Int64,Int64,Int64,Float64},1}}([]),
        Dict{Tuple{Int64,Int64},Int64}([])
    )
end

function prepare(partition::Partition, game)
    partition.game = game
    partition.setStates = Set(partition.states)

    for (a1, obs) in partition.observations
        partition.observations[a1] = unique(obs)
    end
end

function initBounds(partition::Partition)
    L = Lmin(partition.game)
    U = Umax(partition.game)
    n = length(partition.states)

    push!(partition.gamma, repeat([L], n))

    for i = 1:n
        belief = zeros(n)
        belief[i] += 1
        push!(partition.upsilon, (belief, U))
    end
end

function LBValue(partition::Partition, belief::Array{Float64,1})
    return maximum(sum(alpha .* belief) for alpha in partition.gamma)
end

function UBValue(partition::Partition, belief::Array{Float64,1})
    lenUpsilon = length(partition.upsilon)
    nStates = length(partition.states)

    UBvalueLP = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    JuMP.set_optimizer_attribute(UBvalueLP, "OutputFlag", 0)

    @variable(UBvalueLP, lambda[i=1:lenUpsilon] >= 0) # 33f
    @variable(UBvalueLP, delta[si=1:nStates])
    @variable(UBvalueLP, beliefp[si=1:nStates])

    # 33a
    @objective(UBvalueLP, Min, sum(lambda[i] * partition.upsilon[i][2] for i in 1:lenUpsilon)
                               + lipschitzdelta(partition.game) * sum(delta[si] for si in 1:nStates))

    # 33b
    @constraint(UBvalueLP, con33b[si=1:nStates],
        sum(lambda[i] * partition.upsilon[i][1][si] for i in 1:lenUpsilon) == beliefp[si])

    # 33c
    @constraint(UBvalueLP, con33c[si=1:nStates],
        delta[si] >= beliefp[si] - belief[si])

    # 33d
    @constraint(UBvalueLP, con33d[si=1:nStates],
        delta[si] >= belief[si] - beliefp[si])

    # 33e
    @constraint(UBvalueLP, con33e,
        sum(lambda[i] for i in 1:lenUpsilon) == 1)

    optimize!(UBvalueLP)

    return objective_value(UBvalueLP)
end

function width(partition::Partition, belief::Array{Float64,1})
    return UBValue(partition, belief) - LBValue(partition, belief)
end

function pointBasedUpdate(partition::Partition, belief::Array{Float64,1}, alpha::Array{Float64,1}, y::Float64)
    push!(partition.gamma, alpha)
    push!(partition.upsilon, (belief, y))
end
