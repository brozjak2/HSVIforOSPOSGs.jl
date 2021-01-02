mutable struct Partition
    game
    index::Int64
    states::Array{Int64,1}
    setStates::Set{Int64}
    leaderActions::Array{Int64,1}
    observations::Dict{Int64,Array{Int64,1}}

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

    setObservations = Dict{Int64,Set{Int64}}([])
    for transition in partition.game.transitions
        s, a1, a2, o, sp, prob = transition

        if s in partition.setStates
            if haskey(partition.aoTransitions, (a1, o))
                push!(partition.aoTransitions[(a1, o)], transition)
            else
                partition.aoTransitions[(a1, o)] = [transition]
            end

            if haskey(setObservations, a1)
                push!(setObservations[a1], o)
            else
                setObservations[a1] = Set([o])
            end

            if haskey(partition.transitions, s)
                push!(partition.transitions[s], transition)
            else
                partition.transitions[s] = [transition]
            end

            targetPartition = game.states[sp].partition
            if !haskey(partition.partitionTransitions, (a1, o))
                partition.partitionTransitions[(a1, o)] = targetPartition
            else
                @assert partition.partitionTransitions[(a1, o)] == targetPartition "Multipartition transition"
            end
        end
    end

    for (a1, obs) in setObservations
        partition.observations[a1] = collect(obs)
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
    nUpsilon = length(partition.upsilon)
    nStates = length(partition.states)
    UBvalueLP = Model(() -> Gurobi.Optimizer(GRB_ENV[]))
    JuMP.set_optimizer_attribute(UBvalueLP, "OutputFlag", 0)

    @variable(UBvalueLP, lambda[1:nUpsilon] >= 0) # 33f
    @variable(UBvalueLP, delta[1:nStates])
    @variable(UBvalueLP, beliefp[1:nStates])

    # 33a
    @objective(UBvalueLP, Min, sum(lambda[i] * partition.upsilon[i][2] for i in 1:nUpsilon)
                               + lipschitzdelta(partition.game) * sum(delta[sp] for sp in 1:nStates))

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

function width(partition::Partition, belief::Array{Float64,1})
    return UBValue(partition, belief) - LBValue(partition, belief)
end

function pointBasedUpdate(partition::Partition, belief::Array{Float64,1}, alpha::Array{Float64,1}, y::Float64)
    push!(partition.gamma, alpha)
    push!(partition.upsilon, (belief, y))
end
