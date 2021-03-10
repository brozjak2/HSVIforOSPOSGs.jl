function struct_from_indexes_and_float_string(params_string::String, type::Type)
    params_strings = split(params_string, ' ')
    index_params = map((x) -> parse(Int64, x) + 1, params_strings[1:end - 1])
    float_param = parse(Float64, params_strings[end])

    return type(index_params..., float_param)
end

function dictarray_push_or_init(dictarray::Dict{K,Array{V,N}}, key::K, value::V) where {K,V,N}
    if haskey(dictarray, key)
        push!(dictarray[key], value)
    else
        dictarray[key] = [value]
    end
end

function show_struct(io::IO, instance::T) where {T}
    println(io, "$(typeof(instance).name):")
    for field in fieldnames(typeof(instance))
        println(io, " $field = $(getfield(instance, field))")
    end
end

function load(game_file_path::String)
    parsed_game_definition = open(game_file_path) do file
        return ParsedGameDefinition(file)
    end

    return Game(parsed_game_definition)
end

function point_based_update(partition::Partition, belief::Vector{Float64}, alpha::Vector{Float64}, y::Float64)
    push!(partition.gamma, alpha)
    push!(partition.upsilon, (belief, y))
end

function select_ao_pair(partition::Partition, belief::Vector{Float64}, policy1, policy2, rho::Float64)
    weighted_excess_gaps = Dict((a1, o) => weighted_excess(partition, belief, policy1, policy2, a1, o, rho) for a1 in partition.leader_actions for o in partition.observations[a1])

    _, (a1, o) = findmax(weighted_excess_gaps)

    return a1, o
end

function weighted_excess(partition::Partition, belief::Vector{Float64}, policy1,
    policy2, a1::Int64, o::Int64, rho::Float64
)
    @unpack partitions = partition.game

    next_belief = get_next_belief(partition, belief, policy1, policy2, a1, o)
    next_partition = partitions[partition.partition_transitions[(a1, o)]]

    return (ao_pair_probability(partition, belief, policy1, policy2, a1, o)
           * excess(next_partition, next_belief, rho))
end

function get_next_belief(partition::Partition, belief::Vector{Float64},
    policy1, policy2, a1::Int64, o::Int64
)
    game = partition.game
    @unpack states, partitions, sm = game

    ao_inverse_prob = 1 / ao_pair_probability(partition, belief, policy1, policy2, a1, o)
    ao_inverse_prob = isinf(ao_inverse_prob) ? zero(ao_inverse_prob) : ao_inverse_prob # handle division by zero

    target_partition = partitions[partition.partition_transitions[(a1, o)]]
    target_belief = zeros(length(target_partition.states))
    for t in partition.ao_pair_transitions[(a1, o)]
        target_belief[sm[t.sp]] += belief[sm[t.s]] * policy1[partition.a1m[a1]] * policy2[sm[t.s]][states[t.s].a2m[t.a2]] * t.p
    end

    return ao_inverse_prob * target_belief
end

function ao_pair_probability(partition::Partition, belief::Vector{Float64},
    policy1, policy2, a1::Int64, o::Int64
)
    game = partition.game
    @unpack states, sm = game

    return sum(belief[sm[t.s]] * policy1[partition.a1m[a1]] * policy2[sm[t.s]][states[t.s].a2m[t.a2]] * t.p
               for t in partition.ao_pair_transitions[(a1, o)])
end

function excess(partition::Partition, belief::Vector{Float64}, rho::Float64)
    return width(partition, belief) - rho
end

function next_rho(prev_rho::Float64, game::Game, neigh_param_d::Float64)
    return (prev_rho - 2 * game.lipschitz_delta * neigh_param_d) / game.discount_factor
end
