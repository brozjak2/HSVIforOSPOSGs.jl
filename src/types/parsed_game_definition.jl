struct ParsedGameDefinition
    game_params::GameParams

    states_names::Vector{String}
    states_partitions::Vector{Int64}

    leader_actions_names::Vector{String}
    follower_actions_names::Vector{String}
    observations_names::Vector{String}

    follower_actions::Vector{Vector{Int64}}
    leader_actions::Vector{Vector{Int64}}

    transitions::Vector{Transition}
    rewards::Vector{Reward}

    init_partition_index::Int64
    init_belief::Vector{Float64}
end

function Base.show(io::IO, parsed_game_definition::ParsedGameDefinition)
    println(io, "ParsedGameDefinition:")
    for field in fieldnames(ParsedGameDefinition)
        println(io, "$field = $(getfield(parsed_game_definition, field))")
    end
end
