struct State
    name::String

    index::Int64
    belief_index::Int64
    partition::Int64

    follower_actions::Vector{Int64}
    immediate_rewards::Vector{Vector{Float64}}
end

function State(index::Int64, belief_index::Int64)
