struct Reward
    s::Int64 # state
    a1::Int64 # leader action
    a2::Int64 # follower action
    r::Float64 # reward
end

function Reward(reward_string::String)
    reward_strings = split(reward_string, ' ')
    indexes = map((x) -> parse(Int64, x) + 1, reward_strings[1:3])
    reward = parse(Float64, reward_strings[4])

    return Reward(indexes..., reward)
end

function Base.show(io::IO, reward::Reward)
    println(io, "Reward:")
    for field in fieldnames(Reward)
        println(io, "  $field = $(getfield(reward, field))")
    end
end
