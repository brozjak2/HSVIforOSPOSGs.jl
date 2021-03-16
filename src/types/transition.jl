struct Transition
    s::Int64 # state
    a1::Int64 # leader action
    a2::Int64 # follower action
    o::Int64 # observation
    sp::Int64 # next state
    p::Float64 # probability
end

function Transition(transition_string::String)
    transition_strings = split(transition_string, ' ')
    indexes = map((x) -> parse(Int64, x) + 1, transition_strings[1:5])
    probability = parse(Float64, transition_strings[6])

    return Transition(indexes..., probability)
end

function Base.show(io::IO, transition::Transition)
    println(io, "Transition:")
    for field in fieldnames(Transition)
        println(io, "  $field = $(getfield(transition, field))")
    end
end
