struct Transition
    s::Int64 # state
    a1::Int64 # leader action
    a2::Int64 # follower action
    o::Int64 # observation
    sp::Int64 # next state
    p::Float64 # probability
end

Transition(params_string::String) = struct_from_indexes_and_float_string(params_string, Transition)
