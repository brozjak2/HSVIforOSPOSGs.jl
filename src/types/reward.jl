struct Reward
    s::Int64 # state
    a1::Int64 # leader action
    a2::Int64 # follower action
    r::Float64 # reward
end

Reward(params_string::String) = struct_from_indexes_and_float_string(params_string, Reward)

Base.show(io::IO, reward::Reward) = show_struct(io, reward)
