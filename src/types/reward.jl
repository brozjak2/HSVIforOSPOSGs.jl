struct Reward
    s::Int64
    a1::Int64
    a2::Int64
    v::Float64
end

Reward(params_string::String) = struct_from_ints_and_floats_string(params_string, Reward, 4)
