struct Transition
    s::Int64
    a1::Int64
    a2::Int64
    o::Int64
    sp::Int64
    p::Float64
end

Transition(params_string::String) = struct_from_ints_and_floats_string(params_string, Transition, 6)
