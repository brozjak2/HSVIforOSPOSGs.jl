struct MultiPartitionTransitionException <: Exception end
struct InvalidArgumentValue{T} <: Exception
    argument::String
    value::T
end

function Base.showerror(io::IO, e::MultiPartitionTransitionException)
    print(io, "Multi-partition transition encountered in game definition")
end

function Base.showerror(io::IO, e::InvalidArgumentValue)
    print(io, "Invalid value '$(e.value)' specified for argument '$(e.argument)'")
end
