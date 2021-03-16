struct MultiPartitionTransitionException <: Exception end

function Base.showerror(io::IO, e::MultiPartitionTransitionException)
    print(io, "Multi-partition transition encountered in game definition")
end
