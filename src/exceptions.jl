struct MultiPartitionTransitionException <: Exception end

Base.showerror(io::IO, e::MultiPartitionTransitionException) = print(io, "Multi-partition transition encountered in game definition")
