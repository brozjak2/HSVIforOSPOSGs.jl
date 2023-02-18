struct MultiPartitionTransitionException <: Exception end

function Base.showerror(io::IO, ::MultiPartitionTransitionException)
    print(io, "Multi-partition transition encountered in game definition")
end

struct IsNotDistributionException{T} <: Exception
    name::String
    distribution::T
end

function Base.showerror(io::IO, e::IsNotDistributionException)
    print(io, "$(e.name): $(e.distribution) is not probability distribution, sum ≈ $(sum(e.distribution))")
end
