"""
    Recorder

Stores the solver statistics after each iteration.
"""
mutable struct Recorder
    exploration_count::Int
    timestamps::Vector{Float64}
    lb_values::Vector{Float64}
    ub_values::Vector{Float64}
    gaps::Vector{Float64}
    gamma_sizes::Vector{Int}
    upsilon_sizes::Vector{Int}
    exploration_depths::Vector{Int}

    lb_presolve::Float64
    ub_presolve::Float64

    clock_start::Float64

    Recorder() = new(0, [], [], [], [], [], [], [], -Inf, Inf, time())
end

function Base.show(io::IO, recorder::Recorder)
    println(io, "HSVIRecorder:")
    for field in fieldnames(Recorder)
        println(io, "  $field = $(getfield(recorder, field))")
    end
    println(io, @sprintf("  running %7.2fs", time() - recorder.clock_start))
end
