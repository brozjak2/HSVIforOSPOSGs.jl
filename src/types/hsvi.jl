"""
    HSVI

Heuristic search value iteration solver for OSPOSGs.

The constructor takes following keyword arguments:

- `neighborhood`: parameter that guarantees Lipschitz continuity and convergence of the algorithm
- `presolve_epsilon`: precision the solver is trying to achieve during presolve phase
- `presolve_time_limit`: time limit for the presolve phase in seconds
- `optimizer_factory`: function that returns optimizer compatible with JuMP.Model to be used as LP solver
"""
@with_kw_noshow struct HSVI
    neighborhood::Float64 = 1e-6
    presolve_epsilon::Float64 = 1e-2
    presolve_time_limit::Float64 = 300.0
    optimizer_factory::Function = () -> GLPK.Optimizer()
end

function Base.show(io::IO, hsvi::HSVI)
    println(io, "HSVI:")
    for field in fieldnames(HSVI)
        println(io, "  $field = $(getfield(hsvi, field))")
    end
end
