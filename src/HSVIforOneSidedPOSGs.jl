module HSVIforOneSidedPOSGs

using Printf
using JuMP
using Gurobi
using Parameters
using Logging

export hsvi

include("exceptions.jl")
include("types.jl")
include("partition.jl")
include("game.jl")
include("context.jl")
include("linear_programs.jl")
include("utils.jl")
include("hsvi.jl")

const global GRB_ENV = Ref{Gurobi.Env}()

function __init__()
    GRB_ENV[] = Gurobi.Env()
end

end
