module HSVIforOneSidedPOSGs

using Printf
using JuMP
using Gurobi
using Logging

export hsvi

include("exceptions.jl")
include("abstract_types.jl")
include("state.jl")
include("partition.jl")
include("game.jl")
include("load.jl")
include("linear_programs.jl")
include("utils.jl")
include("hsvi.jl")

const global GRB_ENV = Ref{Gurobi.Env}()

function __init__()
    GRB_ENV[] = Gurobi.Env()
end

end
