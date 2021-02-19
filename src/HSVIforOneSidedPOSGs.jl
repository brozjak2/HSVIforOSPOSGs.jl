module HSVIforOneSidedPOSGs

using Printf
using JuMP
using Gurobi
using Parameters
using Logging

export hsvi

include("exceptions.jl")
include("types.jl")
include("value.jl")
include("utils.jl")
include("linear_programs.jl")
include("hsvi.jl")

const global GRB_ENV = Ref{Gurobi.Env}()

function __init__()
    GRB_ENV[] = Gurobi.Env()
end

end
