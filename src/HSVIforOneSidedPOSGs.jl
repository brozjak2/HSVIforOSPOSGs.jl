module HSVIforOneSidedPOSGs

using Flux
using GLPK
using Gurobi
using JuMP
using LinearAlgebra
using Logging
using Parameters
using Printf

export hsvi

include("exceptions.jl")
include("types.jl")
include("load.jl")
include("value.jl")
include("utils.jl")
include("qre.jl")
include("linear_programs.jl")
include("hsvi.jl")

const global GRB_ENV = Ref{Gurobi.Env}()

# set global Gurobi environment at module runtime
function __init__()
    GRB_ENV[] = Gurobi.Env()
end

end
