#!/usr/bin/env julia

# This example demonstrates how to use Gurobi as LP solver
# Note that this example requires Gurobi.jl package to be installed

using HSVIforOSPOSGs
using Gurobi

const GRB_ENV = Gurobi.Env() # Create single Gurobi environment to use for all models

osposg = OSPOSG("games/pursuit-evasion/peg04.osposg")
hsvi = HSVI(optimizer_factory=() -> Gurobi.Optimizer(GRB_ENV)) # Use Gurobi as LP solver

recorder = solve(osposg, hsvi, 1.0, 900.0)
