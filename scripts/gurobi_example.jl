#!/usr/bin/env julia

using HSVIforOSPOSGs
using Gurobi

const GRB_ENV = Gurobi.Env() # Create single Gurobi environment to use for all models

osposg = OSPOSG("games/pursuit-evasion/peg04.osposg")
hsvi = HSVI(optimizer_factory=() -> Gurobi.Optimizer(GRB_ENV)) # Use Gurobi as LP solver

solve(osposg, hsvi, 1.0, 900.0)
