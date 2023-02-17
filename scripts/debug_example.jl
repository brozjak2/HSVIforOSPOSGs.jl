#!/usr/bin/env julia

using HSVIforOSPOSGs
using Logging
using CPLEX

global_logger(ConsoleLogger(stdout, Logging.Debug)) # Set debug level for global logger

osposg = OSPOSG("games/pursuit-evasion/peg05.osposg")
hsvi = HSVI(presolve_time_limit=30.0, optimizer_factory=() -> CPLEX.Optimizer()) # Specify short time limit for presolve phase

solve(osposg, hsvi, 1.0, 60.0) # Specify short time limit and small gap to show unsuccessful run

# Gap is still above desired precision:
println("gap: $(HSVIforOSPOSGs.width(osposg, hsvi))")
