#!/usr/bin/env julia

# This is minimal working example

using HSVIforOSPOSGs

osposg = OSPOSG("games/pursuit-evasion/peg03.osposg") # Load game
hsvi = HSVI() # Create solver

recorder = solve(osposg, hsvi, 1.0, 60.0) # Solve, specify desired gap in initial_belief and time limit

# At this point, `osposg` is solved:
println("gap: $(HSVIforOSPOSGs.width(osposg, hsvi))")
