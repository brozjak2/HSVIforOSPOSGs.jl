#!/usr/bin/env julia

# This example shows how to use records in recorder to plot bound convergence
# Note that this example requires CPLEX.jl package to be installed

using HSVIforOSPOSGs
using CPLEX
using Plots

osposg = OSPOSG("games/pursuit-evasion/peg04.osposg")

hsvi = HSVI(presolve_epsilon=0.001, presolve_time_limit=15.0, optimizer_factory=() -> CPLEX.Optimizer())

recorder = solve(osposg, hsvi, 1.0, 900.0)

plt = plot(recorder.timestamps, hcat(recorder.ub_values, recorder.lb_values);
    label=["UB" "LB"], xlabel="Time [s]", ylabel="Value",
    title="peg04.osposg", linewidth=2
)
scatter!(plt, [recorder.timestamps[1]], [recorder.ub_values[1]]; markershape=:circle, color=1, label="UB presolve", markersize=5)
scatter!(plt, [recorder.timestamps[1]], [recorder.lb_values[1]]; markershape=:circle, color=2, label="LB presolve", markersize=5)

savefig(plt, "assets/convergence.png")
