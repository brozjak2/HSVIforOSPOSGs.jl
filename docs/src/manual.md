```@meta
CurrentModule = HSVIforOSPOSGs
```

```@setup example
using HSVIforOSPOSGs: OSPOSG, HSVI, solve
```

# Manual

This page shows how to use the HSVIforOSPOSGs.

## Getting started

First, a game in `.osgposg` format must be loaded as `OSPOSG`:

```@repl example
osposg = OSPOSG("../../games/pursuit-evasion/peg03.osposg")
```

```@docs
OSPOSG
OSPOSG(path::AbstractString)
```

Then, `HSVI` algorithm must be instantiated/configured. Here we use `HSVI` with default parameters:

```@repl example
hsvi = HSVI()
```

For available parameters and their meaning see below:

```@docs
HSVI
```

The `neigborhood` parameter $D$ of `HSVI` must be within these bounds

$0 < D < \frac{(1 − \gamma) \varepsilon}{2 \delta}$

for the algorithm to have convergence guarantees, where $\gamma$ is discount factor, $\varepsilon$ is specified gap and $\delta$ is Lipschitz delta of the game.

After loading the game and configuring the algorithm we can start solving by running `solve`:

```@repl example
recorder = solve(osposg, hsvi, 1.0, 60.0)
```

```@docs
solve
```

The `recorder` contains records of the algorithm progress after each iteration and can be used for further examination or presentation of results.

```@docs
Recorder
```

## LP Solvers

HSVIforOSPOSGs uses [GLPK](https://www.gnu.org/software/glpk) as a [JuMP](https://github.com/jump-dev/JuMP.jl) solver for linear programs by default because it is open-source and easily installable through standalone Julia package.
However, note that the LP models being solved in HSVI for OSPOSGs are quite complex and GLPK might get stuck on some of them or report them as being unfeasible (although they should be feasible).
Therefore, it is recommended to use more powerful solvers (such as [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio), [Gurobi](https://www.gurobi.com), etc.).
However, installing them is not as straightforward and often requires 3rd party binary and/or proprietary license.
If you wish to use different solver you can do so by installing the JuMP wrapper for the given solver (refer to installation instructions at [CPLEX.jl](https://github.com/jump-dev/CPLEX.jl), [Gurobi.jl](https://github.com/jump-dev/Gurobi.jl), etc.) and then pass the optimizer factory into `HSVI` constructor:

```julia
using CPLEX

HSVI(optimizer_factory=() -> CPLEX.Optimizer())
```

## Logging level

This package utilizes Julia logging facilities to communicate the progress of the algorithm to the user.
By default only info, warning and error messages are displayed.
To display more detailed debuging info, use the following before running the algorithm:

```julia
using Logging

global_logger(ConsoleLogger(stdout, Logging.Debug))
```

On the contrary, to disable info logging use:

```julia
global_logger(ConsoleLogger(stdout, Logging.Warn))
```

## Examples

Additional examples on how to configure and run the algorithm are provided in the [`scripts`](https://github.com/brozjak2/HSVIforOSPOSGs.jl/tree/master/scripts) directory.
