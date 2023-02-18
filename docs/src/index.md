```@meta
CurrentModule = HSVIforOSPOSGs
```

# HSVIforOSPOSGs.jl

This is documentation for [HSVIforOSPOSGs](https://github.com/brozjak2/HSVIforOSPOSGs.jl).

HSVIforOSPOSGs is **unofficial** Julia implementation of heuristic search value iteration (HSVI) algorithm for one-sided partially observable stochastic games (OSPOSGs) as described in [Heuristic Search Value Iteration for One-Sided Partially Observable Stochastic Games](https://doi.org/10.1016/j.artint.2022.103838).
OSPOSGs can be seen as a generalization of [Partially observable Markov decision processes](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process) (POMDPs) and [Stochastic games](https://en.wikipedia.org/wiki/Stochastic_game), where one agent has imperfect information while their opponent has full knowledge of the current situation. The algorithm is multi-agent adaptation of [Heuristic Search Value Iteration for POMDPs](https://arxiv.org/abs/1207.4166).

## Index

- [Manual](@ref)
  - [Getting started](@ref)
  - [LP Solvers](@ref)
  - [Logging level](@ref)
  - [Examples](@ref)
- [Game format](@ref)
- [API Reference](@ref)
