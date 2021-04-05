# ParallelUtilities.jl

[![Build status](https://github.com/jishnub/ParallelUtilities.jl/workflows/CI/badge.svg)](https://github.com/jishnub/ParallelUtilities.jl/actions)
[![codecov](https://codecov.io/gh/jishnub/ParallelUtilities.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jishnub/ParallelUtilities.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jishnub.github.io/ParallelUtilities.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jishnub.github.io/ParallelUtilities.jl/dev)

Parallel mapreduce and other helpful functions for HPC, meant primarily for embarassingly parallel operations that often require one to split up a list of tasks into subsections that may be processed on individual cores.

# Installation

Install the package using

```julia
pkg> add ParallelUtilities
julia> using ParallelUtilities
```

# Quick start

Just replace `mapreduce` by `pmapreduce` in your code and things should work the same.

```julia
julia> @everywhere f(x) = (sleep(1); x^2); # some expensive calculation

julia> nworkers()
2

julia> @time mapreduce(f, +, 1:10) # Serial
 10.021436 seconds (40 allocations: 1.250 KiB)
385

julia> @time pmapreduce(f, +, 1:10) # Parallel
  5.137051 seconds (863 allocations: 39.531 KiB)
385
```

# Usage

See [the documentation](https://jishnub.github.io/ParallelUtilities.jl/stable) for examples and the API.
