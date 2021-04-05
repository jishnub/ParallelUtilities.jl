```@meta
DocTestSetup  = quote
    using ParallelUtilities
end
```

# ParallelUtilities.jl

The `ParallelUtilities` module defines certain functions that are useful in a parallel `mapreduce` operation, with particular focus on HPC systems. The approach is similar to a `@distributed (op) for` loop, where the entire section of iterators is split evenly across workers and reduced locally, followed by a global reduction. The operation is not load-balanced at present, and does not support retry on error.

# Performance

The `pmapreduce`-related functions are expected to be more performant than `@distributed` for loops. As an example, running the following on a Slurm cluster using 2 nodes with 28 cores on each leads to

```julia
julia> using Distributed

julia> using ParallelUtilities

julia> @everywhere f(x) = ones(10_000, 1_000);

julia> A = @time @distributed (+) for i=1:nworkers()
                f(i)
            end;
 22.637764 seconds (3.35 M allocations: 8.383 GiB, 16.50% gc time, 0.09% compilation time)

julia> B = @time pmapreduce(f, +, 1:nworkers());
  2.170926 seconds (20.47 k allocations: 77.117 MiB)

julia> A == B
true
```

The difference increases with the size of data as well as the number of workers. This is because the `pmapreduce*` functions defined in this package perform local reductions before communicating data across nodes. Note that in this case the same operation may be carried out elementwise to obtain better performance.

```julia
julia> @everywhere elsum(x,y) = x .+= y;

julia> A = @time @distributed (elsum) for i=1:nworkers()
               f(i)
           end;
 20.537353 seconds (4.74 M allocations: 4.688 GiB, 2.56% gc time, 1.26% compilation time)

julia> B = @time pmapreduce(f, elsum, 1:nworkers());
  1.791662 seconds (20.50 k allocations: 77.134 MiB)
```

A similar evaluation on 560 cores (20 nodes) takes

```julia
julia> @time for i = 1:10; pmapreduce(f, +, 1:nworkers()); end
145.963834 seconds (2.53 M allocations: 856.693 MiB, 0.12% gc time)

julia> @time for i = 1:10; pmapreduce(f, elsum, 1:nworkers()); end
133.810309 seconds (2.53 M allocations: 856.843 MiB, 0.13% gc time)
```

An example of a mapreduce operation involving large arrays (comparable to the memory allocated to each core) evaluated on 56 cores is

```julia
julia> @everywhere f(x) = ones(12_000, 20_000);

julia> @time ParallelUtilities.pmapreduce(f, elsum, 1:nworkers());
 36.824788 seconds (26.40 k allocations: 1.789 GiB, 0.05% gc time)
```

# Comparison with other parallel mapreduce packages

Other packages that perform parallel mapreduce are [`ParallelMapReduce`](https://github.com/hcarlsso/ParallelMapReduce.jl) and [`Transducers`](https://github.com/JuliaFolds/Transducers.jl). The latter provides a `foldxd` function that performs an associative distributed `mapfold`. The performances of these functions compared to this package (measured on 1 node with 28 cores) are listed below:

```julia
julia> @everywhere f(x) = ones(10_000, 10_000);

julia> A = @time ParallelUtilities.pmapreduce(f, +, 1:nworkers());
 10.105696 seconds (14.03 k allocations: 763.511 MiB)

julia> B = @time ParallelMapReduce.pmapreduce(f, +, 1:nworkers(), algorithm = :reduction_local);
 30.955381 seconds (231.93 k allocations: 41.200 GiB, 7.63% gc time, 0.23% compilation time)

julia> C = @time Transducers.foldxd(+, 1:nworkers() |> Transducers.Map(f));
 30.154166 seconds (655.40 k allocations: 41.015 GiB, 8.65% gc time, 1.03% compilation time)

julia> A == B == C
true
```

Note that at present the performances of the `pmapreduce*` functions defined in this package are not comparable to equivalent MPI implementations. For example, an MPI mapreduce operation using [`MPIMapReduce.jl`](https://github.com/jishnub/MPIMapReduce.jl) computes an inplace sum over `10_000 x 10_000` matrices on each core in

```julia
3.413968 seconds (3.14 M allocations: 1.675 GiB, 2.99% gc time)
```

whereas this package computes it in
```julia
julia> @time ParallelUtilities.pmapreduce(f, elsum, 1:nworkers());
  7.264023 seconds (12.46 k allocations: 763.450 MiB, 1.69% gc time)
```

This performance gap might reduce in the future.

!!! note
    The timings have all been measured on Julia 1.6 on an HPC cluster that has nodes with with 2 Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz CPUs ("Broadwell", 14 cores/socket, 28 cores/node). They are also measured for subsequent runs after an initial precompilation step. The exact evaluation time might also vary depending on the cluster load.

# Known issues

1. This package currently does not implement a specialized `mapreduce` for arrays, so the behavior might differ for specialized array argument types (eg. `DistributedArray`s). This might change in the future.

2. This package deals with distributed (multi-core) parallelism, and at this moment it has not been tested extensively alongside multi-threading. Multithreading + multiprocessing has been tested where the number of threads times the number of processes equals the number of available cores. See [an example](examples/threads.md) of multithreading used in such a form, where each node uses threads locally, and reduction across nodes is performed using multiprocessing.
