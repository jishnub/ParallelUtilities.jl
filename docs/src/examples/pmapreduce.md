# Example of the use of pmapreduce

## Using [ClusterManagers.jl](https://github.com/JuliaParallel/ClusterManagers.jl)

The function `pmapreduce` performs a parallel `mapreduce`. This is primarily useful when the function has to perform an expensive calculation, that is the evaluation time per core exceeds the setup and communication time. This is also useful when each core is allocated memory and has to work with arrays that won't fit into memory collectively, as is often the case on a cluster.

We walk through an example where we initialize and concatenate arrays in serial and in parallel.

We load the necessary modules first

```julia
using ParallelUtilities
using Distributed
```

We define the function that performs the initialization on each core. This step is embarassingly parallel as no communication happens between workers.

```julia
function initialize(x, n)
    inds = 1:n
    d, r = divrem(length(inds), nworkers())
    ninds_local = d + (x <= r)
    A = zeros(Int, 50, ninds_local)
    for ind in eachindex(A)
        A[ind] = ind
    end
    return A
end
```

Next we define the function that calls `pmapreduce`:

```julia
function mapreduce_parallel(n)
    pmapreduce(x -> initialize(x, n), hcat, 1:nworkers())
end
```

We also define a function that carries out a serial mapreduce:

```julia
function mapreduce_serial(n)
    mapreduce(x -> initialize(x, n), hcat, 1:nworkers())
end
```

We compare the performance of the distributed for loop and the parallel mapreduce using `3` nodes with `28` cores on each node.

We define a caller function first

```julia
function compare_with_serial()
    # precompile
    mapreduce_serial(1)
    mapreduce_parallel(nworkers())

    # time
    n = 2_000_000
    println("Tesing serial mapreduce")
    A = @time mapreduce_serial(n)
    println("Tesing pmapreduce")
    B = @time mapreduce_parallel(n)

    # check results
    println("Results match : ", A == B)
end
```

We run this caller on the cluster:
```console
Tesing serial mapreduce
 23.986976 seconds (8.26 k allocations: 30.166 GiB, 11.71% gc time, 0.02% compilation time)
Tesing pmapreduce
  7.465366 seconds (29.55 k allocations: 764.166 MiB)
Results match : true
```

In this case the the overall gain is only around a factor of `3`. In general a parallel mapreduce is advantageous if the time required to evaluate the function far exceeds that required to communicate across workers.

The time required for a `@distributed` `for` loop is unfortunately exceedingly high for it to be practical here.

The full script may be found in the examples directory.

## Using [MPIClusterManagers.jl](https://github.com/JuliaParallel/MPIClusterManagers.jl)

The same script may also be used by initiating an MPI cluster (the cluster in this case has 77 workers + 1 master process). This leads to the timings

```console
Using MPI_TRANSPORT_ALL
Tesing serial mapreduce
 22.263389 seconds (8.07 k allocations: 29.793 GiB, 11.70% gc time, 0.02% compilation time)
Tesing pmapreduce
 11.374551 seconds (65.92 k allocations: 2.237 GiB, 0.46% gc time)
Results match : true
```

The performance is worse in this case than that obtained using `ClusterManagers.jl`.
