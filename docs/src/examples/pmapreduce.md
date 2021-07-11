# Example of the use of pmapreduce

The function `pmapreduce` performs a parallel `mapreduce`. This is primarily useful when the function has to perform an expensive calculation, that is the evaluation time per core exceeds the setup and communication time. This is also useful when each core is allocated memory and has to work with arrays that won't fit into memory collectively, as is often the case on a cluster.

We walk through an example where we initialize and concatenate arrays in serial and in parallel.

We load the necessary modules first

```julia
using ParallelUtilities
using Distributed
```

We define the function that performs the initialization on each core. This step is embarassingly parallel as no communication happens between workers. We simulate an expensive calculation by adding a sleep interval for each index.

```julia
function initialize(sleeptime)
    A = Array{Int}(undef, 20, 20)
    for ind in eachindex(A)
        sleep(sleeptime)
        A[ind] = ind
    end
    return A
end
```

Next we define the function that calls `pmapreduce`:

```julia
function main_pmapreduce(sleeptime)
    pmapreduce(x -> initialize(sleeptime), hcat, 1:20)
end
```

We also define a function that carries out a serial mapreduce:

```julia
function main_mapreduce(sleeptime)
    mapreduce(x -> initialize(sleeptime), hcat, 1:20)
end
```

We compare the performance of the serial and parallel evaluations using 20 cores on one node:

We define a caller function first

```julia
function compare_with_serial()
    # precompile
    main_mapreduce(0)
    main_pmapreduce(0)

    # time
    println("Tesing serial")
    A = @time main_mapreduce(5e-6)
    println("Tesing parallel")
    B = @time main_pmapreduce(5e-6)

    # check results
    println("Results match : ", A == B)
end
```

We run this caller on the cluster:
```julia
julia> compare_with_serial()
Tesing serial
  9.457601 seconds (40.14 k allocations: 1.934 MiB)
Tesing parallel
  0.894611 seconds (23.16 k allocations: 1.355 MiB, 2.56% compilation time)
Results match : true
```

The full script may be found in the examples directory.
