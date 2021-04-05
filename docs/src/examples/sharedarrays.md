# Using SharedArrays in a parallel mapreduce

One might want to carry out a computation across several nodes of a cluster, where each node works on its own shared array. This may be achieved by using a `WorkerPool` that consists of one worker per node, which acts as a root process launching tasks on that node, and eventually returning the local array for an overall reduction across nodes.

We walk through one such example where we concatenate arrays that are locally initialized on each node.

We load the packages necessary, in this case these are `ParallelUtilities`, `SharedArrays` and `Distributed`.

```julia
using ParallelUtilities
using SharedArrays
using Distributed
```

We create a function to initailize the local part on each worker. In this case we simulate a heavy workload by adding a `sleep` period. In other words we assume that the individual elements of the array are expensive to evaluate. We obtain the local indices of the `SharedArray` through the function `localindices`.

```julia
function initialize_localpart(s, sleeptime)
    for ind in localindices(s)
        sleep(sleeptime)
        s[ind] = ind
    end
end
```

We create a function that runs on the root worker on each node and feeds tasks to other workers on that node. We use the function `ParallelUtilities.workers_myhost()` to obtain a list of all workers on the same node. We create the `SharedArray` on these workers so that it is entirely contained on one machine. This is achieved by passing the keyword argument `pids` to the `SharedArray` constructor. We asynchronously spawn tasks to initialize the local parts of the shared array on each worker.

```julia
function initializenode_sharedarray(sleeptime)
    # obtain the workers on the local machine
    pids = ParallelUtilities.workers_myhost()

    # Create a shared array spread across the workers on that node
    s = SharedArray{Int}((2_000,), pids = pids)

    # spawn remote tasks to initialize the local part of the shared array
    @sync for p in pids
        @spawnat p initialize_localpart(s, sleeptime)
    end
    return sdata(s)
end
```

We create a main function that runs on the calling process and concatenates the arrays on each node. This is run on a `WorkerPool` consisting of one worker per node which acts as the root process. We may obtain such a pool through the function `ParallelUtilities.workerpool_nodes()`. Finally we call `pmapreduce` with a mapping function  that initializes an array on each node, which is followed by a concatenation across the nodes.

```julia
function main_sharedarray(sleeptime)
    # obtain the workerpool with one process on each node
    pool = ParallelUtilities.workerpool_nodes()

    # obtain the number of workers in the pool.
    nw_node = nworkers(pool)

    # Evaluate the parallel mapreduce
    pmapreduce(x -> initializenode_sharedarray(sleeptime), hcat, pool, 1:nw_node)
end
```

We compare the results with a serial execution that uses a similar workflow, except we use `Array` instead of `SharedArray` and `mapreduce` instead of `pmapreduce`.

```julia
function initialize_serial(sleeptime)
    pids = ParallelUtilities.workers_myhost()
    s = Array{Int}(undef, 2_000)
    for ind in eachindex(s)
        sleep(sleeptime)
        s[ind] = ind
    end
    return sdata(s)
end

function main_serial(sleeptime)
    pool = ParallelUtilities.workerpool_nodes()
    nw_node = nworkers(pool)
    mapreduce(x -> initialize_serial(sleeptime), hcat, 1:nw_node)
end
```

We create a function to compare the performance of the two. We start with a precompilation run with no sleep time, followed by recording the actual timings.

```julia
function compare_with_serial()
    # precompile
    main_serial(0)
    main_sharedarray(0)

    # time
    println("Testing serial")
    A = @time main_serial(5e-3)
    println("Testing sharedarray")
    B = @time main_sharedarray(5e-3)

    println("Results match : ", A == B)
end
```

We run this script on a Slurm cluster across 2 nodes with 28 cores on each node. The results are:

```julia
julia> compare_with_serial()
Testing serial
 24.624912 seconds (27.31 k allocations: 1.017 MiB)
Testing sharedarray
  1.077752 seconds (4.60 k allocations: 246.281 KiB)
Results match : true
```

The full script may be found in the examples directory.
