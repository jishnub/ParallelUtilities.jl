# Using Threads in a parallel mapreduce

One might want to carry out a computation across several nodes of a cluster, where each node uses multithreading to evaluate a result that is subsequently reduced across all nodes. We walk through one such example where we concatenate arrays that are locally initialized on each node using threads.

We load the packages necessary, in this case these are `ParallelUtilities` and `Distributed`.

```julia
using ParallelUtilities
using Distributed
```

We create a function to initailize the local part on each worker. In this case we simulate a heavy workload by adding a `sleep` period. In other words we assume that the individual elements of the array are expensive to evaluate. We use `Threads.@threads` to split up the loop into sections that are processed on invidual threads.

```julia
function initializenode_threads(sleeptime)
    s = zeros(Int, 2_000)
    Threads.@threads for ind in eachindex(s)
        sleep(sleeptime)
        s[ind] = ind
    end
    return s
end
```

We create a main function that runs on the calling process and launches the array initialization task on each node. This is run on a `WorkerPool` consisting of one worker per node which acts as the root process. We may obtain such a pool through the function `ParallelUtilities.workerpool_nodes()`. The array creation step on each node is followed by an eventual concatenation.

```julia
function main_threads(sleeptime)
    # obtain the workerpool with one process on each node
    pool = ParallelUtilities.workerpool_nodes()

    # obtain the number of workers in the pool.
    nw_nodes = nworkers(pool)

    # Evaluate the parallel mapreduce
    pmapreduce(x -> initializenode_threads(sleeptime), hcat, pool, 1:nw_nodes)
end
```

We compare the results with a serial execution that uses a similar workflow, except we use `mapreduce` instead of `pmapreduce` and do not use threads.

```julia
function initialize_serial(sleeptime)
    s = zeros(Int, 2_000)
    for ind in eachindex(s)
        sleep(sleeptime)
        s[ind] = ind
    end
    return s
end

function main_serial(sleeptime)
    pool = ParallelUtilities.workerpool_nodes()
    nw_nodes = nworkers(pool)
    mapreduce(x -> initialize_serial(sleeptime), hcat, 1:nw_nodes)
end
```

We create a function to compare the performance of the two. We start with a precompilation run with no sleep time, followed by recording the actual timings with a sleep time of 5 milliseconds for each index of the array.

```julia
function compare_with_serial()
    # precompile
    main_serial(0)
    main_threads(0)

    # time
    println("Testing serial")
    A = @time main_serial(5e-3);
    println("Testing threads")
    B = @time main_threads(5e-3);

    println("Results match : ", A == B)
end
```

We run this script on a Slurm cluster across 2 nodes with 28 cores on each node. The results are:

```julia
julia> compare_with_serial()
Testing serial
 24.601593 seconds (22.49 k allocations: 808.266 KiB)
Testing threads
  0.666256 seconds (3.71 k allocations: 201.703 KiB)
Results match : true
```

The full script may be found [here](threads.jl). To run this, use

```julia
julia> @everywhere include("threads.jl")

julia> ThreadsTiming.compare_with_serial()
```

