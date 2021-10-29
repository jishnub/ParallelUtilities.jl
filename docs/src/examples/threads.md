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
    s = zeros(Int, 5_000)
    Threads.@threads for ind in eachindex(s)
        sleep(sleeptime)
        s[ind] = ind
    end
    return s
end
```

We create a main function that runs on the calling process and launches the array initialization task on each node. The array creation step on each node is followed by an eventual concatenation.

```julia
function pmapreduce_threads(sleeptime)
    pmapreduce(x -> initializenode_threads(sleeptime), hcat, 1:nworkers())
end
```

We compare the results with
* a `mapreduce` that uses a similar workflow, except the operation takes place entirely on one node
* a `@distributed` mapreduce, where the evaluation is spread across nodes.

```julia
function mapreduce_threads(sleeptime)
    mapreduce(x -> initializenode_threads(sleeptime), hcat, 1:nworkers())
end
```

We create a function to compare the performance of the two. We start with a precompilation run with no sleep time, followed by recording the actual timings with a sleep time of 5 milliseconds for each index of the array.

```julia
function compare_with_serial()
    # precompile
    mapreduce_threads(0)
    mapreduce_distributed_threads(0)
    pmapreduce_threads(0)
    # time
    sleeptime = 1e-2
    println("Testing threaded mapreduce")
    A = @time mapreduce_threads(sleeptime);
    println("Testing threaded+distributed mapreduce")
    B = @time mapreduce_distributed_threads(sleeptime);
    println("Testing threaded pmapreduce")
    C = @time pmapreduce_threads(sleeptime);

    println("Results match : ", A == B == C)
end
```

We run this script on a Slurm cluster across 2 nodes with 28 cores on each node. The results are:

```console
Testing threaded mapreduce
  4.161118 seconds (66.27 k allocations: 2.552 MiB, 0.95% compilation time)
Testing threaded+distributed mapreduce
  2.232924 seconds (48.64 k allocations: 2.745 MiB, 3.20% compilation time)
Testing threaded pmapreduce
  2.432104 seconds (6.79 k allocations: 463.788 KiB, 0.44% compilation time)
Results match : true
```

We see that there is little difference in evaluation times between the `@distributed` reduction and `pmapreduce`, both of which are roughly doubly faster than the one-node evaluation.

The full script along with the Slurm jobscript may be found in the examples directory.
