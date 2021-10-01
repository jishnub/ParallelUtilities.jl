# Using threads in a parallel mapreduce

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

The full script may be found in the examples directory.

# Performing a threaded pmap

In this example we evaluate the eigenvalues of a number of matrices. Each call to `LinearAlgebra.eigvals` may use multi-threading in `BLAS`, and we distribute the vector of matrices in such a way that the product of the number of threads per worker and the number of workers is equal to the number of matrices.

We load the packages necessary, in this case these are `ParallelUtilities`, `Distributed` and `LinearAlgebra`. We load `LinearAlgebra` on all workers to set the number of threads used in the eigenvalue evaluation locally.

```julia
using ParallelUtilities
using ParallelUtilities: pmap_threadscoop
using Distributed
@everywhere using LinearAlgebra
```

As an example we choose to find the eigenvalues of dense diagonal matrices, and evaluate the trace and determinant from these. We define a matrix that generates these matrices locally on each worker to avoid communicating across workers.

```julia
genmatrix(d) = collect(Diagonal(1:d))
```

Next we define a function that evaluates the eigenvalues and computes the trace and determinant.

```julia
f(d, nthreads) = begin
    BLAS.set_num_threads(nthreads)
    M = genmatrix(d)
    lam = eigvals(M)
    sum(lam), prod(lam)
end
```

We define helper functions to evaluate `f` serially, in parallel using all workers and 1 thread per worker, and using a combination of workers and threads.

```julia
serial(d, n) = map(_ -> f(d, 1), 1:n)
parallel(d, n) = pmap(_ -> f(d, 1), 1:n)
parallel_threads(d, n, nthreads) = pmap_threadscoop(_ -> f(d, nthreads), nthreads, 1:n)
```

Finally we define a main function that calls the helper functions and measures the runtimes:

```julia
function compare_with_serial()
    # precompile
    serial(1, 1)
    parallel(1, 1)
    parallel_threads(1, 1, 1)

    # time
    d = 2000 # size of the matrix is d × d
    n = nworkers() # number of matrices

    println("Testing serial")
    A = @time serial(d, n);

    println("Testing distributed")
    B = @time parallel(d, n);

    C = map((2, 4, 7, 14, 28)) do nthreads
        println("Testing distributed + $nthreads threads")
        @time parallel_threads(d, n, nthreads);
    end

    println("Results match : ", all(==(A), [B, C...]))
end
```

We run this script on a Slurm cluster across 2 nodes with 28 cores on each node. The results are:

```julia
julia> ThreadsPmapTiming.compare_with_serial()
Testing serial
 40.568560 seconds (956 allocations: 3.377 GiB, 0.25% gc time)
Testing distributed
  3.560556 seconds (5.01 k allocations: 222.859 KiB)
Testing distributed + 2 threads
  3.240148 seconds (9.41 k allocations: 417.156 KiB)
Testing distributed + 4 threads
  1.944696 seconds (8.75 k allocations: 386.984 KiB)
Testing distributed + 7 threads
  1.842619 seconds (8.45 k allocations: 367.219 KiB)
Testing distributed + 14 threads
  2.833750 seconds (8.26 k allocations: 360.484 KiB)
Testing distributed + 28 threads
  6.456497 seconds (8.16 k allocations: 350.188 KiB)
Results match : true
```

We see that the performance is optimal in this case using 4 to 7 threads per worker.
In general it might require some tinkering to find the optimal combination of threads and workers.

The full script may be found in the examples directory.
