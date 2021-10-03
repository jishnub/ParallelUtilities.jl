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
using ParallelUtilities: pmap_threadedfn, pmapreduce_threadedfn
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
function pmap_with_threads()
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
julia> ThreadsPmapTiming.pmap_with_threads()
Testing serial
 41.391285 seconds (956 allocations: 3.377 GiB, 0.60% gc time)
Testing distributed
  3.566095 seconds (3.46 k allocations: 138.875 KiB)
Testing distributed + 2 threads
  3.330016 seconds (10.50 k allocations: 475.625 KiB)
Testing distributed + 4 threads
  1.986760 seconds (10.41 k allocations: 466.516 KiB)
Testing distributed + 7 threads
  1.893500 seconds (10.37 k allocations: 456.234 KiB)
Testing distributed + 14 threads
  2.939750 seconds (10.34 k allocations: 455.719 KiB)
Testing distributed + 28 threads
  6.341226 seconds (10.33 k allocations: 456.734 KiB)
Results match : true

```

We see that the performance is optimal in this case using 4 to 7 threads per worker.
In general it might require some tinkering to find the optimal combination of threads and workers.

We may similarly carry out a `pmapreduce` with a threaded function:

```julia
g(d, nthreads) = begin
    BLAS.set_num_threads(nthreads)
    M = genmatrix(d)
    eigvals(M)
end

serial_mapreduce(d, n) = mapreduce(_ -> g(d, 1), hcat, 1:n)
parallel_mapreduce(d, n) = pmapreduce(_ -> g(d, 1), hcat, 1:n)
parallel_threads_mapreduce(d, n, nthreads) =
    pmapreduce_threadedfn(_ -> g(d, nthreads), nthreads, hcat, 1:n)
```

We define a function that times these:
```julia
function pmapreduce_with_threads()
    # time
    d = 2000 # size of the matrix is d × d
    n = nworkers() # number of matrices

    println("Testing serial_mapreduce")
    A = @time serial_mapreduce(d, n);

    println("Testing distributed mapreduce")
    B = @time parallel_mapreduce(d, n);

    C = map((2, 4, 7, 14, 28)) do nthreads
        println("Testing distributed + $nthreads threads")
        @time parallel_threads_mapreduce(d, n, nthreads);
    end

    println("Results match : ", all(==(A), [B, C...]))
end
```

The results are:

```julia
julia> ThreadsPmapTiming.pmapreduce_with_threads()
Testing serial_mapreduce
 41.410278 seconds (894 allocations: 3.401 GiB, 0.61% gc time)
Testing distributed mapreduce
  3.931968 seconds (21.20 k allocations: 1.713 MiB)
Testing distributed + 2 threads
  4.040329 seconds (16.87 k allocations: 1.587 MiB)
Testing distributed + 4 threads
  2.308510 seconds (12.38 k allocations: 1.402 MiB)
Testing distributed + 7 threads
  2.202479 seconds (10.42 k allocations: 1.325 MiB)
Testing distributed + 14 threads
  3.514926 seconds (9.14 k allocations: 1.270 MiB)
Testing distributed + 28 threads
  6.707958 seconds (8.54 k allocations: 1.243 MiB)
Results match : true
```

As before, the performance is optimal using 4-7 threads.

The full script may be found in the examples directory.
