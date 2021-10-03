module ThreadsPmapTiming

using ParallelUtilities
using ParallelUtilities: pmap_threadedfn, pmapreduce_threadedfn
using Distributed
@everywhere using LinearAlgebra

genmatrix(d) = collect(Diagonal(1:d))

f(d, nthreads) = begin
    BLAS.set_num_threads(nthreads)
    M = genmatrix(d)
    lam = eigvals(M)
    sum(lam), prod(lam)
end

serial(d, n) = map(_ -> f(d, 1), 1:n)
parallel(d, n) = pmap(_ -> f(d, 1), 1:n)
parallel_threads(d, n, nthreads) =
    pmap_threadedfn(_ -> f(d, nthreads), nthreads, 1:n)

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

g(d, nthreads) = begin
    BLAS.set_num_threads(nthreads)
    M = genmatrix(d)
    eigvals(M)
end

serial_mapreduce(d, n) = mapreduce(_ -> g(d, 1), hcat, 1:n)
parallel_mapreduce(d, n) = pmapreduce(_ -> g(d, 1), hcat, 1:n)
parallel_threads_mapreduce(d, n, nthreads) =
    pmapreduce_threadedfn(_ -> g(d, nthreads), nthreads, hcat, 1:n)

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

end

