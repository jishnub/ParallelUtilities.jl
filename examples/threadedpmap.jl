module ThreadsPmapTiming

using ParallelUtilities
using ParallelUtilities: pmap_threadscoop
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
parallel_threads(d, n, nthreads) = pmap_threadscoop(_ -> f(d, nthreads), nthreads, 1:n)

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

end

