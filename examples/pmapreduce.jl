module PMapReduceTiming

using ParallelUtilities
using Distributed

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

function mapreduce_serial(n)
    mapreduce(x -> initialize(x, n), hcat, 1:nworkers())
end

function mapreduce_parallel(n)
    pmapreduce(x -> initialize(x, n), hcat, 1:nworkers())
end

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

end
