module PMapReduceTiming

using ParallelUtilities
using Distributed

function initialize(sleeptime)
    A = Array{Int}(undef, 20, 20)
    for ind in eachindex(A)
        sleep(sleeptime)
        A[ind] = ind
    end
    return A
end

function main_mapreduce(sleeptime)
    mapreduce(x -> initialize(sleeptime), hcat, 1:20)
end

function main_pmapreduce(sleeptime)
    pmapreduce(x -> initialize(sleeptime), hcat, 1:20)
end

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

end
