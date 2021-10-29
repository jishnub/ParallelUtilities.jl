module ThreadsTiming

using ParallelUtilities
using Distributed

function initializenode_threads(sleeptime)
    s = zeros(Int, 5_000)
    Threads.@threads for ind in eachindex(s)
        sleep(sleeptime)
        s[ind] = ind
    end
    return s
end

function mapreduce_threads(sleeptime)
    mapreduce(x -> initializenode_threads(sleeptime), hcat, 1:nworkers())
end

function mapreduce_distributed_threads(sleeptime)
    @distributed hcat for _ in 1:nworkers()
        initializenode_threads(sleeptime)
    end
end

function pmapreduce_threads(sleeptime)
    pmapreduce(x -> initializenode_threads(sleeptime), hcat, 1:nworkers())
end

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

end
