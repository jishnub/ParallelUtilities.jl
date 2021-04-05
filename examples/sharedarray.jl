module SharedArraysTiming

using ParallelUtilities
using SharedArrays
using Distributed

function initialize_localpart(s, sleeptime)
    for ind in localindices(s)
        sleep(sleeptime)
        s[ind] = ind
    end
end

function initializenode_sharedarray(sleeptime)
    pids = ParallelUtilities.workers_myhost()
    s = SharedArray{Int}((2_000,), pids = pids)
    @sync for (ind, p) in enumerate(pids)
        @spawnat p initialize_localpart(s, sleeptime)
    end
    return sdata(s)
end

function initialize_serial(sleeptime)
    pids = ParallelUtilities.workers_myhost()
    s = Array{Int}(undef, 2_000)
    for ind in eachindex(s)
        sleep(sleeptime)
        s[ind] = ind
    end
    return sdata(s)
end

function main_sharedarray(sleeptime)
    workers_node_pool = ParallelUtilities.workerpool_node()
    w_node = workers(workers_node_pool)
    pmapreduce(x -> initializenode_sharedarray(sleeptime), hcat, workers_node_pool, 1:length(w_node))
end

function main_serial(sleeptime)
    workers_node_pool = ParallelUtilities.workerpool_node()
    w_node = workers(workers_node_pool)
    mapreduce(x -> initialize_serial(sleeptime), hcat, 1:length(w_node))
end

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

end
