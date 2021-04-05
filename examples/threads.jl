module ThreadsTiming

using ParallelUtilities
using Distributed

function initialize_serial(sleeptime)
    s = zeros(Int, 2_000)
    for ind in eachindex(s)
        sleep(sleeptime)
        s[ind] = ind
    end
    return s
end

function initializenode_threads(sleeptime)
    s = zeros(Int, 2_000)
    Threads.@threads for ind in eachindex(s)
        sleep(sleeptime)
        s[ind] = ind
    end
    return s
end

function main_threads(sleeptime)
    workers_node_pool = ParallelUtilities.workerpool_nodes()
    nw_nodes = nworkers(workers_node_pool)
    pmapreduce(x -> initializenode_threads(sleeptime), hcat, workers_node_pool, 1:nw_nodes)
end

function main_serial(sleeptime)
    workers_node_pool = ParallelUtilities.workerpool_nodes()
    nw_nodes = nworkers(workers_node_pool)
    mapreduce(x -> initialize_serial(sleeptime), hcat, 1:nw_nodes)
end

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

end
