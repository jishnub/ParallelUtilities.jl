module ClusterQueryUtils

using Distributed
using DataStructures

export hostnames
export nodenames
export procs_node
export nprocs_node
export workersactive
export chooseworkers
export maybetrimmedworkerpool
export workerpool_nodes
export workers_myhost

"""
    hostnames([procs = workers()])

Return the hostname of each worker in `procs`. This is obtained by evaluating
`Libc.gethostname()` on each worker asynchronously.
"""
function hostnames(procs = workers())
    hostnames = Vector{String}(undef, length(procs))

    @sync for (ind,p) in enumerate(procs)
        @async hostnames[ind] = @fetchfrom p Libc.gethostname()
    end
    return hostnames
end


"""
    nodenames([procs = workers()])

Return the unique hostnames that the workers in `procs` lie on.
On an HPC system these are usually the hostnames of the nodes involved.
"""
nodenames(procs = workers()) = nodenames(hostnames(procs))

function nodenames(hostnames::AbstractVector{String})
    unique(hostnames)
end

"""
    procs_node([procs = workers()])

Return the worker ids on each host of the cluster.
On an HPC system this would return the workers on each node.
"""
function procs_node(procs = workers())
    hosts = hostnames(procs)
    nodes = nodenames(hosts)
    procs_node(procs, hosts, nodes)
end

function procs_node(procs, hosts, nodes)
    OrderedDict(node => procs[findall(isequal(node),hosts)] for node in nodes)
end

"""
    nprocs_node([procs = workers()])

Return the number of workers on each host.
On an HPC system this would return the number of workers on each node.
"""
function nprocs_node(procs = workers())
    nprocs_node(hostnames(procs))
end

function nprocs_node(hostnames::AbstractVector{String})
    nodes = nodenames(hostnames)
    nprocs_node(hostnames, nodes)
end

function nprocs_node(hostnames::AbstractVector, nodes::AbstractVector)
    OrderedDict(node => count(isequal(node), hostnames) for node in nodes)
end

function nprocs_node(d::AbstractDict)
    OrderedDict(node => length(procs) for (node, procs) in d)
end

function workersactive(pool::AbstractWorkerPool, len::Integer,
    workers_on_hosts::AbstractDict = procs_node(workers(pool)))

    nw = min(nworkers(pool), len)
    chooseworkers(workers(pool), len, workers_on_hosts)
end

function chooseworkers(workerspool, n::Integer, workers_on_hosts::AbstractDict = procs_node(workerspool))
    n >= 1 || throw(ArgumentError("number of workers to choose must be >= 1"))
    length(workerspool) <= n && return workerspool
    myhost = Libc.gethostname()
    if myhost in keys(workers_on_hosts)
        if length(workers_on_hosts[myhost]) >= n
            return workers_on_hosts[myhost][1:n]
        else
            w_chosen = workers_on_hosts[myhost]
            np_left = n - length(w_chosen)
            for (host, workers_host) in workers_on_hosts
                np_left <= 0 && break
                host == myhost && continue
                workers_host_section = @view workers_host[1:min(length(workers_host), np_left)]
                w_chosen = vcat(w_chosen, workers_host_section)
                np_left -= length(workers_host_section)
            end
            return w_chosen
        end
    else
        return workerspool[1:n]
    end
end

function maybetrimmedworkerpool(workers, N)
    w = chooseworkers(workers, N)
    WorkerPool(w)
end

"""
    workerpool_nodes([pool::AbstractWorkerPool = WorkerPool(workers())])

Return a `WorkerPool` with one worker per machine/node of the cluster.
"""
workerpool_nodes(pool::AbstractWorkerPool) = WorkerPool(oneworkerpernode(workers(pool)))
workerpool_nodes() = WorkerPool(oneworkerpernode(workers()))

"""
    oneworkerpernode([workers::AbstractVector{<:Integer} = workers()])

Return a subsample of workers such that each `pid` in the returned vector is located on
one machine/node of the cluster.
"""
function oneworkerpernode(workers::AbstractVector{<:Integer} = workers())
    workers_on_hosts = procs_node(workers)
    [first(v) for v in values(workers_on_hosts)]
end

"""
    workers_myhost([workers::AbstractVector{<:Integer} = workers()])

Return a list of all workers that are on the local machine/node of the cluster.
"""
workers_myhost(workers::AbstractVector{<:Integer} = workers()) = procs_node(workers)[Libc.gethostname()]
workers_myhost(pool::AbstractWorkerPool) = workers_myhost(workers(pool))

end
