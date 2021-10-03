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
export oneworkerpernode
export workerpool_nodes
export workerpool_threadedfn
export workerspernode
export workerspernode_threadedfn
export workers_myhost
export @everynode

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

function maybetrimmedworkerpool(workers::AbstractVector{<:Integer}, N)
    w = chooseworkers(workers, N)
    WorkerPool(w)
end

function maybetrimmedworkerpool(pool::AbstractWorkerPool, N)
    pool_trimmed = maybetrimmedworkerpool(workers(pool), N)
    typeof(pool)(workers(pool_trimmed))
end

"""
    workerpool_nodes([pool::AbstractWorkerPool = WorkerPool(workers())], [T::Type{<:AbstractWorkerPool} = WorkerPool])

Return an `AbstractWorkerPool` of type `T` with one worker per machine/node of the cluster.
"""
workerpool_nodes(w::AbstractVector{<:Integer}, T::Type{<:AbstractWorkerPool} = WorkerPool) =
    T(oneworkerpernode(w))
workerpool_nodes(pool::AbstractWorkerPool, T::Type{<:AbstractWorkerPool} = typeof(pool)) =
    workerpool_nodes(workers(pool), T)
workerpool_nodes(T::Type{<:AbstractWorkerPool} = WorkerPool) =
    workerpool_nodes(workers(), T)

"""
    oneworkerpernode([workers::AbstractVector{<:Integer} = workers()])
    oneworkerpernode(pool::AbstractWorkerPool)

Return a subsample of `workers` such that each `pid` in the returned vector is located on
one machine/node of the cluster.
"""
oneworkerpernode(workers::AbstractVector{<:Integer} = workers()) = workerspernode(1, workers)
oneworkerpernode(pool::AbstractWorkerPool) = oneworkerpernode(workers(pool))

"""
    workerspernode(n::Integer, [workers::AbstractVector{<:Integer} = workers()])
    workerspernode(n::Integer, pool::AbstractWorkerPool)

Return a subsample of `workers` such that there are at most `n` workers on each machine/node
of the cluster.

    workerspernode(nw::AbstractVector{<:Integer}, workers::AbstractVector{<:Integer} = workers())

Return a subsample of `workers` such that there are at most `nw[i]` workers on the `i`th machine/node
of the cluster.
"""
function workerspernode(n::Integer, workers::AbstractVector{<:Integer} = workers())
    workers_on_hosts = procs_node(workers)
    reduce(vcat, [v[1:min(n, lastindex(v))] for v in values(workers_on_hosts)])
end
function workerspernode(nw_node::AbstractVector{<:Integer}, workers::AbstractVector{<:Integer} = workers())
    workers_on_hosts = procs_node(workers)
    length(nw_node) == length(keys(workers_on_hosts)) ||
        throw(ArgumentError("length of workers per host must match the number of hosts"))
    p = [v[1:min(nw, lastindex(v))] for (nw, v) in zip(nw_node, values(workers_on_hosts))]
    reduce(vcat, p)
end
workerspernode(n, pool::AbstractWorkerPool) = workerspernode(n, workers(pool))


"""
    workerpool_threadedfn(nthreads::Integer, [pool::AbstractWorkerPool = WorkerPool(workers())], [T::Type{<:AbstractWorkerPool} = WorkerPool])

Return an `AbstractWorkerPool` of type `T` with a subsample of workers from `workers(pool)` such that
each worker on each machine/node of the cluster may spawn `nthreads` threads cooperatively with other
local workers. If the number of workers available is `nthreads × m` on a node then this returns `m`
workers on that node.

See also: [`workerspernode_threadedfn`](@ref)
"""
workerpool_threadedfn(nthreads::Integer, w::AbstractVector{<:Integer}, T::Type{<:AbstractWorkerPool} = WorkerPool) =
    T(workerspernode_threadedfn(nthreads, w))
workerpool_threadedfn(nthreads::Integer, pool::AbstractWorkerPool, T::Type{<:AbstractWorkerPool} = typeof(pool)) =
    workerpool_threadedfn(nthreads, workers(pool), T)
workerpool_threadedfn(nthreads::Integer, T::Type{<:AbstractWorkerPool} = WorkerPool) =
    workerpool_threadedfn(nthreads, workers(), T)

"""
    workerspernode_threadedfn(nthreads::Integer, [workers::AbstractVector{<:Integer} = workers()])
    workerspernode_threadedfn(nthreads::Integer, pool::AbstractWorkerPool)

Return a subsample of `workers` such that each workers on a machine/node of the cluster
may spawn `nthreads` threads cooperatively with other local workers.
If the number of workers available is `nthreads × m` on a node then this returns `m`
workers on that node.

See also: [`workerpool_threadedfn`](@ref)
"""
function workerspernode_threadedfn(nthreads::Integer, workers::AbstractVector{<:Integer} = workers())
    workers_on_hosts = procs_node(workers)
    nw_node = [max(1, length(v) ÷ nthreads) for v in values(workers_on_hosts)]
    workerspernode(nw_node, workers)
end
workerspernode_threadedfn(nthreads::Integer, pool::AbstractWorkerPool) =
    workerspernode_threadedfn(nthreads, workers(pool))

"""
    workers_myhost([workers::AbstractVector{<:Integer} = workers()])

Return a list of all workers that are on the local machine/node of the cluster.
"""
workers_myhost(workers::AbstractVector{<:Integer} = workers()) = procs_node(workers)[Libc.gethostname()]
workers_myhost(pool::AbstractWorkerPool) = workers_myhost(workers(pool))

"""
    @everynode [procs()] expr

Evaluate an expression `expr` on one worker on each machine/node of the cluster.
This is complementary to `Distributed.@everywhere` that evaluates a function on each process.
The process on which `expr` is evaluated on any node is arbitrary, and should not be relied upon.
"""
macro everynode(procs, ex)
    quote
        pool = workerpool_nodes(procs)
        @everywhere workers(pool) $ex
    end
end
macro everynode(ex)
    quote
        pool = workerpool_nodes(procs())
        @everywhere workers(pool) $ex
    end
end

end
