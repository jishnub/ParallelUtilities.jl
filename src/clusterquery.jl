module ClusterQueryUtils

using Distributed
using DataStructures

export hostnames
export nodenames
export procs_node
export nprocs_node

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

end
