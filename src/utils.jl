"""
	nworkersactive(iterators::Tuple)

Number of workers required to contain the outer product of the iterators.
"""
@inline function nworkersactive(iterators::Tuple)
	min(nworkers(), ntasks(iterators))
end
@inline nworkersactive(ps::ProductSplit) = nworkersactive(ps.iterators)
@inline nworkersactive(args::AbstractRange...) = nworkersactive(args)

"""
	workersactive(iterators::Tuple)

Workers required to split the outer product of the iterators. 
If `ntasks(iterators) < nworkers()` then the first `ntasks(iterators)`
workers are chosen.
"""
@inline workersactive(iterators::Tuple) = workers()[1:nworkersactive(iterators)]
@inline workersactive(ps::ProductSplit) = workersactive(ps.iterators)
@inline workersactive(args::AbstractRange...) = workersactive(args)

"""
	gethostnames(procs = workers())

Return the hostname of each worker in `procs`. This is obtained by evaluating 
`Libc.gethostname()` on each worker.
"""
function gethostnames(procs = workers())
	hostnames = Vector{String}(undef,length(procs))
	@sync for (ind,p) in enumerate(procs)
		@async hostnames[ind] = @fetchfrom p Libc.gethostname()
	end
	return hostnames
end

"""
	nodenames(procs = workers())

Return the unique hostnames that the workers in `procs` lie on. 
On an HPC system these are usually the hostnames of the nodes involved.
"""
nodenames(procs = workers()) = nodenames(gethostnames(procs))
function nodenames(hostnames::Vector{String})
	nodes = unique(hostnames)
end

"""
	procs_node(procs = workers())

Return the worker ids on each host of the cluster.
On an HPC system this would return the workers on each node.
"""
function procs_node(procs = workers())
	hosts = gethostnames(procs)
	nodes = nodenames(hosts)
	procs_node(procs,hosts,nodes)
end

function procs_node(procs,hosts,nodes)
	d = OrderedDict{String,Vector{Int}}()
	for node in nodes
		p = procs[findall(isequal(node),hosts)]
		d[node] = p
	end
	return d
end

"""
	nprocs_node(procs = workers())

Return the number of workers on each host.
On an HPC system this would return the number of workers on each node.
"""
function nprocs_node(procs = workers())
	nprocs_node(gethostnames(procs))
end

function nprocs_node(hostnames::Vector{String})
	nodes = nodenames(hostnames)
	nprocs_node(hostnames,nodes)	
end

function nprocs_node(hostnames::Vector{String},nodes::Vector{String})
	Dict(node=>count(isequal(node),hostnames) for node in nodes)
end

function nprocs_node(d::AbstractDict{String,AbstractVector{<:Integer}})
	nphost = OrderedDict{String,Int}()
	for (node,pnode) in d
		nphost[node] = length(pnode)
	end
	return nphost
end


