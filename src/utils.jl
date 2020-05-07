@inline function nworkersactive(iterators::Tuple)
	nt = ntasks(iterators)
	nw = nworkers()
	nt <= nw ? nt : nw
end
@inline nworkersactive(ps::ProductSplit) = nworkersactive(ps.iterators)
@inline nworkersactive(itp::Iterators.ProductIterator) = nworkersactive(itp.iterators)
@inline nworkersactive(args...) = nworkersactive(args)
@inline workersactive(iterators::Tuple) = workers()[1:nworkersactive(iterators)]
@inline workersactive(ps::ProductSplit) = workersactive(ps.iterators)
@inline workersactive(itp::Iterators.ProductIterator) = workersactive(itp.iterators)
@inline workersactive(args...) = workersactive(args)

function gethostnames(procs_used = workers())
	hostnames = Vector{String}(undef,length(procs_used))
	@sync for (ind,p) in enumerate(procs_used)
		@async hostnames[ind] = @fetchfrom p Libc.gethostname()
	end
	return hostnames
end

nodenames(hostnames::Vector{String}) = unique(hostnames)
nodenames(procs_used::Vector{<:Integer} = workers()) = nodenames(gethostnames(procs_used))

function nprocs_node(hostnames::Vector{String})
	nodes = nodenames(hostnames)
	nprocs_node(hostnames,nodes)	
end

function nprocs_node(hostnames::Vector{String},nodes::Vector{String})
	Dict(node=>count(isequal(node),hostnames) for node in nodes)
end

nprocs_node(procs_used::Vector{<:Integer} = workers()) = nprocs_node(gethostnames(procs_used))
