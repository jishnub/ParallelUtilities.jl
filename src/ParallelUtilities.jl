module ParallelUtilities

using Reexport
@reexport using Distributed

export ProductSplit,split_across_processors,split_product_across_processors,
get_pid_of_split_array,get_pid_range_of_split_array,pid_range,
get_index_in_split_array,pid_and_index,
extrema_from_split_array,extrema_common_lastdim,
workers_active,nworkers_active,worker_rank,
pmapsum,pmapreduce,pmap_onebatch_per_worker,
get_nodes,get_hostnames,get_nprocs_node

# The fundamental iterator that behaves like an Iterator.ProductIterator

struct ProcessorNumberError <: Exception 
	p :: Int
	np :: Int
end

function Base.showerror(io::IO,p::ProcessorNumberError)
	print(io,"Processor id $(p.p) does not line in the range $(1:p.np)")
end

struct DecreasingIteratorError <: Exception 
end

function Base.showerror(io::IO,p::DecreasingIteratorError)
	print(io,"All the iterators need to be strictly increasing")
end

struct ProductSplit{T,N,Q}
	iterators :: NTuple{N,Q}
	togglelevels :: NTuple{N,Int}
	np :: Int
	p :: Int
	firstind :: Int
	lastind :: Int

	function ProductSplit(iterators::NTuple{N,Q},togglelevels::NTuple{N,Int},
		np::Int,p::Int,firstind::Int,lastind::Int) where {N,Q<:AbstractRange}

		1 <= p <= np || throw(ProcessorNumberError(p,np))
		T = NTuple{N,eltype(Q)}

		# Check to make sure that all the iterators are increasing
		all(x->step(x)>0,iterators) || throw(DecreasingIteratorError())

		new{T,N,Q}(iterators,togglelevels,np,p,firstind,lastind)
	end
end
Base.eltype(::ProductSplit{T}) where {T} = T

function _cumprod(len)
	(0,_cumprod(first(len),Base.tail(len))...)
end

_cumprod(::Int,::Tuple{}) = ()
function _cumprod(n::Int,tl::Tuple)
	(n,_cumprod(n*first(tl),Base.tail(tl))...)
end

function ProductSplit(iterators::NTuple{N,Q},np,p) where {N,Q}
	T = NTuple{N,eltype(Q)}
	len = Base.Iterators._prod_size(iterators)
	Nel = prod(len)
	togglelevels = _cumprod(len)
	d,r = divrem(Nel,np)
	firstind = d*(p-1) + min(r,p-1) + 1
	lastind = d*p + min(r,p)
	ProductSplit(iterators,togglelevels,np,p,firstind,lastind)
end

@inline Base.@propagate_inbounds Base.first(p::ProductSplit) = 
	_first(p.iterators,childindex(p,p.firstind)...)
	
@inline function _first(t::Tuple,ind::Int,rest::Int...)
	@boundscheck (1 <= ind <= length(first(t))) || throw(BoundsError(first(t),ind))
	(@inbounds first(t)[ind],_first(Base.tail(t),rest...)...)
end
@inline _first(::Tuple{},rest...) = ()

@inline Base.length(p::ProductSplit) = p.lastind - p.firstind + 1
@inline Base.lastindex(p::ProductSplit) = p.lastind - p.firstind + 1

@inline function childindex(p::ProductSplit,ind::Int)
	tl = reverse(Base.tail(p.togglelevels))
	reverse(childindex(tl,ind))
end

@inline function childindex(tl::Tuple,ind::Int)
	t = first(tl)
	k = div(ind-1,t)
	(k+1,childindex(Base.tail(tl),ind-k*t)...)
end

# First iterator gets the final remainder
@inline childindex(::Tuple{},ind::Int) = (ind,)

@inline childindexshifted(p::ProductSplit,ind::Int) = childindex(p, (ind - 1) + p.firstind)

@inline Base.@propagate_inbounds function Base.getindex(p::ProductSplit,ind::Int)
	_getindex(p,childindexshifted(p, ind)...)
end
# This needs to be a separate function to deal with the case of a single child iterator, in which case 
# it's not clear if the single index is for the ProductSplit or the child iterator

# This method asserts that the number of indices are correct
@inline Base.@propagate_inbounds function _getindex(p::ProductSplit{<:Any,N},
	inds::Vararg{Int,N}) where {N}
	
	_getindex(p.iterators,inds...)
end

@inline function _getindex(p::Tuple,ind::Int,rest::Int...)
	@boundscheck (1 <= ind <= length(first(p))) || throw(BoundsError(first(p),ind))
	(@inbounds first(p)[ind],_getindex(Base.tail(p),rest...)...)
end
@inline _getindex(::Tuple{},rest::Int...) = ()

function Base.iterate(p::ProductSplit,state=(first(p),1))
	el,n = state

	if n > length(p)
		return nothing
	elseif n == length(p)
		# In this case the next value doesn't matter, so just return something arbitary
		return (el,(p[1],n+1))
	end

	(el,(p[n+1],n+1))
end

@inline Base.@propagate_inbounds function _firstlastalongdim(p::ProductSplit{<:Any,N},dim::Int,
	firstindchild::Tuple=childindex(p,p.firstind),
	lastindchild::Tuple=childindex(p,p.lastind)) where {N}

	_firstlastalongdim(p.iterators,dim,firstindchild,lastindchild)
end

@inline Base.@propagate_inbounds function _firstlastalongdim(iterators::NTuple{N,<:Any},dim::Int,
	firstindchild::Tuple,lastindchild::Tuple) where {N}

	@boundscheck (1 <= dim <= N) || throw(BoundsError(iterators,dim))

	iter = @inbounds iterators[dim]

	fic = @inbounds firstindchild[dim]
	lic = @inbounds lastindchild[dim]

	first_iter = @inbounds iter[fic]
	last_iter = @inbounds iter[lic]

	(first_iter,last_iter)
end

function _checkrollover(p::ProductSplit{<:Any,N},dim::Int,
	firstindchild::Tuple=childindex(p,p.firstind),
	lastindchild::Tuple=childindex(p,p.lastind)) where {N}

	_checkrollover(p.iterators,dim,firstindchild,lastindchild)
end

function _checkrollover(t::NTuple{N,<:Any},dim::Int,
	firstindchild::Tuple,lastindchild::Tuple) where {N}

	if dim > 0
		return _checkrollover(Base.tail(t),dim-1,Base.tail(firstindchild),Base.tail(lastindchild))
	end

	!_checknorollover(reverse(t),reverse(firstindchild),reverse(lastindchild))
end

function _checknorollover(t,firstindchild,lastindchild)
	iter = first(t)
	first_iter = iter[first(firstindchild)]
	last_iter = iter[first(lastindchild)]

	(last_iter == first_iter) & 
		_checknorollover(Base.tail(t),Base.tail(firstindchild),Base.tail(lastindchild))
end
_checknorollover(::Tuple{},::Tuple{},::Tuple{}) = true

@inline function Base.maximum(p::ProductSplit{<:Any,1},dim::Int=1)
	@boundscheck (dim > 1) && throw(BoundsError(p.iterators,dim))
	lastindchild = childindex(p,p.lastind)
	@inbounds lic_dim = lastindchild[1]
	@inbounds iter = p.iterators[1]
	iter[lic_dim]
end

@inline function Base.maximum(p::ProductSplit{<:Any,N},dim::Int) where {N}

	@boundscheck (1 <= dim <= N) || throw(BoundsError(p.iterators,dim))
	
	firstindchild = childindex(p,p.firstind)
	lastindchild = childindex(p,p.lastind)

	@inbounds first_iter,last_iter = _firstlastalongdim(p,dim,firstindchild,lastindchild)

	v = last_iter

	# The last index will not roll over so this can be handled easily
	if dim == N
		return v
	end

	if _checkrollover(p,dim,firstindchild,lastindchild)
		iter = @inbounds p.iterators[dim]
		v = maximum(iter)
	end

	return v
end

@inline function Base.minimum(p::ProductSplit{<:Any,1},dim::Int=1)
	@boundscheck (dim > 1) && throw(BoundsError(p.iterators,dim))
	firstindchild = childindex(p,p.firstind)
	@inbounds fic_dim = firstindchild[1]
	@inbounds iter = p.iterators[1]
	iter[fic_dim]
end

@inline function Base.minimum(p::ProductSplit{<:Any,N},dim::Int) where {N}
	
	@boundscheck (1 <= dim <= N) || throw(BoundsError(p.iterators,dim))

	firstindchild = childindex(p,p.firstind)
	lastindchild = childindex(p,p.lastind)

	@inbounds first_iter,last_iter = _firstlastalongdim(p,dim,firstindchild,lastindchild)

	v = first_iter

	# The last index will not roll over so this can be handled easily
	if dim == N
		return v
	end

	if _checkrollover(p,dim,firstindchild,lastindchild)
		iter = @inbounds p.iterators[dim]
		v = minimum(iter)
	end

	return v
end

@inline function Base.extrema(p::ProductSplit{<:Any,1},dim::Int=1)
	@boundscheck (dim > 1) && throw(BoundsError(p.iterators,dim))
	firstindchild = childindex(p,p.firstind)
	lastindchild = childindex(p,p.lastind)
	@inbounds fic_dim = firstindchild[1]
	@inbounds lic_dim = lastindchild[1]
	@inbounds iter = p.iterators[1]
	
	(iter[fic_dim],iter[lic_dim])
end

@inline function Base.extrema(p::ProductSplit{<:Any,N},dim::Int) where {N}
	
	@boundscheck (1 <= dim <= N) || throw(BoundsError(p.iterators,dim))

	firstindchild = childindex(p,p.firstind)
	lastindchild = childindex(p,p.lastind)

	@inbounds first_iter,last_iter = _firstlastalongdim(p,dim,firstindchild,lastindchild)

	v = (first_iter,last_iter)
	# The last index will not roll over so this can be handled easily
	if dim == N
		return v
	end

	if _checkrollover(p,dim,firstindchild,lastindchild)
		iter = @inbounds p.iterators[dim]
		v = extrema(iter)
	end

	return v
end

_infullrange(val::T,p::ProductSplit{T}) where {T} = _infullrange(val,p.iterators)

function _infullrange(val,t::Tuple)
	first(val) in first(t) && _infullrange(Base.tail(val),Base.tail(t))
end
_infullrange(::Tuple{},::Tuple{}) = true

# This struct is just a wrapper to flip the tuples before comparing
struct LittleEndianTuple{T}
	t :: T
end

Base.isless(a::LittleEndianTuple{T},b::LittleEndianTuple{T}) where {T} = reverse(a.t) < reverse(b.t)
Base.isequal(a::LittleEndianTuple{T},b::LittleEndianTuple{T}) where {T} = a.t == b.t

function Base.in(val::T,p::ProductSplit{T}) where {T}
	_infullrange(val,p) || return false
	
	val_lt = LittleEndianTuple(val)
	first_iter = LittleEndianTuple(p[1])
	last_iter = LittleEndianTuple(p[end])

	first_iter <= val_lt <= last_iter
end

###################################################################################################

function worker_rank()
	if nworkers()==1
		return 1
	end
	myid()-minimum(workers())+1
end

function split_across_processors(num_tasks::Integer,np=nworkers(),pid=worker_rank())
    split_product_across_processors((1:num_tasks,),np,pid)
end

function split_product_across_processors(iterators::Tuple,
	np::Integer=nworkers(),pid::Integer=worker_rank())
	
	ProductSplit(iterators,np,pid)
end

function get_pid_of_split_array(iterators::Tuple,val::Tuple,np::Int)
	
	_infullrange(val,iterators) || return nothing

	# We may carry out a binary search as the iterators are sorted
	left,right = 1,np

	while left <= right
		mid = floor(Int,(left+right)/2)
		ps = ProductSplit(iterators,np,mid)

		if LittleEndianTuple(val) < LittleEndianTuple(first(ps))
			right = mid - 1
		elseif LittleEndianTuple(val) > LittleEndianTuple(last(ps))
			left = mid + 1
		else
			return mid
		end
	end

	return nothing
end

# This function is necessary when you're changing np
function get_pid_range_of_split_array(ps::ProductSplit,np_new::Int)
	
	if length(ps)==0
		return 0:-1 # empty range
	end

	pid_start = get_pid_of_split_array(ps.iterators,first(ps),np_new)
	if length(ps) == 1
		return pid_start:pid_start
	end

	pid_end = get_pid_of_split_array(ps.iterators,last(ps),np_new)
	return pid_start:pid_end
end

function get_index_in_split_array(ps::ProductSplit{T},val::T) where {T}
	# Can carry out a binary search

	(length(ps) == 0 || val ∉ ps) && return nothing

	left,right = 1,length(ps)

	val == first(ps) && return left
	val == last(ps) && return right

	while left <= right
		mid = floor(Int,(left+right)/2)
		val_mid = @inbounds ps[mid]

		if LittleEndianTuple(val) < LittleEndianTuple(val_mid)
			right = mid - 1
		elseif LittleEndianTuple(val) > LittleEndianTuple(val_mid)
			left = mid + 1
		else
			return mid
		end
	end
	
	return nothing
end

function get_index_in_split_array(iterators::Tuple,val::Tuple,np::Integer,pid::Integer)
	ps = split_product_across_processors(iterators,np,pid)
	get_index_in_split_array(ps,val)
end

function pid_and_index(iterators::Tuple,val::Tuple,np::Integer)
	pid = get_pid_of_split_array(iterators,val,np)
	index = get_index_in_split_array(iterators,val,np,pid)
	return pid,index
end

function pid_range(iterators::Tuple,vals::Tuple,np::Int)
	pid_first = get_pid_of_split_array(iterators,first(vals),np)
	(pid_first,pid_range(iterators,Base.tail(vals),np)...)
end
pid_range(::Tuple,::Tuple{},::Int) = ()

workers_active(arr) = workers()[1:min(length(arr),nworkers())]

workers_active(arrs...) = workers_active(Iterators.product(arrs...))

nworkers_active(args...) = length(workers_active(args...))

function extrema_from_split_array(ps::ProductSplit)
	_extrema_from_split_array(ps,1,ps.iterators)
end

function _extrema_from_split_array(ps::ProductSplit,dim::Int,iterators::Tuple)
	(extrema(ps,dim),_extrema_from_split_array(ps,dim+1,Base.tail(iterators))...)
end
_extrema_from_split_array(::ProductSplit,::Int,::Tuple{}) = ()

function extrema_common_lastdim(ps::ProductSplit{<:Any,N}) where {N}
	m = extrema_from_split_array(ps)
	lastvar_min = last(m)[1]
	lastvar_max = last(m)[2]

	val_first = first(ps)
	val_last = last(ps)
	min_vals = collect(val_first[1:end-1])
	max_vals = collect(val_last[1:end-1])

	for val in ps
		val_rev = reverse(val)
		lastvar = first(val_rev)
		(lastvar_min < lastvar < lastvar_max) && continue

		for (ind,vi) in enumerate(Base.tail(val_rev))
			if lastvar==lastvar_min
				min_vals[N-ind] = min(min_vals[N-ind],vi)
			end
			if lastvar==lastvar_max
				max_vals[N-ind] = max(max_vals[N-ind],vi)
			end
		end
	end

	[(m,lastvar_min) for m in min_vals],[(m,lastvar_max) for m in max_vals]
end

function get_hostnames(procs_used=workers())
	hostnames = Vector{String}(undef,length(procs_used))
	@sync for (ind,p) in enumerate(procs_used)
		@async hostnames[ind] = @fetchfrom p Libc.gethostname()
	end
	return hostnames
end

get_nodes(hostnames::Vector{String}) = unique(hostnames)
get_nodes(procs_used::Vector{<:Integer}=workers()) = get_nodes(get_hostnames(procs_used))

function get_nprocs_node(hostnames::Vector{String})
	nodes = get_nodes(hostnames)
	get_nprocs_node(hostnames,nodes)	
end

function get_nprocs_node(hostnames::Vector{String},nodes::Vector{String})
	Dict(node=>count(isequal(node),hostnames) for node in nodes)
end

get_nprocs_node(procs_used::Vector{<:Integer}=workers()) = get_nprocs_node(get_hostnames(procs_used))

# This function does not sort the values, so it might be faster
function pmapsum_remotechannel(::Type{T},f::Function,iterable,args...;kwargs...) where {T}

	procs_used = workers_active(iterable)

	num_workers = length(procs_used);
	hostnames = get_hostnames(procs_used);
	nodes = get_nodes(hostnames);
	pid_rank0_on_node = [procs_used[findfirst(isequal(node),hostnames)] for node in nodes];

	nprocs_node = get_nprocs_node(procs_used)
	node_channels = Dict(node=>RemoteChannel(()->Channel{T}(nprocs_node[node]),pid_node)
		for (node,pid_node) in zip(nodes,pid_rank0_on_node))

	# Worker at which final reduction takes place
	p_final = first(pid_rank0_on_node)

	sum_channel = RemoteChannel(()->Channel{T}(length(pid_rank0_on_node)),p_final)
	result = nothing

	# Run the function on each processor and compute the sum at each node
	@sync for (rank,(p,node)) in enumerate(zip(procs_used,hostnames))
		@async begin
			
			node_remotechannel = node_channels[node]
			np_node = nprocs_node[node]
			
			iterable_on_proc = split_across_processors(iterable,num_workers,rank)
			@spawnat p put!(node_remotechannel,
				f(iterable_on_proc,args...;kwargs...))

			@async if p in pid_rank0_on_node
				f = @spawnat p put!(sum_channel,
					sum(take!(node_remotechannel) for i=1:np_node))
				wait(f)
				@spawnat p finalize(node_remotechannel)
			end

			@async if p==p_final
				result = @fetchfrom p_final sum(take!(sum_channel)
					for i=1:length(pid_rank0_on_node))
				@spawnat p finalize(sum_channel)
			end
		end
	end

	return result :: T
end

# Store the processor id with the value
struct pval{T}
	p :: Int
	parent :: T
end

function pmapreduce_remotechannel(::Type{T},fmap::Function,freduce::Function,
	iterable,args...;kwargs...) where {T}

	procs_used = workers_active(iterable)

	num_workers = length(procs_used);
	hostnames = get_hostnames(procs_used);
	nodes = get_nodes(hostnames);
	pid_rank0_on_node = [procs_used[findfirst(isequal(node),hostnames)] for node in nodes];

	nprocs_node = get_nprocs_node(procs_used)
	node_channels = Dict(node=>RemoteChannel(()->Channel{T}(nprocs_node[node]),pid_node)
		for (node,pid_node) in zip(nodes,pid_rank0_on_node))

	# Worker at which final reduction takes place
	p_final = first(pid_rank0_on_node)

	reduce_channel = RemoteChannel(()->Channel{T}(length(pid_rank0_on_node)),p_final)
	result = nothing

	# Run the function on each processor and compute the sum at each node
	@sync for (rank,(p,node)) in enumerate(zip(procs_used,hostnames))
		@async begin
			
			node_remotechannel = node_channels[node]
			np_node = nprocs_node[node]
			
			iterable_on_proc = split_across_processors(iterable,num_workers,rank)
			@spawnat p put!(node_remotechannel,
				pval(p,fmap(iterable_on_proc,args...;kwargs...)))

			@async if p in pid_rank0_on_node
				f = @spawnat p begin 
					vals = [take!(node_remotechannel) for i=1:np_node ]
					sort!(vals,by=x->x.p)
					put!(reduce_channel,pval(p,freduce(v.parent for v in vals))	)
				end
				wait(f)
				@spawnat p finalize(node_remotechannel)
			end

			@async if p==p_final
				result = @fetchfrom p_final begin
					vals = [take!(reduce_channel) for i=1:length(pid_rank0_on_node)]
					sort!(vals,by=x->x.p)
					freduce(v.parent for v in vals)
				end
				@spawnat p finalize(reduce_channel)
			end
		end
	end

	return result :: T
end

function pmapsum_remotechannel(f::Function,iterable,args...;kwargs...)
	pmapsum_remotechannel(Any,f,iterable,args...;kwargs...)
end

function pmapreduce_remotechannel(fmap::Function,freduce::Function,iterable,args...;kwargs...)
	pmapreduce_remotechannel(Any,fmap,freduce,iterable,args...;kwargs...)
end

function pmapsum_distributedfor(f::Function,iterable,args...;kwargs...)
	@distributed (+) for i in 1:nworkers()
		np = nworkers_active(iterable)
		iter_proc = split_across_processors(iterable,np,i)
		f(iter_proc,args...;kwargs...)
	end
end

function pmapreduce_distributedfor(fmap::Function,freduce::Function,iterable,args...;kwargs...)
	@distributed freduce for i in 1:nworkers()
		np = nworkers_active(iterable)
		iter_proc = split_across_processors(iterable,np,i)
		fmap(iter_proc,args...;kwargs...)
	end
end

pmapsum(args...;kwargs...) = pmapsum_remotechannel(args...;kwargs...)
pmapreduce(args...;kwargs...) = pmapreduce_remotechannel(args...;kwargs...)

function pmap_onebatch_per_worker(f::Function,iterable,args...;kwargs...)

	procs_used = workers_active(iterable)
	num_workers = get(kwargs,:num_workers,length(procs_used))
	if num_workers<length(procs_used)
		procs_used = procs_used[1:num_workers]
	end
	num_workers = length(procs_used)

	futures = Vector{Future}(undef,num_workers)
	@sync for (rank,p) in enumerate(procs_used)
		@async begin
			iterable_on_proc = split_across_processors(iterable,num_workers,rank)
			futures[rank] = @spawnat p f(iterable_on_proc,args...;kwargs...)
		end
	end
	return futures
end

end # module
