module ParallelUtilities

using Reexport
@reexport using Distributed

export ProductSplit,split_across_processors,split_product_across_processors,
get_processor_id_from_split_array,procid_allmodes,mode_index_in_file,
get_processor_range_from_split_array,workers_active,nworkers_active,worker_rank,
get_index_in_split_array,procid_and_mode_index,extrema_from_split_array,
pmapsum,pmapreduce,pmap_onebatch_per_worker,moderanges_common_lastarray,
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

struct OrderedTuple{T}
	t :: T
end

function Base.:<=(a::OrderedTuple{T},b::OrderedTuple{T}) where {T}
	_le(reverse(a.t),reverse(b.t))	
end

function _le(t1::Tuple,t2::Tuple)
	first(t1) < first(t2) || ((first(t1) == first(t2)) & _le(Base.tail(t1),Base.tail(t2)))
end
_le(::Tuple{},::Tuple{}) = true

function Base.in(val::T,p::ProductSplit{T}) where {T}
	_infullrange(val,p) || return false
	
	val_ot = OrderedTuple(val)
	first_iter = OrderedTuple(p[1])
	last_iter = OrderedTuple(p[end])

	first_iter <= val_ot <= last_iter
end

###################################################################################################

function worker_rank()
	if nworkers()==1
		return 1
	end
	myid()-minimum(workers())+1
end

# function split_across_processors_iterators(arr₁::Base.Iterators.ProductIterator,num_procs,proc_id)

#     @assert(proc_id<=num_procs,"processor rank has to be less than number of workers engaged")

#     num_tasks = length(arr₁);

#     num_tasks_per_process,num_tasks_leftover = div(num_tasks,num_procs),mod(num_tasks,num_procs)

#     num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
#     task_start = num_tasks_per_process*(proc_id-1) + min(num_tasks_leftover,proc_id-1) + 1;

#     Iterators.take(Iterators.drop(arr₁,task_start-1),num_tasks_on_proc)
# end

# function split_product_across_processors_iterators(arrs_tuple,num_procs,proc_id)
# 	split_across_processors_iterators(Iterators.product(arrs_tuple...),num_procs,proc_id)
# end

function split_across_processors(num_tasks::Integer,num_procs=nworkers(),proc_id=worker_rank())
    split_product_across_processors((1:num_tasks,),num_procs,proc_id)
end

function split_product_across_processors(arrs_tuple::Tuple,
	num_procs::Integer=nworkers(),proc_id::Integer=worker_rank())
	
	ProductSplit(arrs_tuple,num_procs,proc_id)
end

function get_processor_id_from_split_array(arr₁::AbstractVector,arr₂::AbstractVector,
	(arr₁_value,arr₂_value)::Tuple,num_procs::Integer)
	# Find the closest match in arrays

	if (arr₁_value ∉ arr₁) || (arr₂_value ∉ arr₂)
		return nothing # invalid
	end
	
	num_tasks = length(arr₁)*length(arr₂);

	a1_match_index = searchsortedfirst(arr₁,arr₁_value)
	a2_match_index = searchsortedfirst(arr₂,arr₂_value)

	num_tasks_per_process,num_tasks_leftover = div(num_tasks,num_procs),mod(num_tasks,num_procs)

	proc_id = 1
	num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
	total_tasks_till_proc_id = num_tasks_on_proc

	task_no = 0

	for (ind2,a2) in enumerate(arr₂), (ind1,a1) in enumerate(arr₁)
		
		task_no +=1
		if task_no > total_tasks_till_proc_id
			proc_id += 1
			num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
			total_tasks_till_proc_id += num_tasks_on_proc
		end

		if ind2< a2_match_index
			continue
		end

		if (ind2 == a2_match_index) && (ind1 == a1_match_index)
			break
		end
	end

	return proc_id
end

function get_processor_id_from_split_array(iterators,val,num_procs)
	for proc_id in 1:num_procs
		tasks_on_proc = split_product_across_processors(iterators,num_procs,proc_id)
		if val ∈ tasks_on_proc
			return proc_id
		end
	end
	return nothing
end

function get_processor_id_from_split_array(iter::ProductSplit{T},val::T) where {T}
	get_processor_id_from_split_array(iter.iterators_product,val,iter.num_procs)
end

function get_processor_range_from_split_array(iter,vals,num_procs::Integer)
	
	if isempty(vals)
		return 0:-1 # empty range
	end

	first_task = first(vals) 
	proc_id_start = get_processor_id_from_split_array(iter,first_task,num_procs)

	last_task = first_task
	if length(vals) == 1
		return proc_id_start:proc_id_start
	end

	for t in vals
		last_task = t
	end

	proc_id_end = get_processor_id_from_split_array(iter,last_task,num_procs)
	return proc_id_start:proc_id_end
end

function get_processor_range_from_split_array(iter::ProductSplit{T},val::T) where {T}
	get_processor_range_from_split_array(iter.iterators_product,val,iterators.num_procs)
end

get_processor_range_from_split_array(arr₁::AbstractVector,arr₂::AbstractVector,
	vals,num_procs::Integer) =
	get_processor_range_from_split_array(Iterators.product(arr₁,arr₂),
		vals,num_procs)

function get_index_in_split_array(iter_section,val::Tuple)
	if isnothing(iter_section)
		return nothing
	end
	for (ind,val_ind) in enumerate(iter_section)
		if val_ind == val
			return ind
		end
	end
	nothing
end

function procid_and_mode_index(arr₁::AbstractVector,arr₂::AbstractVector,
	(arr₁_value,arr₂_value)::Tuple,num_procs::Integer)
	proc_id_mode = get_processor_id_from_split_array(arr₁,arr₂,(arr₁_value,arr₂_value),num_procs)
	modes_in_procid_file = split_product_across_processors(arr₁,arr₂,num_procs,proc_id_mode)
	mode_index = get_index_in_split_array(modes_in_procid_file,(arr₁_value,arr₂_value))
	return proc_id_mode,mode_index
end

function procid_and_mode_index(iter,val::Tuple,num_procs::Integer)

	proc_id_mode = get_processor_id_from_split_array(iter,val,num_procs)
	modes_in_procid_file = split_across_processors(iter,num_procs,proc_id_mode)
	mode_index = get_index_in_split_array(modes_in_procid_file,val)
	return proc_id_mode,mode_index
end

function mode_index_in_file(arr₁::AbstractVector,arr₂::AbstractVector,
	(arr₁_value,arr₂_value)::Tuple,num_procs::Integer,proc_id_mode::Integer)

	modes_in_procid_file = split_product_across_processors(arr₁,arr₂,num_procs,proc_id_mode)
	mode_index = get_index_in_split_array(modes_in_procid_file,(arr₁_value,arr₂_value))
end

function procid_allmodes(arr₁::AbstractVector,arr₂::AbstractVector,
	iter,num_procs=nworkers_active(arr₁,arr₂))
	procid = zeros(Int64,length(iter))
	for (ind,mode) in enumerate(iter)
		procid[ind] = get_processor_id_from_split_array(arr₁,arr₂,mode,num_procs)
	end
	return procid
end

workers_active(arr) = workers()[1:min(length(arr),nworkers())]

workers_active(arrs...) = workers_active(Iterators.product(arrs...))

nworkers_active(args...) = length(workers_active(args...))

function extrema_from_split_array(iterable)
	val_first = first(iterable)
	min_vals = collect(val_first)
	max_vals = copy(min_vals)

	for val in iterable
		for (ind,vi) in enumerate(val)
			min_vals[ind] = min(min_vals[ind],vi)
			max_vals[ind] = max(max_vals[ind],vi)
		end
	end
	collect(zip(min_vals,max_vals))
end

function moderanges_common_lastarray(iterable)
	m = extrema_from_split_array(iterable)
	lastvar_min = last(m)[1]
	lastvar_max = last(m)[2]

	val_first = first(iterable)
	min_vals = collect(val_first[1:end-1])
	max_vals = copy(min_vals)

	for val in iterable
		for (ind,vi) in enumerate(val[1:end-1])
			if val[end]==lastvar_min
				min_vals[ind] = min(min_vals[ind],vi)
			end
			if val[end]==lastvar_max
				max_vals[ind] = max(max_vals[ind],vi)
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
