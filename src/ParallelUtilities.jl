module ParallelUtilities

using Reexport
@reexport using Distributed

export  ProductSplit,
	evenlyscatterproduct,
	ntasks,
	whichproc,
	whichproc,
	procrange_recast,
	localindex,
	procid_and_localindex,
	extremadims,
	extrema_commonlastdim,
	workersactive,
	nworkersactive,
	workerrank,
	nodenames,
	gethostnames,
	nprocs_node,
	pmapbatch,
	pmapbatch_elementwise,
	pmapsum,
	pmapsum_elementwise,
	pmapreduce,
	pmapreduce_commutative,
	pmapreduce_commutative_elementwise

# The fundamental iterator that behaves like an Iterator.Take{Iterator.Drop{Iterator.ProductIterator}}

struct ProcessorNumberError <: Exception 
	p :: Int
	np :: Int
end
function Base.showerror(io::IO,err::ProcessorNumberError)
	print(io,"processor id $(err.p) does not line in the range $(1:err.np)")
end

struct DecreasingIteratorError <: Exception 
end
function Base.showerror(io::IO,err::DecreasingIteratorError)
	print(io,"all the iterators need to be strictly increasing")
end

struct TaskNotPresentError{T,U} <: Exception
	t :: T
	task :: U
end
function Base.showerror(io::IO,err::TaskNotPresentError)
	print(io,"could not find the task $task in the list $t")
end

struct BoundsErrorPS{T}
	ps :: T
	ind :: Int
end
function Base.showerror(io::IO,err::BoundsErrorPS)
	print(io,"attempt to access $(length(err.ps))-element ProductSplit at index $(err.ind)")
end

struct ProductSplit{T,N,Q}
	iterators :: Q
	togglelevels :: NTuple{N,Int}
	np :: Int
	p :: Int
	firstind :: Int
	lastind :: Int

	function ProductSplit(iterators::Tuple{Vararg{AbstractRange,N}},togglelevels::NTuple{N,Int},
		np::Int,p::Int,firstind::Int,lastind::Int) where {N}

		1 <= p <= np || throw(ProcessorNumberError(p,np))
		T = Tuple{map(eltype,iterators)...}
		Q = typeof(iterators)

		# Ensure that all the iterators are strictly increasing
		all(x->step(x)>0,iterators) || throw(DecreasingIteratorError())

		new{T,N,Q}(iterators,togglelevels,np,p,firstind,lastind)
	end
end
Base.eltype(::ProductSplit{T}) where {T} = T

function _cumprod(len::Tuple)
	(0,_cumprod(first(len),Base.tail(len))...)
end

@inline _cumprod(::Int,::Tuple{}) = ()
function _cumprod(n::Int,tl::Tuple)
	(n,_cumprod(n*first(tl),Base.tail(tl))...)
end

@inline ntasks(tl::Tuple) = prod(map(length,tl))
@inline ntasks(ps::ProductSplit) = ntasks(ps.iterators)

function ProductSplit(iterators::Tuple{Vararg{AbstractRange}},np::Int,p::Int)
	len = Base.Iterators._prod_size(iterators)
	Nel = prod(len)
	togglelevels = _cumprod(len)
	d,r = divrem(Nel,np)
	firstind = d*(p-1) + min(r,p-1) + 1
	lastind = d*p + min(r,p)
	ProductSplit(iterators,togglelevels,np,p,firstind,lastind)
end
ProductSplit(::Tuple{},::Int,::Int) = throw(ArgumentError("Need at least one iterator"))

Base.isempty(ps::ProductSplit) = (ps.firstind > ps.lastind)

@inline Base.@propagate_inbounds function Base.first(ps::ProductSplit)
	isempty(ps) ? nothing : _first(ps.iterators,childindex(ps,ps.firstind)...)
end

@inline Base.@propagate_inbounds function _first(t::Tuple,ind::Int,rest::Int...)
	@boundscheck (1 <= ind <= length(first(t))) || throw(BoundsError(first(t),ind))
	(@inbounds first(t)[ind],_first(Base.tail(t),rest...)...)
end
@inline _first(::Tuple{}) = ()

@inline Base.@propagate_inbounds function Base.last(ps::ProductSplit)
	isempty(ps) ? nothing : _last(ps.iterators,childindex(ps,ps.lastind)...)
end

@inline Base.@propagate_inbounds function _last(t::Tuple,ind::Int,rest::Int...)
	@boundscheck (1 <= ind <= length(first(t))) || throw(BoundsError(first(t),ind))
	(@inbounds first(t)[ind],_last(Base.tail(t),rest...)...)
end
@inline _last(::Tuple{}) = ()

@inline Base.length(ps::ProductSplit) = ps.lastind - ps.firstind + 1

@inline Base.firstindex(ps::ProductSplit) = 1
@inline Base.lastindex(ps::ProductSplit) = ps.lastind - ps.firstind + 1

@inline function childindex(ps::ProductSplit,ind::Int)
	tl = reverse(Base.tail(ps.togglelevels))
	reverse(childindex(tl,ind))
end

@inline function childindex(tl::Tuple,ind::Int)
	t = first(tl)
	k = div(ind-1,t)
	(k+1,childindex(Base.tail(tl),ind-k*t)...)
end

# First iterator gets the final remainder
@inline childindex(::Tuple{},ind::Int) = (ind,)

@inline childindexshifted(ps::ProductSplit,ind::Int) = childindex(ps, (ind - 1) + ps.firstind)

@inline Base.@propagate_inbounds function Base.getindex(ps::ProductSplit,ind::Int)
	@boundscheck 1 <= ind <= length(ps) || throw(BoundsErrorPS(ps,ind))
	_getindex(ps,childindexshifted(ps, ind)...)
end
# This needs to be a separate function to deal with the case of a single child iterator, in which case 
# it's not clear if the single index is for the ProductSplit or the child iterator

# This method asserts that the number of indices are correct
@inline Base.@propagate_inbounds function _getindex(ps::ProductSplit{<:Any,N},
	inds::Vararg{Int,N}) where {N}
	
	_getindex(ps.iterators,inds...)
end

@inline function _getindex(t::Tuple,ind::Int,rest::Int...)
	@boundscheck (1 <= ind <= length(first(t))) || throw(BoundsError(first(t),ind))
	(@inbounds first(t)[ind],_getindex(Base.tail(t),rest...)...)
end
@inline _getindex(::Tuple{},rest::Int...) = ()

function Base.iterate(ps::ProductSplit,state=(first(ps),1))
	el,n = state

	if n > length(ps)
		return nothing
	elseif n == length(ps)
		# In this case the next value doesn't matter, so just return something arbitary
		return (el,(first(ps),n+1))
	end

	(el,(ps[n+1],n+1))
end

@inline Base.@propagate_inbounds function _firstlastalongdim(pss::ProductSplit{<:Any,N},dim::Int,
	firstindchild::Tuple=childindex(ps,ps.firstind),
	lastindchild::Tuple=childindex(ps,ps.lastind)) where {N}

	_firstlastalongdim(pss.iterators,dim,firstindchild,lastindchild)
end

@inline Base.@propagate_inbounds function _firstlastalongdim(iterators::Tuple{Vararg{Any,N}},dim::Int,
	firstindchild::Tuple,lastindchild::Tuple) where {N}

	@boundscheck (1 <= dim <= N) || throw(BoundsError(iterators,dim))

	iter = @inbounds iterators[dim]

	fic = @inbounds firstindchild[dim]
	lic = @inbounds lastindchild[dim]

	first_iter = @inbounds iter[fic]
	last_iter = @inbounds iter[lic]

	(first_iter,last_iter)
end

function _checkrollover(ps::ProductSplit{<:Any,N},dim::Int,
	firstindchild::Tuple=childindex(ps,ps.firstind),
	lastindchild::Tuple=childindex(ps,ps.lastind)) where {N}

	_checkrollover(ps.iterators,dim,firstindchild,lastindchild)
end

function _checkrollover(t::Tuple{Vararg{Any,N}},dim::Int,
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

@inline function Base.maximum(ps::ProductSplit{<:Any,1},dim::Int=1)
	@boundscheck (dim > 1) && throw(BoundsError(ps.iterators,dim))
	isempty(ps) && return nothing
	lastindchild = childindex(ps,ps.lastind)
	@inbounds lic_dim = lastindchild[1]
	@inbounds iter = ps.iterators[1]
	iter[lic_dim]
end

@inline function Base.maximum(ps::ProductSplit{<:Any,N},dim::Int) where {N}

	@boundscheck (1 <= dim <= N) || throw(BoundsError(ps.iterators,dim))

	isempty(ps) && return nothing
	
	firstindchild = childindex(ps,ps.firstind)
	lastindchild = childindex(ps,ps.lastind)

	@inbounds first_iter,last_iter = _firstlastalongdim(ps,dim,firstindchild,lastindchild)

	v = last_iter

	# The last index will not roll over so this can be handled easily
	if dim == N
		return v
	end

	if _checkrollover(ps,dim,firstindchild,lastindchild)
		iter = @inbounds ps.iterators[dim]
		v = maximum(iter)
	end

	return v
end

@inline function Base.minimum(ps::ProductSplit{<:Any,1},dim::Int=1)
	@boundscheck (dim > 1) && throw(BoundsError(ps.iterators,dim))
	isempty(ps) && return nothing
	firstindchild = childindex(ps,ps.firstind)
	@inbounds fic_dim = firstindchild[1]
	@inbounds iter = ps.iterators[1]
	iter[fic_dim]
end

@inline function Base.minimum(ps::ProductSplit{<:Any,N},dim::Int) where {N}
	
	@boundscheck (1 <= dim <= N) || throw(BoundsError(ps.iterators,dim))

	isempty(ps) && return nothing

	firstindchild = childindex(ps,ps.firstind)
	lastindchild = childindex(ps,ps.lastind)

	@inbounds first_iter,last_iter = _firstlastalongdim(ps,dim,firstindchild,lastindchild)

	v = first_iter

	# The last index will not roll over so this can be handled easily
	if dim == N
		return v
	end

	if _checkrollover(ps,dim,firstindchild,lastindchild)
		iter = @inbounds ps.iterators[dim]
		v = minimum(iter)
	end

	return v
end

@inline function Base.extrema(ps::ProductSplit{<:Any,1},dim::Int=1)
	@boundscheck (dim > 1) && throw(BoundsError(ps.iterators,dim))
	isempty(ps) && return nothing
	firstindchild = childindex(ps,ps.firstind)
	lastindchild = childindex(ps,ps.lastind)
	@inbounds fic_dim = firstindchild[1]
	@inbounds lic_dim = lastindchild[1]
	@inbounds iter = ps.iterators[1]
	
	(iter[fic_dim],iter[lic_dim])
end

@inline function Base.extrema(ps::ProductSplit{<:Any,N},dim::Int) where {N}
	
	@boundscheck (1 <= dim <= N) || throw(BoundsError(ps.iterators,dim))

	isempty(ps) && return nothing

	firstindchild = childindex(ps,ps.firstind)
	lastindchild = childindex(ps,ps.lastind)

	@inbounds first_iter,last_iter = _firstlastalongdim(ps,dim,firstindchild,lastindchild)

	v = (first_iter,last_iter)
	# The last index will not roll over so this can be handled easily
	if dim == N
		return v
	end

	if _checkrollover(ps,dim,firstindchild,lastindchild)
		iter = @inbounds ps.iterators[dim]
		v = extrema(iter)
	end

	return v
end

extremadims(ps::ProductSplit) = _extremadims(ps,1,ps.iterators)

function _extremadims(ps::ProductSplit,dim::Int,iterators::Tuple)
	(extrema(ps,dim),_extremadims(ps,dim+1,Base.tail(iterators))...)
end
_extremadims(::ProductSplit,::Int,::Tuple{}) = ()

function extrema_commonlastdim(ps::ProductSplit{<:Any,N}) where {N}

	isempty(ps) && return nothing
	
	m = extremadims(ps)
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

_infullrange(val::T,ps::ProductSplit{T}) where {T} = _infullrange(val,ps.iterators)

function _infullrange(val,t::Tuple)
	first(val) in first(t) && _infullrange(Base.tail(val),Base.tail(t))
end
@inline _infullrange(::Tuple{},::Tuple{}) = true

# This struct is just a wrapper to flip the tuples before comparing
struct ReverseLexicographicTuple{T}
	t :: T
end

Base.isless(a::ReverseLexicographicTuple{T},b::ReverseLexicographicTuple{T}) where {T} = reverse(a.t) < reverse(b.t)
Base.isequal(a::ReverseLexicographicTuple{T},b::ReverseLexicographicTuple{T}) where {T} = a.t == b.t

function Base.in(val::T,ps::ProductSplit{T}) where {T}
	_infullrange(val,ps) || return false
	
	val_lt = ReverseLexicographicTuple(val)
	first_iter = ReverseLexicographicTuple(ps[1])
	last_iter = ReverseLexicographicTuple(ps[end])

	first_iter <= val_lt <= last_iter
end

###################################################################################################

function evenlyscatterproduct(num_tasks::Integer,np=nworkers(),procid=workerrank())
    evenlyscatterproduct((1:num_tasks,),np,procid)
end

function evenlyscatterproduct(iterators::Tuple,
	np::Integer=nworkers(),procid::Integer=workerrank())
	
	ProductSplit(iterators,np,procid)
end

evenlyscatterproduct(itp::Iterators.ProductIterator,args...) = 
	evenlyscatterproduct(itp.iterators,args...)

function whichproc(iterators::Tuple,val::Tuple,np::Int)
	
	_infullrange(val,iterators) || return nothing

	# We may carry out a binary search as the iterators are sorted
	left,right = 1,np

	val_t = ReverseLexicographicTuple(val)

	while left <= right
		mid = floor(Int,(left+right)/2)
		ps = ProductSplit(iterators,np,mid)

		# If np is greater than the number of ntasks then it's possible
		# that ps is empty. In this case the value must be somewhere in
		# the previous workers. Otherwise each worker has some tasks and 
		# these are sorted, so carry out a binary seaarch

		if isempty(ps) || val_t < ReverseLexicographicTuple(first(ps))
			right = mid - 1
		elseif val_t > ReverseLexicographicTuple(last(ps))
			left = mid + 1
		else
			return mid
		end
	end

	return nothing
end

whichproc(iterators::Tuple,::Nothing,np::Int) = nothing

# This function tells us the range of processors that would be involved
# if we are to compute the tasks contained in the list ps on np_new processors.
# The total list of tasks is contained in iterators, and might differ from 
# ps.iterators (eg if ps contains a subsection of the parameter set)
function procrange_recast(iterators::Tuple,ps::ProductSplit,np_new::Int)
	
	if isempty(ps)
		return 0:-1 # empty range
	end

	procid_start = whichproc(iterators,first(ps),np_new)
	if procid_start === nothing
		throw(TaskNotPresentError(iterators,first(ps)))
	end
	if length(ps) == 1
		procid_end = procid_start
	else
		procid_end = whichproc(iterators,last(ps),np_new)
		if procid_end === nothing
			throw(TaskNotPresentError(iterators,last(ps)))
		end
	end
	
	return procid_start:procid_end
end

procrange_recast(ps::ProductSplit,np_new::Int) = procrange_recast(ps.iterators,ps,np_new)

function localindex(ps::ProductSplit{T},val::T) where {T}
	# Can carry out a binary search

	(isempty(ps) || val ∉ ps) && return nothing

	left,right = 1,length(ps)

	val == first(ps) && return left
	val == last(ps) && return right

	val_t = ReverseLexicographicTuple(val)

	while left <= right
		mid = floor(Int,(left+right)/2)
		val_mid = @inbounds ps[mid]

		if val_t < ReverseLexicographicTuple(val_mid)
			right = mid - 1
		elseif val_t > ReverseLexicographicTuple(val_mid)
			left = mid + 1
		else
			return mid
		end
	end
	
	return nothing
end

localindex(::ProductSplit,::Nothing) = nothing

function localindex(iterators::Tuple,val::Tuple,np::Integer,procid::Integer)
	ps = evenlyscatterproduct(iterators,np,procid)
	localindex(ps,val)
end

function procid_and_localindex(iterators::Tuple,val::Tuple,np::Integer)
	procid = whichproc(iterators,val,np)
	index = localindex(iterators,val,np,procid)
	return procid,index
end

function workerrank()
	rank = 1
	if myid() in workers()
		rank = myid()-minimum(workers())+1
	end
	return rank
end

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

############################################################################################
# pmapsum and pmapreduce
############################################################################################

# This function does not sort the values, so it might be faster
function pmapreduce_commutative(fmap::Function,freduce::Function,iterators::Tuple,args...;kwargs...)

	procs_used = workersactive(iterators)

	num_workers_active = length(procs_used);
	hostnames = gethostnames(procs_used);
	nodes = nodenames(hostnames);
	procid_rank1_on_node = [procs_used[findfirst(isequal(node),hostnames)] for node in nodes];
	Nnodes_reduction = length(procid_rank1_on_node)

	nprocs_node_dict = nprocs_node(procs_used)
	node_channels = Dict(
		node=>(
			out = RemoteChannel(()->Channel{Any}(nprocs_node_dict[node]),procid_node),
			err = RemoteChannel(()->Channel{Bool}(nprocs_node_dict[node]),procid_node),
			)
			for (node,procid_node) in zip(nodes,procid_rank1_on_node))

	# Worker at which the final reduction takes place
	p_final = first(procid_rank1_on_node)

	finalnode_reducechannel = RemoteChannel(()->Channel{Any}(length(procid_rank1_on_node)),p_final)
	finalnode_errorchannel = RemoteChannel(()->Channel{Bool}(length(procid_rank1_on_node)),p_final)

	result_channel = RemoteChannel(()->Channel{Any}(1))
	error_channel = RemoteChannel(()->Channel{Bool}(1))

	# Run the function on each processor and compute the reduction at each node
	@sync for (rank,(p,node)) in enumerate(zip(procs_used,hostnames))
		@async begin
			
			eachnode_reducechannel = node_channels[node].out
			eachnode_errorchannel = node_channels[node].err

			p_parent = eachnode_reducechannel.where
			
			np_node = nprocs_node_dict[node]
			
			iterable_on_proc = evenlyscatterproduct(iterators,num_workers_active,rank)

			@spawnat p begin
				try
					res = fmap(iterable_on_proc,args...;kwargs...)
					put!(eachnode_reducechannel,res)
					put!(eachnode_errorchannel,false)
				catch e
					put!(eachnode_errorchannel,true)
					rethrow()
				finally
					# Don't need references to the intermediate reduction channels on other nodes
					for (node_i,channels_i) in node_channels
						if node_i == node
							continue
						end

						finalize(channels_i.out)
						finalize(channels_i.err)
					end
					if p != p_parent
						finalize(eachnode_errorchannel)
						finalize(eachnode_reducechannel)
						finalize(finalnode_errorchannel)
						finalize(finalnode_reducechannel)
					end
					if p != p_final
						if p != result_channel.where
							finalize(result_channel)
						end
						if p != error_channel.where
							finalize(error_channel)
						end
					end
				end
			end

			@async if p == p_parent
				@spawnat p begin
					try
						anyerror = any(take!(eachnode_errorchannel) for i=1:np_node)
						if !anyerror
							res = freduce(take!(eachnode_reducechannel) for i=1:np_node)
							put!(finalnode_reducechannel,res)
							put!(finalnode_errorchannel,false)
						else
							put!(finalnode_errorchannel,true)
						end
					catch e
						put!(finalnode_errorchannel,true)
						rethrow()
					finally
						finalize(eachnode_errorchannel)
						finalize(eachnode_reducechannel)

						if p != p_final
							finalize(finalnode_errorchannel)
							finalize(finalnode_reducechannel)
						end
					end
				end
			end

			@async if p == p_final
				@spawnat p begin
					try
						anyerror = any(take!(finalnode_errorchannel) for i=1:Nnodes_reduction)
						if !anyerror
							res = freduce(take!(finalnode_reducechannel) for i=1:Nnodes_reduction)
							put!(result_channel,res)
							put!(error_channel,false)
						else
							put!(error_channel,true)
						end
					catch e
						put!(error_channel,true)
						rethrow()
					finally
						finalize(finalnode_errorchannel)
						finalize(finalnode_reducechannel)

						if p != result_channel.where
							finalize(result_channel)
						end
						if p != error_channel.where
							finalize(error_channel)
						end
					end
				end
			end
		end
	end

	anyerror = take!(error_channel)
	if !anyerror
		return take!(result_channel)
	end
end

function pmapreduce_commutative(fmap::Function,freduce::Function,
	itp::Iterators.ProductIterator,args...;kwargs...)

	pmapreduce_commutative(fmap,freduce,itp.iterators,args...;kwargs...)
end

function pmapreduce_commutative(fmap::Function,freduce::Function,iterable,args...;kwargs...)
	pmapreduce_commutative(fmap,freduce,(iterable,),args...;kwargs...)
end

function pmapreduce_commutative_elementwise(fmap::Function,freduce::Function,iterable,args...;kwargs...)
	pmapreduce_commutative(plist->freduce(asyncmap(x->fmap(x...,args...;kwargs...),plist)),
		freduce,iterable,args...;kwargs...)
end

function pmapsum(fmap::Function,iterable,args...;kwargs...)
	pmapreduce_commutative(fmap,sum,iterable,args...;kwargs...)
end

function pmapsum_elementwise(fmap::Function,iterable,args...;kwargs...)
	pmapsum(plist->sum(asyncmap(x->fmap(x...,args...;kwargs...),plist)),iterable)
end

# Store the processor id with the value
struct pval{T}
	p :: Int
	parent :: T
end

function pmapreduce(fmap::Function,freduce::Function,iterable::Tuple,args...;kwargs...)

	procs_used = workersactive(iterable)

	num_workers_active = length(procs_used);
	hostnames = gethostnames(procs_used);
	nodes = nodenames(hostnames);
	procid_rank1_on_node = [procs_used[findfirst(isequal(node),hostnames)] for node in nodes];
	Nnodes_reduction = length(procid_rank1_on_node)

	nprocs_node_dict = nprocs_node(procs_used)
	node_channels = Dict(
		node=>(
			out = RemoteChannel(()->Channel{pval}(nprocs_node_dict[node]),procid_node),
			err = RemoteChannel(()->Channel{Bool}(nprocs_node_dict[node]),procid_node),
			)
			for (node,procid_node) in zip(nodes,procid_rank1_on_node))

	# Worker at which the final reduction takes place
	p_final = first(procid_rank1_on_node)

	finalnode_reducechannel = RemoteChannel(()->Channel{pval}(Nnodes_reduction),p_final)
	finalnode_errorchannel = RemoteChannel(()->Channel{Bool}(Nnodes_reduction),p_final)

	result_channel = RemoteChannel(()->Channel{Any}(1))
	error_channel = RemoteChannel(()->Channel{Bool}(1))

	# Run the function on each processor and compute the reduction at each node
	@sync for (rank,(p,node)) in enumerate(zip(procs_used,hostnames))
		@async begin
			
			eachnode_reducechannel = node_channels[node].out
			eachnode_errorchannel = node_channels[node].err

			p_parent = eachnode_reducechannel.where

			np_node = nprocs_node_dict[node]
			
			iterable_on_proc = evenlyscatterproduct(iterable,num_workers_active,rank)
			@spawnat p begin
				try
					res = pval(p,fmap(iterable_on_proc,args...;kwargs...))
					put!(eachnode_reducechannel,res)
					put!(eachnode_errorchannel,false)
				catch e
					put!(eachnode_errorchannel,true)
					rethrow()
				finally
					# Don't need references to the intermediate reduction channels on other nodes
					for (node_i,channels_i) in node_channels
						if node_i == node
							continue
						end

						finalize(channels_i.out)
						finalize(channels_i.err)
					end
					if p != p_parent
						finalize(eachnode_errorchannel)
						finalize(eachnode_reducechannel)
						finalize(finalnode_errorchannel)
						finalize(finalnode_reducechannel)
					end
					if p != p_final
						if p != result_channel.where
							finalize(result_channel)
						end
						if p != error_channel.where
							finalize(error_channel)
						end
					end
				end
			end

			@async if p == p_parent
				@spawnat p begin
					try
						anyerror = any(take!(eachnode_errorchannel) for i=1:np_node)
						if !anyerror
							vals = [take!(eachnode_reducechannel) for i=1:np_node]
							sort!(vals,by=x->x.p)
							res = pval(p,freduce(v.parent for v in vals))	
							put!(finalnode_reducechannel,res)
							put!(finalnode_errorchannel,false)
						else
							put!(finalnode_errorchannel,true)
						end
					catch e
						put!(finalnode_errorchannel,true)
						rethrow()
					finally
						finalize(eachnode_errorchannel)
						finalize(eachnode_reducechannel)

						if p != p_final
							finalize(finalnode_errorchannel)
							finalize(finalnode_reducechannel)
						end
					end
				end
			end

			@async if p == p_final
				@spawnat p begin
					try
						anyerror = any(take!(finalnode_errorchannel) for i=1:Nnodes_reduction)
						if !anyerror
							vals = [take!(finalnode_reducechannel) for i=1:Nnodes_reduction]
							sort!(vals,by=x->x.p)
							res = freduce(v.parent for v in vals)
							put!(result_channel,res)
							put!(error_channel,false)
						else
							put!(error_channel,true)
						end
					catch e
						put!(error_channel,true)
						rethrow()
					finally
						finalize(finalnode_errorchannel)
						finalize(finalnode_reducechannel)

						if p != result_channel.where
							finalize(result_channel)
						end
						if p != error_channel.where
							finalize(error_channel)
						end
					end
				end
			end
		end
	end

	anyerror = take!(error_channel)
	if !anyerror
		return take!(result_channel)
	end
end

function pmapreduce(fmap::Function,freduce::Function,
	itp::Iterators.ProductIterator,args...;kwargs...)
	pmapreduce(fmap,freduce,itp.iterators,args...;kwargs...)
end

function pmapreduce(fmap::Function,freduce::Function,iterable,args...;kwargs...)
	pmapreduce(fmap,freduce,(iterable,),args...;kwargs...)
end

############################################################################################
# pmap in batches without reduction
############################################################################################

function pmapbatch(f::Function,iterable::Tuple,args...;
	num_workers = nworkersactive(iterable),kwargs...)

	procs_used = workersactive(iterable)

	if num_workers < length(procs_used)
		procs_used = procs_used[1:num_workers]
	end
	num_workers = length(procs_used)

	futures = Vector{Future}(undef,num_workers)
	@sync for (rank,p) in enumerate(procs_used)
		@async begin
			iterable_on_proc = evenlyscatterproduct(iterable,num_workers,rank)
			futures[rank] = @spawnat p f(iterable_on_proc,args...;kwargs...)
		end
	end
	vcat(asyncmap(fetch,futures)...)
end

function pmapbatch(f::Function,itp::Iterators.ProductIterator,args...;kwargs...)
	pmapbatch(f,itp.iterators,args...;kwargs...)
end

function pmapbatch(f::Function,iterable,args...;kwargs...)
	pmapbatch(f,(iterable,),args...;kwargs...)
end

function pmapbatch_elementwise(f::Function,iterable,args...;kwargs...)
	pmapbatch(plist->asyncmap(x->f(x...,args...;kwargs...),plist),iterable)
end

end # module
