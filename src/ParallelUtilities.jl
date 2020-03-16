module ParallelUtilities
using ProgressMeter

using Reexport
@reexport using Distributed

export  ProductSplit,
	evenlyscatterproduct,
	ntasks,
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

function Base.iterate(ps::ProductSplit{T},state=(first(ps),1)) where {T}
	el,n = state

	if n > length(ps)
		return nothing
	elseif n == length(ps)
		# In this case the next value doesn't matter, so just return something arbitary
		next_state = (el::T,n+1)
	else
		next_state = (ps[n+1]::T,n+1)
	end

	(el::T,next_state)
end

@inline Base.@propagate_inbounds function _firstlastalongdim(ps::ProductSplit{<:Any,N},dim::Int,
	firstindchild::Tuple=childindex(ps,ps.firstind),
	lastindchild::Tuple=childindex(ps,ps.lastind)) where {N}

	_firstlastalongdim(ps.iterators,dim,firstindchild,lastindchild)
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

const RemoteChannelContainer{T} = NamedTuple{(:out, :err),Tuple{RemoteChannel{Channel{T}},RemoteChannel{Channel{Bool}}}}

@inline Base.eltype(::RemoteChannelContainer{T}) where {T} = T

function RemoteChannelContainer{T}(n::Int,p::Int) where {T}
	out = RemoteChannel(()->Channel{T}(n),p)
    err = RemoteChannel(()->Channel{Bool}(n),p)
    RemoteChannelContainer{T}((out,err))
end
RemoteChannelContainer{T}(n::Int) where {T} = RemoteChannelContainer{T}(n,myid())
RemoteChannelContainer(n::Int,p::Int) = RemoteChannelContainer{Any}(n,p)
RemoteChannelContainer(n::Int) = RemoteChannelContainer{Any}(n,myid())

struct BinaryTreeError <: Exception 
	n :: Int
end
function Base.showerror(io::IO,err::BinaryTreeError)
	print(io,"attempt to construct a binary tree with $(err.n) children")
end

# Each node has N children, where N can be 0,1 or 2
struct BinaryTreeNode
	p :: Int
	parent :: Int
	nchildren :: Int

	function BinaryTreeNode(p::Int,p_parent::Int,N::Int)
		(0 <= N <= 2) || throw(BinaryTreeError(N))
		new(p,p_parent,N)
	end
end

@inline parentnoderank(i::Int) = max(div(i,2),1)

@inline nchildren(b::BinaryTreeNode) = b.nchildren

struct BinaryTree{T}
	N :: Int # total number of nodes
	h :: Int # Number of levels
	twochildendind :: Int
	onechildendind :: Int
	procs :: T

	function BinaryTree(procs::T) where {T<:AbstractVector{Int}}

		N = length(procs)
		(N >=1) || throw(DomainError(N,"need at least 1 node to create a binary tree"))

		h = floor(Int,log2(N)) # Number of levels of the tree
		Ninternalnodes = 2^h - 1
		Nleaf = N - Ninternalnodes
		Nonechildinternalnodes = (Ninternalnodes > 0) ? rem(Nleaf,2) : 0
		twochildendind = div(N-1,2)
		onechildstartind = twochildendind + 1
		onechildendind = onechildstartind + Nonechildinternalnodes - 1

		new{T}(N,h,twochildendind,onechildendind,procs)
	end
end

Base.length(b::BinaryTree) = b.N

function Base.getindex(tree::BinaryTree,i::Int)
	procs = tree.procs
	
	p = procs[i]
	p_parent = procs[parentnoderank(i)]
	
	if i <= tree.twochildendind
		# These nodes have two children each
		nchildren = 2
	elseif i <= tree.onechildendind
		# These nodes have one child each
		nchildren = 1
	else
		# These nodes have no children
		nchildren = 0
	end

	BinaryTreeNode(p,p_parent,nchildren)
end

struct BranchChannel{Tmap,Tred}
	p :: Int
	selfchannels :: RemoteChannelContainer{Tmap}
	parentchannels :: RemoteChannelContainer{Tred}
	childrenchannels :: RemoteChannelContainer{Tred}
	nchildren :: Int

	function BranchChannel(p::Int,selfchannels::RemoteChannelContainer{Tmap},
		parentchannels::RemoteChannelContainer{Tred},
		childrenchannels::RemoteChannelContainer{Tred},nchildren::Int) where {Tmap,Tred}

		(0 <= nchildren <= 2) || throw(BinaryTreeError(nchildren))
	
		new{Tmap,Tred}(p,selfchannels,parentchannels,childrenchannels,nchildren)
	end
end
@inline nchildren(b::BranchChannel) = b.nchildren

function BranchChannel(p::Int,::Type{Tmap},
	parentchannels::RemoteChannelContainer{Tred},nchildren::Int) where {Tmap,Tred}

	(0 <= nchildren <= 2) || throw(BinaryTreeError(nchildren))
	selfchannels = RemoteChannelContainer{Tmap}(1,p)
	childrenchannels = RemoteChannelContainer{Tred}(nchildren,p)
	BranchChannel(p,selfchannels,parentchannels,childrenchannels,nchildren)
end

function BranchChannel{Tmap,Tred}(p::Int,nchildren::Int) where {Tmap,Tred}
	(0 <= nchildren <= 2) || throw(BinaryTreeError(nchildren))
	parentchannels = RemoteChannelContainer{Tred}(1,p)
	BranchChannel(p,Tmap,parentchannels,nchildren)
end
BranchChannel{Tmap,Tred}(nchildren::Int) where {Tmap,Tred} = BranchChannel{Tmap,Tred}(myid(),nchildren)

function finalize_except_wherewhence(r::RemoteChannel)
	if (myid() != r.where) && (myid() != r.whence)
		finalize(r)
	end
end
finalize_except_wherewhence(r::RemoteChannelContainer) = map(finalize_except_wherewhence,(r.out,r.err))

function Base.finalize(r::RemoteChannelContainer)
	finalize(r.out)
	finalize(r.err)
end

function Base.finalize(bc::BranchChannel)
	finalize(bc.selfchannels)
	finalize(bc.childrenchannels)
	finalize_except_wherewhence(bc.parentchannels)
end

function createbranchchannels(::Type{Tmap},::Type{Tred},tree::BinaryTree) where {Tmap,Tred}
	branches = Vector{BranchChannel{Tmap,Tred}}(undef,length(tree))

	# the first node has to be created separately as its children will be linked to it
	firstnode = tree[1]
	N = nchildren(firstnode)
	p = firstnode.p
	branches[1] = BranchChannel{Tmap,Tred}(p,N)

	for i=2:length(tree)
		node = tree[i]
		p = node.p
		parentnodebranches = branches[parentnoderank(i)]
		parentchannels = parentnodebranches.childrenchannels
		branches[i] = BranchChannel(p,Tmap,parentchannels,nchildren(node))
	end

	return branches
end

function createbranchchannels(::Type{Tmap},::Type{Tred},iterators::Tuple) where {Tmap,Tred}
	tree = BinaryTree(workersactive(iterators))
	createbranchchannels(Tmap,Tred,tree)
end
@inline createbranchchannels(iterators::Tuple) = createbranchchannels(Any,Any,iterators)

abstract type Ordering end
struct Sorted <: Ordering end
struct Unsorted <: Ordering end

# Store the processor id with the value, necessary for sorting
struct pval{T}
	p :: Int
	parent :: T
end

# Function to obtain the value of pval types
@inline value(p::pval) = p.parent
@inline value(p::Any) = p

@inline pval(val::T) where {T} = pval{T}(myid(),val)
@inline pval{T}(val::T) where {T} = pval{T}(myid(),val)
@inline pval{T}(val) where {T} = pval{T}(myid(),convert(T,value(val)))

@inline Base.:(==)(p1::pval,p2::pval) = (p1.p == p2.p) && (value(p1) == value(p2))

@inline Base.convert(::Type{pval{T}},x) where {T} = pval(T(value(x)))
@inline Base.convert(::Type{pval{Any}},x) = pval{Any}(value(x))
@inline Base.convert(::Type{pval{Any}},x::pval{Any}) = x
@inline Base.convert(::Type{pval{T}},x::pval{T}) where {T} = x

############################################################################################
# Map
############################################################################################

# Wrap a pval around the mapped value if sorting is necessary
@inline function maybepvalput!(pipe::BranchChannel{T},val) where {T}
	put!(pipe.selfchannels.out,val)
end
@inline function maybepvalput!(pipe::BranchChannel{T},val) where {T<:pval}
	valT = T(value(val))
	put!(pipe.selfchannels.out,valT)
end

function mapTreeNode(fmap::Function,iterator,pipe::BranchChannel,
	progress::RemoteChannel,args...;kwargs...)
	# Evaluate the function
	# Store the error flag locally
	# If there are no errors then store the result locally
	# No communication with other nodes happens here
	try
		res = fmap(iterator,args...;kwargs...)
		maybepvalput!(pipe,res)
		put!(pipe.selfchannels.err,false)
	catch
		put!(pipe.selfchannels.err,true)
		rethrow()
	finally
		put!(progress,(true,false))
	end
end

############################################################################################
# Reduction
############################################################################################

function reducedvalue(freduce::Function,pipe::BranchChannel{Tmap,Tred},::Unsorted) where {Tmap,Tred}
	self = take!(pipe.selfchannels.out) :: Tmap

	N = nchildren(pipe)
	res = if N > 0
			reducechildren = freduce(take!(pipe.childrenchannels.out)::Tred for i=1:N)::Tred
			freduce((reducechildren,self)) :: Tred
		else
			freduce((self,)) :: Tred
		end
end

function reducedvalue(freduce::Function,pipe::BranchChannel{Tmap,Tred},::Sorted) where {Tmap,Tred}
	N = nchildren(pipe)

	vals = Vector{Tred}(undef,N+1)
	@sync begin
		@async begin
			selfval = take!(pipe.selfchannels.out)::Tmap
			selfvalred = freduce((value(selfval),))
			vals[1] = pval(selfvalred)
		end
		@async for i=2:N+1
			vals[i] = take!(pipe.childrenchannels.out) :: Tred
		end
	end

	sort!(vals,by=x->x.p)
	Tred(freduce(value(v) for v in vals))
end

function reduceTreeNode(freduce::Function,pipe::BranchChannel{Tmap,Tred},
	ifsort::Ordering,progress::RemoteChannel) where {Tmap,Tred}
	# This function that communicates with the parent and children

	# Start by checking if there is any error locally in the map,
	# and if there's none then check if there are any errors on the children
	anyerr = take!(pipe.selfchannels.err) || 
				any(take!(pipe.childrenchannels.err) for i=1:nchildren(pipe))

	# Evaluate the reduction only if there's no error
	# In either case push the error flag to the parent
	if !anyerr
		try
			res = reducedvalue(freduce,pipe,ifsort) :: Tred
			put!(pipe.parentchannels.out,res)
			put!(pipe.parentchannels.err,false)
		catch e
			put!(pipe.parentchannels.err,true)
			rethrow()
		finally
			put!(progress,(false,true))
		end
	else
		put!(pipe.parentchannels.err,true)
		put!(progress,(false,true))
	end

	finalize(pipe)
end

function return_unless_error(r::RemoteChannelContainer)
	anyerror = take!(r.err)
	if !anyerror
		return value(take!(r.out))
	end
end

@inline return_unless_error(b::BranchChannel) = return_unless_error(b.parentchannels)

function pmapreduceworkers(fmap::Function,freduce::Function,iterators::Tuple,
	branches,ord::Ordering,args...;kwargs...)

	num_workers_active = nworkersactive(iterators)

	nmap,nred = 0,0
	progresschannel = RemoteChannel(()->Channel{Tuple{Bool,Bool}}(2num_workers_active))
	progressbar = Progress(2num_workers_active,1,"Progress in pmapreduce : ")

	# Run the function on each processor and compute the reduction at each node
	@sync begin
		for (rank,mypipe) in enumerate(branches)
			@async begin
				p = mypipe.p
				iterable_on_proc = evenlyscatterproduct(iterators,num_workers_active,rank)

				@spawnat p mapTreeNode(fmap,iterable_on_proc,mypipe,
										progresschannel,args...;kwargs...)
				@spawnat p reduceTreeNode(freduce,mypipe,ord,progresschannel)
			end
		end
		
		for i = 1:2num_workers_active
			mapdone,reddone = take!(progresschannel)
			if mapdone
				nmap += 1
			end
			if reddone
				nred += 1
			end
			next!(progressbar;showvalues=[(:map,nmap),(:reduce,nred)])
		end
		finish!(progressbar)
	end

	return_unless_error(first(branches))
end

function infer_returntypes(fmap,freduce,x::T,args...;kwargs...) where {T<:ProductSplit}
	fmap_padded(x) = fmap(x,args...;kwargs...)
	Tmap = first(Base.return_types(fmap_padded,(T,)))
	Tred = first(Base.return_types(freduce,(Tuple{Tmap},)))
	Tmap,Tred
end

function infer_returntypes(fmap,freduce,iterators::Tuple,args...;kwargs...)
	iteratorsPS = evenlyscatterproduct(iterators,1,1)
	infer_returntypes(fmap,freduce,iteratorsPS,args...;kwargs...)
end

# This function does not sort the values, so it might be faster
function pmapreduce_commutative(fmap::Function,::Type{Tmap},
	freduce::Function,::Type{Tred},iterators::Tuple,args...;kwargs...) where {Tmap,Tred}
	
	branches = createbranchchannels(Tmap,Tred,iterators)
	pmapreduceworkers(fmap,freduce,iterators,branches,Unsorted(),args...;kwargs...)
end

function pmapreduce_commutative(fmap::Function,freduce::Function,iterators::Tuple,args...;
	infer_types = true, kwargs...)

	if infer_types
		Tmap,Tred = infer_returntypes(fmap,freduce,iterators,args...;kwargs...)
	else
		Tmap,Tred = Any,Any
	end

	pmapreduce_commutative(fmap,Tmap,freduce,Tred,iterators,args...;kwargs...)
end

function pmapreduce_commutative(fmap::Function,::Type{Tmap},freduce::Function,
	::Type{Tred},itp::Iterators.ProductIterator,args...;kwargs...) where {Tmap,Tred}

	pmapreduce_commutative(fmap,Tmap,freduce,Tred,itp.iterators,args...;kwargs...)
end

function pmapreduce_commutative(fmap::Function,freduce::Function,
	itp::Iterators.ProductIterator,args...;kwargs...)

	pmapreduce_commutative(fmap,freduce,itp.iterators,args...;kwargs...)
end

function pmapreduce_commutative(fmap::Function,::Type{Tmap},freduce::Function,::Type{Tred},
	iterable,args...;kwargs...) where {Tmap,Tred}
	pmapreduce_commutative(fmap,Tmap,freduce,Tred,(iterable,),args...;kwargs...)
end

function pmapreduce_commutative(fmap::Function,freduce::Function,iterable,args...;kwargs...)
	pmapreduce_commutative(fmap,freduce,(iterable,),args...;kwargs...)
end

function pmapreduce_commutative_elementwise(fmap::Function,::Type{Tmap},
	freduce::Function,::Type{Tred},iterable,args...;kwargs...) where {Tmap,Tred}
	
	pmapreduce_commutative(plist->freduce(asyncmap(x->fmap(x...,args...;kwargs...),plist)),
		Tred,freduce,Tred,iterable,args...;kwargs...)
end

function pmapreduce_commutative_elementwise(fmap::Function,freduce::Function,iterable,args...;kwargs...)
	pmapreduce_commutative(plist->freduce(asyncmap(x->fmap(x...,args...;kwargs...),plist)),
		freduce,iterable,args...;kwargs...)
end

function pmapsum(fmap::Function,::Type{T},iterable,args...;kwargs...) where {T}
	pmapreduce_commutative(fmap,T,sum,T,iterable,args...;kwargs...)
end

function pmapsum(fmap::Function,iterable,args...;kwargs...)
	pmapreduce_commutative(fmap,sum,iterable,args...;kwargs...)
end

function pmapsum_elementwise(fmap::Function,::Type{T},iterable,args...;kwargs...) where {T}
	pmapsum(plist->sum(asyncmap(x->fmap(x...,args...;kwargs...),plist)),T,iterable)
end

function pmapsum_elementwise(fmap::Function,iterable,args...;kwargs...)
	pmapsum(plist->sum(asyncmap(x->fmap(x...,args...;kwargs...),plist)),iterable)
end

function pmapreduce(fmap::Function,::Type{Tmap},freduce::Function,::Type{Tred},
	iterators::Tuple,args...;kwargs...) where {Tmap,Tred}

	branches = createbranchchannels(pval{Tmap},pval{Tred},iterators)
	pmapreduceworkers(fmap,freduce,iterators,branches,Sorted(),args...;kwargs...)
end

function pmapreduce(fmap::Function,freduce::Function,iterators::Tuple,args...;
	infer_types = true, kwargs...)

	if infer_types
		Tmap,Tred = infer_returntypes(fmap,freduce,iterators,args...;kwargs...)
	else
		Tmap,Tred = Any,Any
	end
	pmapreduce(fmap,Tmap,freduce,Tred,iterators,args...;kwargs...)
end

function pmapreduce(fmap::Function,::Type{Tmap},freduce::Function,::Type{Tred},
	itp::Iterators.ProductIterator,args...;kwargs...) where {Tmap,Tred}

	pmapreduce(fmap,Tmap,freduce,Tred,itp.iterators,args...;kwargs...)
end

function pmapreduce(fmap::Function,freduce::Function,
	itp::Iterators.ProductIterator,args...;kwargs...)

	pmapreduce(fmap,freduce,itp.iterators,args...;kwargs...)
end

function pmapreduce(fmap::Function,::Type{Tmap},freduce::Function,::Type{Tred},
	iterable,args...;kwargs...) where {Tmap,Tred}
	
	pmapreduce(fmap,Tmap,freduce,Tred,(iterable,),args...;kwargs...)
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
