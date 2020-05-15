"""
	ProductSplit{T,N,Q}

Iterator that loops over the outer product of ranges in 
reverse-lexicographic order. The ranges need to be strictly
increasing. Given `N` ranges, 
each element returned by the iterator will be 
a tuple of length `N` with one element from each range.
"""
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

function mwerepr(ps::ProductSplit)
	"ProductSplit("*repr(ps.iterators)*","*repr(ps.np)*","*repr(ps.p)*")"
end
function Base.summary(io::IO,ps::ProductSplit)
	print(io,length(ps),"-element ",mwerepr(ps))
end

function _cumprod(len::Tuple)
	(0,_cumprod(first(len),Base.tail(len))...)
end

@inline _cumprod(::Int,::Tuple{}) = ()
function _cumprod(n::Int,tl::Tuple)
	(n,_cumprod(n*first(tl),Base.tail(tl))...)
end

"""
	ntasks(iterators::Tuple)

The total number of elements in the outer product of the ranges contained in 
`iterators`, equal to `prod(length.(iterators))`
"""
@inline ntasks(iterators::Tuple) = mapreduce(length,*,iterators)
@inline ntasks(ps::ProductSplit) = ntasks(ps.iterators)

"""
	ProductSplit(iterators, np::Int, p::Int)

Construct a `ProductSplit` iterator that represents the outer product 
of the iterators split over `np` workers, with this instance reprsenting 
the values on the `p`-th worker.

# Examples
```jldoctest
julia> ProductSplit((1:2,4:5), 2, 1) |> collect
2-element Array{Tuple{Int64,Int64},1}:
 (1, 4)
 (2, 4)

julia> ProductSplit((1:2,4:5), 2, 2) |> collect
2-element Array{Tuple{Int64,Int64},1}:
 (1, 5)
 (2, 5)
```
"""
function ProductSplit(iterators::Tuple{Vararg{AbstractRange}},np::Int,p::Int)
	len = size.(iterators,1)
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
	@boundscheck 1 <= ind <= length(ps) || throw(BoundsError(ps,ind))
	_getindex(ps,childindexshifted(ps, ind)...)
end
# This needs to be a separate function to deal with the case of a single child iterator, in which case 
# it's not clear if the single index is for the ProductSplit or the child iterator

# This method asserts that the number of indices is correct
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

@inline function Base.maximum(ps::ProductSplit{<:Any,1})
	isempty(ps) && return nothing
	lastindchild = childindex(ps,ps.lastind)
	@inbounds lic_dim = lastindchild[1]
	@inbounds iter = ps.iterators[1]
	iter[lic_dim]
end

"""
	maximum(ps::ProductSplit; dim::Int)

Compute the maximum value of the range number `dim` that is
contained in `ps`.

# Examples
```jldoctest
julia> ps = ProductSplit((1:2,4:5),2,1);

julia> collect(ps)
2-element Array{Tuple{Int64,Int64},1}:
 (1, 4)
 (2, 4)

julia> maximum(ps,dim=1)
2

julia> maximum(ps,dim=2)
4
```
"""
@inline function Base.maximum(ps::ProductSplit{<:Any,N};dim::Int) where {N}

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

@inline function Base.minimum(ps::ProductSplit{<:Any,1})
	isempty(ps) && return nothing
	firstindchild = childindex(ps,ps.firstind)
	@inbounds fic_dim = firstindchild[1]
	@inbounds iter = ps.iterators[1]
	iter[fic_dim]
end

"""
	minimum(ps::ProductSplit; dim::Int)

Compute the minimum value of the range number `dim` that is
contained in `ps`.

# Examples
```jldoctest
julia> ps = ProductSplit((1:2,4:5),2,1);

julia> collect(ps)
2-element Array{Tuple{Int64,Int64},1}:
 (1, 4)
 (2, 4)

julia> minimum(ps,dim=1)
1

julia> minimum(ps,dim=2)
4
```
"""
@inline function Base.minimum(ps::ProductSplit{<:Any,N};dim::Int) where {N}
	
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

@inline function Base.extrema(ps::ProductSplit{<:Any,1})
	isempty(ps) && return nothing
	firstindchild = childindex(ps,ps.firstind)
	lastindchild = childindex(ps,ps.lastind)
	@inbounds fic_dim = firstindchild[1]
	@inbounds lic_dim = lastindchild[1]
	@inbounds iter = ps.iterators[1]
	
	(iter[fic_dim],iter[lic_dim])
end

"""
	extrema(ps::ProductSplit; dim::Int)

Compute the minimum and maximum of the range number `dim` that is
contained in `ps`.

# Examples
```jldoctest
julia> ps = ProductSplit((1:2,4:5),2,1);

julia> collect(ps)
2-element Array{Tuple{Int64,Int64},1}:
 (1, 4)
 (2, 4)

julia> extrema(ps,dim=1)
(1, 2)

julia> extrema(ps,dim=2)
(4, 4)
```
"""
@inline function Base.extrema(ps::ProductSplit{<:Any,N};dim::Int) where {N}
	
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

"""
	extremadims(ps::ProductSplit)

Compute the extrema of all the ranges contained in `ps`.

# Examples
```jldoctest
julia> ps = ProductSplit((1:2,4:5), 2, 1);

julia> collect(ps)
2-element Array{Tuple{Int64,Int64},1}:
 (1, 4)
 (2, 4)

julia> extremadims(ps)
((1, 2), (4, 4))
```
"""
extremadims(ps::ProductSplit) = _extremadims(ps,1,ps.iterators)

function _extremadims(ps::ProductSplit,dim::Int,iterators::Tuple)
	(extrema(ps;dim=dim),_extremadims(ps,dim+1,Base.tail(iterators))...)
end
_extremadims(::ProductSplit,::Int,::Tuple{}) = ()

"""
	extrema_commonlastdim(ps::ProductSplit)

Return the reverse-lexicographic extrema of values taken from 
ranges contained in `ps`, where the pairs of ranges are constructed 
by concatenating each dimension with the last one.

For two ranges this simply returns ([first(ps)],[last(ps)]).

# Examples
```jldoctest
julia> ps = ProductSplit((1:3,4:7,2:7),10,2);

julia> collect(ps)
8-element Array{Tuple{Int64,Int64,Int64},1}:
 (3, 6, 2)
 (1, 7, 2)
 (2, 7, 2)
 (3, 7, 2)
 (1, 4, 3)
 (2, 4, 3)
 (3, 4, 3)
 (1, 5, 3)

julia> extrema_commonlastdim(ps)
([(1, 2), (6, 2)], [(3, 3), (5, 3)])
```
"""
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

"""
	whichproc(iterators::Tuple, val::Tuple, np::Integer )

Return the processor rank that will contain `val` if the outer 
product of the ranges contained in `iterators` is split evenly 
across `np` processors.

# Examples
```jldoctest
julia> iters = (1:4, 2:3);

julia> np = 2;

julia> ProductSplit(iters, np, 2) |> collect
4-element Array{Tuple{Int64,Int64},1}:
 (1, 3)
 (2, 3)
 (3, 3)
 (4, 3)

julia> whichproc(iters, (2,3), np)
2
``` 
"""
function whichproc(iterators, val, np::Integer)
	
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
end

whichproc(iterators, ::Nothing, np::Integer) = nothing

# This function tells us the range of processors that would be involved
# if we are to compute the tasks contained in the list ps on np_new processors.
# The total list of tasks is contained in iterators, and might differ from 
# ps.iterators (eg if ps contains a subsection of the parameter set)
"""
	procrange_recast(iterators::Tuple, ps::ProductSplit, np_new::Integer)

Return the range of processor ranks that would contain the values in `ps` if 
the outer produce of the ranges in `iterators` is split across `np_new` 
workers.

The values contained in `ps` should be a subsection of the outer product of 
the ranges in `iterators`.

# Examples
```jldoctest
julia> iters = (1:10,4:6,1:4);

julia> ps = ProductSplit(iters, 5, 2);

julia> procrange_recast(iters, ps, 10)
3:4
```
"""
function procrange_recast(iterators::Tuple, ps::ProductSplit, np_new::Integer)
	
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

function procrange_recast(ps::ProductSplit, np_new::Integer)
	procrange_recast(ps.iterators,ps,np_new)
end

"""
	localindex(ps::ProductSplit{T}, val::T) where {T}

Return the index of `val` in `ps`. Return `nothing` if the value
is not found.

# Examples
```jldoctest
julia> ps = ProductSplit((1:3,4:5:20), 3, 2);

julia> collect(ps)
4-element Array{Tuple{Int64,Int64},1}:
 (2, 9)
 (3, 9)
 (1, 14)
 (2, 14)

julia> localindex(ps,(3,9))
2
```
"""
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
end

localindex(::ProductSplit, ::Nothing) = nothing

function localindex(iterators::Tuple, val::Tuple, np::Integer,procid::Integer)
	ps = ProductSplit(iterators,np,procid)
	localindex(ps,val)
end

"""
	whichproc_localindex(iterators::Tuple, val::Tuple, np::Integer)

Return `(wrank,lind)`, where `wrank` is the
rank of the worker that `val` will reside on if the outer product 
of the ranges in `iterators` is spread over `np` workers, and `lind` is
the index of `val` in the local section on that worker.

# Examples
```jldoctest
julia> iters = (1:4,2:8);

julia> np = 10;

julia> whichproc_localindex(iters, (2,4), np)
(4, 1)

julia> ProductSplit(iters, np, 4) |> collect
3-element Array{Tuple{Int64,Int64},1}:
 (2, 4)
 (3, 4)
 (4, 4)
```
"""
function whichproc_localindex(iterators::Tuple, val::Tuple, np::Integer)
	procid = whichproc(iterators,val,np)
	index = localindex(iterators,val,np,procid)
	return procid,index
end