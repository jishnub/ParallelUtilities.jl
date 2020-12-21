"""
    AbstractConstrainedProduct{T,N}

Supertype of [`ProductSplit`](@ref) and [`ProductSection`](@ref).
"""
abstract type AbstractConstrainedProduct{T,N} end
Base.eltype(::AbstractConstrainedProduct{T}) where {T} = T
Base.ndims(::AbstractConstrainedProduct{<:Any,N}) where {N} = N

"""
    ProductSplit{T,N,Q}

Iterator that loops over the outer product of ranges in 
reverse-lexicographic order. The ranges need to be strictly
increasing. Given `N` ranges, 
each element returned by the iterator will be 
a tuple of length `N` with one element from each range.

See also: [`ProductSection`](@ref)
"""
struct ProductSplit{T,N,Q} <: AbstractConstrainedProduct{T,N}
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
        all(x->step(x)>0,iterators) || 
        throw(ArgumentError("all the iterators need to be strictly increasing"))

        new{T,N,Q}(iterators,togglelevels,np,p,firstind,lastind)
    end
end

workerrank(ps::ProductSplit) = ps.p

"""
    ProductSection{T,N,Q}

Iterator that loops over a specified section of the 
outer product of the ranges provided in 
reverse-lexicographic order. The ranges need to be strictly
increasing. Given `N` ranges, 
each element returned by the iterator will be 
a tuple of length `N` with one element from each range.

See also: [`ProductSplit`](@ref)
"""
struct ProductSection{T,N,Q} <: AbstractConstrainedProduct{T,N}
    iterators :: Q
    togglelevels :: NTuple{N,Int}
    firstind :: Int
    lastind :: Int

    function ProductSection(iterators::Tuple{Vararg{AbstractRange,N}},togglelevels::NTuple{N,Int},
        firstind::Int,lastind::Int) where {N}

        T = Tuple{eltype.(iterators)...}
        Q = typeof(iterators)

        # Ensure that all the iterators are strictly increasing
        all(x->step(x)>0,iterators) || 
        throw(ArgumentError("all the iterators need to be strictly increasing"))

        new{T,N,Q}(iterators,togglelevels,firstind,lastind)
    end
end

function mwerepr(ps::ProductSplit)
    "ProductSplit("*repr(ps.iterators)*", "*repr(ps.np)*", "*repr(ps.p)*")"
end
function Base.summary(io::IO, ps::ProductSplit)
    print(io, length(ps),"-element ", mwerepr(ps))
end
function Base.show(io::IO, ps::ProductSplit)
    print(io, mwerepr(ps))
end

function _cumprod(len::Tuple)
    (0,_cumprod(first(len),Base.tail(len))...)
end

_cumprod(::Integer,::Tuple{}) = ()
function _cumprod(n::Integer, tl::Tuple)
    (n,_cumprod(n*first(tl),Base.tail(tl))...)
end

@deprecate ntasks(x::Tuple) prod(length, x)

"""
    ntasks(iterators::Tuple)

The total number of elements in the outer product of the ranges contained in 
`iterators`, equal to `prod(length, iterators)`
"""
ntasks
ntasks(ps::AbstractConstrainedProduct) = ntasks(ps.iterators)

"""
    ProductSplit(iterators::Tuple{Vararg{AbstractRange}}, np::Integer, p::Integer)

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
function ProductSplit(iterators::Tuple{Vararg{AbstractRange}}, np::Integer, p::Integer)
    len = size.(iterators,1)
    Nel = prod(len)
    togglelevels = _cumprod(len)
    d,r = divrem(Nel,np)
    firstind = d*(p-1) + min(r,p-1) + 1
    lastind = d*p + min(r,p)
    ProductSplit(iterators,togglelevels,np,p,firstind,lastind)
end
ProductSplit(::Tuple{},::Integer,::Integer) = throw(ArgumentError("Need at least one iterator"))

"""
    ProductSection(iterators::Tuple{Vararg{AbstractRange}}, inds::AbstractUnitRange)

Construct a `ProductSection` iterator that represents a 1D view of the outer product
of the ranges provided in `iterators`, with the range of indices in the view being
specified by `inds`.

# Examples
```jldoctest
julia> p = ParallelUtilities.ProductSection((1:3,4:6), 5:8);

julia> collect(p)
4-element Array{Tuple{Int64,Int64},1}:
 (2, 5)
 (3, 5)
 (1, 6)
 (2, 6)

julia> collect(p) == collect(Iterators.product(1:3, 4:6))[5:8]
true
```
"""
function ProductSection(iterators::Tuple{Vararg{AbstractRange}},
    inds::AbstractUnitRange)

    isempty(inds) && throw(ArgumentError("range of indices must not be empty"))
    firstind,lastind = extrema(inds)

    len = size.(iterators,1)
    Nel = prod(len)
    1 <= firstind || throw(
        ArgumentError("the range of indices must start from a number ≥ 1"))
    lastind <= Nel || throw(
        ArgumentError("the maximum index must be less than or equal to the total number of elements = $Nel"))
    togglelevels = _cumprod(len)
    ProductSection(iterators,togglelevels,firstind,lastind)
end
function ProductSection(::Tuple{},::AbstractUnitRange)
    throw(ArgumentError("Need at least one iterator"))
end

Base.isempty(ps::AbstractConstrainedProduct) = (ps.firstind > ps.lastind)

function Base.first(ps::AbstractConstrainedProduct)
    isempty(ps) ? nothing : @inbounds _first(ps.iterators, childindex(ps, ps.firstind)...)
end

Base.@propagate_inbounds function _first(t::Tuple, ind::Integer, rest::Integer...)
    @boundscheck (1 <= ind <= length(first(t))) || throw(BoundsError(first(t),ind))
    (@inbounds first(t)[ind], _first(Base.tail(t), rest...)...)
end
_first(::Tuple{}) = ()

function Base.last(ps::AbstractConstrainedProduct)
    isempty(ps) ? nothing : @inbounds _last(ps.iterators, childindex(ps, ps.lastind)...)
end

Base.@propagate_inbounds function _last(t::Tuple, ind::Integer, rest::Integer...)
    @boundscheck (1 <= ind <= length(first(t))) || throw(BoundsError(first(t),ind))
    (@inbounds first(t)[ind], _last(Base.tail(t), rest...)...)
end
_last(::Tuple{}) = ()

Base.length(ps::AbstractConstrainedProduct) = ps.lastind - ps.firstind + 1

Base.firstindex(ps::AbstractConstrainedProduct) = 1
Base.lastindex(ps::AbstractConstrainedProduct) = ps.lastind - ps.firstind + 1

"""
    childindex(ps::AbstractConstrainedProduct, ind)

Return a tuple containing the indices of the individual iterators 
corresponding to the element that is present at index `ind` in the 
outer product of the iterators.

# Examples
```jldoctest
julia> ps = ProductSplit((1:5, 2:4, 1:3), 7, 1);

julia> ParallelUtilities.childindex(ps, 6)
(1, 2, 1)

julia> v = collect(Iterators.product(1:5, 2:4, 1:3));

julia> getindex.(ps.iterators, ParallelUtilities.childindex(ps,6)) == v[6]
true
```

See also: [`childindexshifted`](@ref)
"""
function childindex(ps::AbstractConstrainedProduct, ind)
    tl = reverse(Base.tail(ps.togglelevels))
    reverse(childindex(tl,ind))
end

function childindex(tl::Tuple, ind)
    t = first(tl)
    k = div(ind - 1, t)
    (k+1,childindex(Base.tail(tl), ind - k*t)...)
end

# First iterator gets the final remainder
childindex(::Tuple{}, ind) = (ind,)

"""
    childindexshifted(ps::AbstractConstrainedProduct, ind)

Return a tuple containing the indices in the individual iterators 
given an index of a `AbstractConstrainedProduct`.

# Examples
```jldoctest
julia> ps = ProductSplit((1:5, 2:4, 1:3), 7, 3);

julia> cinds = ParallelUtilities.childindexshifted(ps, 3)
(2, 1, 2)

julia> getindex.(ps.iterators, cinds) == ps[3]
true
```

See also: [`childindex`](@ref)
"""
function childindexshifted(ps::AbstractConstrainedProduct, ind)
    childindex(ps, (ind - 1) + ps.firstind)
end

Base.@propagate_inbounds function Base.getindex(ps::AbstractConstrainedProduct, ind)
    @boundscheck 1 <= ind <= length(ps) || throw(BoundsError(ps,ind))
    _getindex(ps, childindexshifted(ps, ind)...)
end
# This needs to be a separate function to deal with the case of a single child iterator, in which case 
# it's not clear if the single index is for the ProductSplit or the child iterator

# This method asserts that the number of indices is correct
Base.@propagate_inbounds function _getindex(ps::AbstractConstrainedProduct{<:Any,N},
    inds::Vararg{Integer,N}) where {N}
    
    _getindex(ps.iterators,inds...)
end

Base.@propagate_inbounds function _getindex(t::Tuple,ind::Integer,rest::Integer...)
    @boundscheck (1 <= ind <= length(first(t))) || throw(BoundsError(first(t),ind))
    (@inbounds first(t)[ind], _getindex(Base.tail(t),rest...)...)
end
_getindex(::Tuple{}, ::Integer...) = ()

function Base.iterate(ps::AbstractConstrainedProduct{T}, state=(first(ps), 1)) where {T}
    el,n = state

    if n > length(ps)
        return nothing
    elseif n == length(ps)
        # In this case the next value doesn't matter, so just return something arbitary
        next_state = (el::T, n+1)
    else
        next_state = (ps[n+1]::T, n+1)
    end

    (el::T, next_state)
end

function _firstlastalongdim(ps::AbstractConstrainedProduct, dim,
    firstindchild::Tuple = childindex(ps, ps.firstind),
    lastindchild::Tuple = childindex(ps, ps.lastind))

    iter = ps.iterators[dim]

    fic = firstindchild[dim]
    lic = lastindchild[dim]

    first_iter = iter[fic]
    last_iter = iter[lic]

    (first_iter,last_iter)
end

function _checkrollover(ps::AbstractConstrainedProduct, dim,
    firstindchild::Tuple=childindex(ps,ps.firstind),
    lastindchild::Tuple=childindex(ps,ps.lastind))

    _checkrollover(ps.iterators,dim,firstindchild,lastindchild)
end

function _checkrollover(t::Tuple, dim, firstindchild::Tuple, lastindchild::Tuple)

    if dim > 0
        return _checkrollover(Base.tail(t), dim-1, Base.tail(firstindchild), Base.tail(lastindchild))
    end

    !_checknorollover(reverse(t), reverse(firstindchild), reverse(lastindchild))
end

function _checknorollover(t, firstindchild, lastindchild)
    iter = first(t)
    first_iter = iter[first(firstindchild)]
    last_iter = iter[first(lastindchild)]

    (last_iter == first_iter) & 
        _checknorollover(Base.tail(t),Base.tail(firstindchild),Base.tail(lastindchild))
end
_checknorollover(::Tuple{}, ::Tuple{}, ::Tuple{}) = true

function _nrollovers(ps::AbstractConstrainedProduct, dim::Integer)
    dim == ndims(ps) && return 0
    nelements(ps, dim + 1) - 1
end

"""
    nelements(ps::AbstractConstrainedProduct; dim::Integer)

Compute the number of unique values in the section of the `dim`-th range contained in `ps`.

# Examples
```jldoctest
julia> ps = ProductSplit((1:5, 2:4, 1:3), 7, 3);

julia> collect(ps)
7-element Array{Tuple{Int64,Int64,Int64},1}:
 (5, 4, 1)
 (1, 2, 2)
 (2, 2, 2)
 (3, 2, 2)
 (4, 2, 2)
 (5, 2, 2)
 (1, 3, 2)

julia> ParallelUtilities.nelements(ps, dim = 1)
5

julia> ParallelUtilities.nelements(ps, dim = 2)
3

julia> ParallelUtilities.nelements(ps, dim = 3)
2
```
"""
function nelements(ps::AbstractConstrainedProduct, dim::Integer)
    Base.depwarn("nelements(ps, dim) is deprecated, please use nelements(ps, dim = dim)", :nelements)
    nelements(ps, dim = dim)
end
function nelements(ps::AbstractConstrainedProduct; dim::Integer)
    1 <= dim <= ndims(ps) || throw(ArgumentError("1 ⩽ dim ⩽ N=$(ndims(ps)) not satisfied for dim=$dim"))

    iter = ps.iterators[dim]

    if _nrollovers(ps,dim) == 0
        st = first(ps)[dim]
        en = last(ps)[dim]
        stind = searchsortedfirst(iter,st)
        enind = searchsortedfirst(iter,en)
        nel = length(stind:enind)
    elseif _nrollovers(ps,dim) > 1
        nel = length(iter)
    else
        st = first(ps)[dim]
        en = last(ps)[dim]
        stind = searchsortedfirst(iter,st)
        enind = searchsortedfirst(iter,en)
        if stind > enind
            # some elements are missed out
            nel = length(stind:length(iter)) + length(1:enind)
        else
            nel = length(iter)
        end
    end
    return nel
end


"""
    maximum(ps::AbstractConstrainedProduct; dim::Integer)

Compute the maximum value of the section of the range number `dim` contained in `ps`.

# Examples
```jldoctest
julia> ps = ProductSplit((1:2,4:5),2,1);

julia> collect(ps)
2-element Array{Tuple{Int64,Int64},1}:
 (1, 4)
 (2, 4)

julia> maximum(ps, dim = 1)
2

julia> maximum(ps, dim = 2)
4
```
"""
function Base.maximum(ps::AbstractConstrainedProduct; dim::Integer)

    isempty(ps) && return nothing
    
    firstindchild = childindex(ps, ps.firstind)
    lastindchild = childindex(ps, ps.lastind)

    first_iter,last_iter = _firstlastalongdim(ps, dim, firstindchild, lastindchild)

    v = last_iter

    # The last index will not roll over so this can be handled easily
    if dim == ndims(ps)
        return v
    end

    if _checkrollover(ps, dim, firstindchild, lastindchild)
        iter = ps.iterators[dim]
        v = maximum(iter)
    end

    return v
end

function Base.maximum(ps::AbstractConstrainedProduct, dim::Integer)
    Base.depwarn("maximum(ps::AbstractConstrainedProduct, dim) is deprecated, use maximum(ps, dim = dim) instead", :maximum)
    maximum(ps, dim = dim)
end

function Base.maximum(ps::AbstractConstrainedProduct{<:Any,1})
    isempty(ps) && return nothing
    lastindchild = childindex(ps, ps.lastind)
    lic_dim = lastindchild[1]
    iter = ps.iterators[1]
    iter[lic_dim]
end

"""
    minimum(ps::AbstractConstrainedProduct; dim::Integer)

Compute the minimum value of the section of the range number `dim` contained in `ps`.

# Examples
```jldoctest
julia> ps = ProductSplit((1:2, 4:5), 2, 1);

julia> collect(ps)
2-element Array{Tuple{Int64,Int64},1}:
 (1, 4)
 (2, 4)

julia> minimum(ps, dim = 1)
1

julia> minimum(ps, dim = 2)
4
```
"""
function Base.minimum(ps::AbstractConstrainedProduct; dim::Integer)
    
    isempty(ps) && return nothing

    firstindchild = childindex(ps, ps.firstind)
    lastindchild = childindex(ps, ps.lastind)

    first_iter,last_iter = _firstlastalongdim(ps, dim, firstindchild, lastindchild)

    v = first_iter

    # The last index will not roll over so this can be handled easily
    if dim == ndims(ps)
        return v
    end

    if _checkrollover(ps, dim, firstindchild, lastindchild)
        iter = ps.iterators[dim]
        v = minimum(iter)
    end

    return v
end

function Base.minimum(ps::AbstractConstrainedProduct, dim::Integer)
    Base.depwarn("minimum(ps::AbstractConstrainedProduct, dim) is deprecated, use minimum(ps, dim = dim) instead", :minimum)
    minimum(ps, dim = dim)
end

function Base.minimum(ps::AbstractConstrainedProduct{<:Any,1})
    isempty(ps) && return nothing
    firstindchild = childindex(ps,ps.firstind)
    fic_dim = firstindchild[1]
    iter = ps.iterators[1]
    iter[fic_dim]
end

"""
    extrema(ps::AbstractConstrainedProduct; dim::Integer)

Compute the `extrema` of the section of the range number `dim` contained in `ps`.

# Examples
```jldoctest
julia> ps = ProductSplit((1:2, 4:5), 2, 1);

julia> collect(ps)
2-element Array{Tuple{Int64,Int64},1}:
 (1, 4)
 (2, 4)

julia> extrema(ps, dim = 1)
(1, 2)

julia> extrema(ps, dim = 2)
(4, 4)
```
"""
function Base.extrema(ps::AbstractConstrainedProduct; dim::Integer)
    
    isempty(ps) && return nothing

    firstindchild = childindex(ps, ps.firstind)
    lastindchild = childindex(ps, ps.lastind)

    first_iter,last_iter = _firstlastalongdim(ps, dim, firstindchild, lastindchild)

    v = (first_iter,last_iter)
    # The last index will not roll over so this can be handled easily
    if dim == ndims(ps)
        return v
    end

    if _checkrollover(ps, dim, firstindchild, lastindchild)
        iter = ps.iterators[dim]
        v = extrema(iter)
    end

    return v
end

function Base.extrema(ps::AbstractConstrainedProduct{<:Any,1})
    isempty(ps) && return nothing
    firstindchild = childindex(ps, ps.firstind)
    lastindchild = childindex(ps, ps.lastind)
    fic_dim = firstindchild[1]
    lic_dim = lastindchild[1]
    iter = ps.iterators[1]
    
    (iter[fic_dim], iter[lic_dim])
end

function Base.extrema(ps::AbstractConstrainedProduct, dim::Integer)
    Base.depwarn("extrema(ps::AbstractConstrainedProduct, dim) is deprecated, use extrema(ps, dim = dim) instead", :extrema)
    extrema(ps, dim = dim)
end

"""
    extremadims(ps::AbstractConstrainedProduct)

Compute the extrema of the sections of all the ranges contained in `ps`. 
Functionally this is equivalent to 

```julia
map(i -> extrema(ps, dim = i), 1:ndims(ps))
```

but it is implemented more efficiently. 

Returns a `Tuple` containing the `(min, max)` pairs along each 
dimension, such that the `i`-th index of the result contains the `extrema` along the section of the `i`-th range
contained locally.

# Examples
```jldoctest
julia> ps = ProductSplit((1:2, 4:5), 2, 1);

julia> collect(ps)
2-element Array{Tuple{Int64,Int64},1}:
 (1, 4)
 (2, 4)

julia> extremadims(ps)
((1, 2), (4, 4))
```
"""
function extremadims(ps::AbstractConstrainedProduct)
    Base.depwarn("extremadims will not be exported in a future release, please call it as ParallelUtilities.extremadims instead", :extremadims)
    _extremadims(ps, 1, ps.iterators)
end

function _extremadims(ps::AbstractConstrainedProduct, dim::Integer, iterators::Tuple)
    (extrema(ps; dim = dim), _extremadims(ps, dim+1, Base.tail(iterators))...)
end
_extremadims(::AbstractConstrainedProduct, ::Integer, ::Tuple{}) = ()

"""
    extrema_commonlastdim(ps::AbstractConstrainedProduct)

Return the reverse-lexicographic extrema of values taken from 
ranges contained in `ps`, where the pairs of ranges are constructed 
by concatenating the ranges along each dimension with the last one.

For two ranges this simply returns `([first(ps)], [last(ps)])`.

# Examples
```jldoctest
julia> ps = ProductSplit((1:3,4:7,2:7), 10, 2);

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
function extrema_commonlastdim(ps::AbstractConstrainedProduct{<:Any,N}) where {N}

    Base.depwarn("extrema_commonlastdim will not be exported in a future release, please call it as ParallelUtilities.extrema_commonlastdim instead", :extrema_commonlastdim)

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

_infullrange(val::T, ps::AbstractConstrainedProduct{T}) where {T} = _infullrange(val,ps.iterators)

function _infullrange(val, t::Tuple)
    first(val) in first(t) && _infullrange(Base.tail(val),Base.tail(t))
end
_infullrange(::Tuple{}, ::Tuple{}) = true

function c2l_rec(iprev, nprev, ax, inds)
    i = searchsortedfirst(ax[1],inds[1])
    inew = iprev + (i-1)*nprev
    n = nprev*length(ax[1])
    c2l_rec(inew, n, Base.tail(ax), Base.tail(inds))
end

c2l_rec(i, n, ::Tuple{}, ::Tuple{}) = i

_cartesiantolinear(ax, inds) = c2l_rec(1,1,ax,inds)

"""
    indexinproduct(iterators::NTuple{N,AbstractRange}, val::NTuple{N,Any}) where {N}

Return the index of `val` in the outer product of `iterators`, 
where `iterators` is a `Tuple` of increasing `AbstractRange`s. 
Return nothing if `val` is not present.

# Examples
```jldoctest
julia> iterators = (1:4, 1:3, 3:5);

julia> val = (2, 2, 4);

julia> ind = ParallelUtilities.indexinproduct(iterators,val)
18

julia> collect(Iterators.product(iterators...))[ind] == val
true
```
"""
function indexinproduct(iterators::Tuple{Vararg{AbstractRange,N}},
    val::Tuple{Vararg{Any,N}}) where {N}

    all(in.(val,iterators)) || return nothing

    ax = axes.(iterators,1)
    individual_inds = searchsortedfirst.(iterators,val)

    _cartesiantolinear(ax, individual_inds)
end

indexinproduct(::Tuple{}, ::Tuple) = throw(ArgumentError("need at least one iterator"))

function Base.in(val::T, ps::AbstractConstrainedProduct{T}) where {T}
    _infullrange(val,ps) || return false
    
    ind = indexinproduct(ps.iterators, val)
    ps.firstind <= ind <= ps.lastind
end

# This struct is just a wrapper to flip the tuples before comparing
struct ReverseLexicographicTuple{T}
    t :: T
end

Base.isless(a::ReverseLexicographicTuple{T}, b::ReverseLexicographicTuple{T}) where {T} = reverse(a.t) < reverse(b.t)
Base.isequal(a::ReverseLexicographicTuple{T}, b::ReverseLexicographicTuple{T}) where {T} = a.t == b.t

"""
    whichproc(iterators::Tuple, val::Tuple, np::Integer)

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
        mid = div(left+right, 2)
        ps = ProductSplit(iterators, np, mid)

        # If np is greater than the number of ntasks then it's possible
        # that ps is empty. In this case the value must be somewhere in
        # the previous workers. Otherwise each worker has some tasks and 
        # these are sorted, so carry out a binary search

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
julia> iters = (1:10, 4:6, 1:4);

julia> ps = ProductSplit(iters, 5, 2);

julia> procrange_recast(iters, ps, 10)
3:4
```
"""
function procrange_recast(iterators::Tuple, ps::AbstractConstrainedProduct, np_new::Integer)
    
    Base.depwarn("procrange_recast will not be exported in a future release, please call it as ParallelUtilities.procrange_recast instead", :procrange_recast)

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

"""
    procrange_recast(ps::AbstractConstrainedProduct, np_new::Integer)

Return the range of processor ranks that would contain the values in `ps` if the 
iterators used to construct `ps` were split across `np_new` processes.

# Examples
```jldoctest
julia> iters = (1:10, 4:6, 1:4);

julia> ps = ProductSplit(iters, 5, 2); # split across 5 processes initially

julia> procrange_recast(ps, 10) # If `iters` were spread across 10 processes
3:4
```
"""
function procrange_recast(ps::AbstractConstrainedProduct, np_new::Integer)
    procrange_recast(ps.iterators, ps, np_new)
end

"""
    localindex(ps::AbstractConstrainedProduct{T}, val::T) where {T}

Return the index of `val` in `ps`. Return `nothing` if the value
is not found.

# Examples
```jldoctest
julia> ps = ProductSplit((1:3, 4:5:20), 3, 2);

julia> collect(ps)
4-element Array{Tuple{Int64,Int64},1}:
 (2, 9)
 (3, 9)
 (1, 14)
 (2, 14)

julia> localindex(ps, (3,9))
2
```
"""
function localindex(ps::AbstractConstrainedProduct{T}, val::T) where {T}

    (isempty(ps) || val ∉ ps) && return nothing

    indflat = indexinproduct(ps.iterators, val)
    indflat - ps.firstind + 1
end

# this is only needed because first and last return nothing if the ProductSplit is empty
localindex(::AbstractConstrainedProduct, ::Nothing) = nothing

function localindex(iterators::Tuple, val::Tuple, np::Integer, p::Integer)
    ps = ProductSplit(iterators, np, p)
    localindex(ps, val)
end

"""
    whichproc_localindex(iterators::Tuple, val::Tuple, np::Integer)

Return `(rank,ind)`, where `rank` is the
rank of the worker that `val` will reside on if the outer product 
of the ranges in `iterators` is spread over `np` workers, and `ind` is
the index of `val` in the local section on that worker.

# Examples
```jldoctest
julia> iters = (1:4, 2:8);

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
    procid = whichproc(iterators, val, np)
    index = localindex(iterators, val, np, procid)
    return procid, index
end

#################################################################

"""
    dropleading(ps::AbstractConstrainedProduct)

Return a `ProductSection` leaving out the first iterator contained in `ps`. 
The range of values of the remaining iterators in the 
resulting `ProductSection` will be the same as in `ps`.

# Examples
```jldoctest
julia> ps = ProductSplit((1:5, 2:4, 1:3), 7, 3);

julia> collect(ps)
7-element Array{Tuple{Int64,Int64,Int64},1}:
 (5, 4, 1)
 (1, 2, 2)
 (2, 2, 2)
 (3, 2, 2)
 (4, 2, 2)
 (5, 2, 2)
 (1, 3, 2)

julia> ParallelUtilities.dropleading(ps) |> collect
3-element Array{Tuple{Int64,Int64},1}:
 (4, 1)
 (2, 2)
 (3, 2)
```
"""
function dropleading(ps::AbstractConstrainedProduct)
    isempty(ps) && throw(ArgumentError("need at least one iterator"))
    iterators = Base.tail(ps.iterators)
    first_element = Base.tail(first(ps))
    last_element = Base.tail(last(ps))
    firstind = indexinproduct(iterators, first_element)
    lastind = indexinproduct(iterators, last_element)
    ProductSection(iterators,firstind:lastind)
end