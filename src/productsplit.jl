struct TaskNotPresentError{T,U} <: Exception
    t :: T
    task :: U
end
function Base.showerror(io::IO, err::TaskNotPresentError)
    print(io, "could not find the task $(err.task) in the list $(err.t)")
end

"""
    AbstractConstrainedProduct{T, N, Q}

Supertype of [`ProductSplit`](@ref) and [`ProductSection`](@ref).
"""
abstract type AbstractConstrainedProduct{T, N, Q} end
Base.eltype(::AbstractConstrainedProduct{T}) where {T} = T

_niterators(::AbstractConstrainedProduct{<:Any, N}) where {N} = N

const IncreasingAbstractConstrainedProduct{T, N} =
    AbstractConstrainedProduct{T, N, <:NTuple{N, AbstractUnitRange}}

"""
    ProductSection{T, N, Q<:NTuple{N,AbstractRange}}

Iterator that loops over a specified section of the
outer product of ranges in. If the ranges are strictly increasing, the
iteration will be in reverse - lexicographic order.
Given `N` ranges, each element returned by the iterator will be
a tuple of length `N` with one element from each range.

See also: [`ProductSplit`](@ref)
"""
struct ProductSection{T, N, Q <: NTuple{N,AbstractRange}} <: AbstractConstrainedProduct{T, N, Q}
    iterators :: Q
    togglelevels :: NTuple{N, Int}
    firstind :: Int
    lastind :: Int

    function ProductSection(iterators::Tuple{Vararg{AbstractRange, N}}, togglelevels::NTuple{N, Int},
        firstind::Int, lastind::Int) where {N}

        # Ensure that all the iterators are strictly increasing
        all(x->step(x)>0, iterators) ||
        throw(ArgumentError("all the ranges need to be strictly increasing"))

        T = Tuple{map(eltype, iterators)...}

        new{T, N, typeof(iterators)}(iterators, togglelevels, firstind, lastind)
    end
end

function _cumprod(len::Tuple)
    (0, _cumprod(first(len), Base.tail(len))...)
end

_cumprod(::Integer,::Tuple{}) = ()
function _cumprod(n::Integer, tl::Tuple)
    (n, _cumprod(n*first(tl), Base.tail(tl))...)
end

function takedrop(ps::ProductSection)
    drop = ps.firstind - 1
    take = ps.lastind - ps.firstind + 1
    Iterators.take(Iterators.drop(Iterators.product(ps.iterators...), drop), take)
end

"""
    ProductSection(iterators::Tuple{Vararg{AbstractRange}}, inds::AbstractUnitRange)

Construct a `ProductSection` iterator that represents a 1D view of the outer product
of the ranges provided in `iterators`, with the range of indices in the view being
specified by `inds`.

# Examples
```jldoctest
julia> p = ParallelUtilities.ProductSection((1:3, 4:6), 5:8);

julia> collect(p)
4-element $(Vector{Tuple{Int, Int}}):
 (2, 5)
 (3, 5)
 (1, 6)
 (2, 6)

julia> collect(p) == collect(Iterators.product(1:3, 4:6))[5:8]
true
```
"""
function ProductSection(iterators::Tuple{Vararg{AbstractRange}}, inds::AbstractUnitRange)
    firstind, lastind = first(inds), last(inds)

    len = map(length, iterators)
    Nel = prod(len)
    1 <= firstind || throw(
        ArgumentError("the range of indices must start from a number ≥ 1"))
    lastind <= Nel || throw(
        ArgumentError("the maximum index must be less than or equal to the total number of elements = $Nel"))
    togglelevels = _cumprod(len)
    ProductSection(iterators, togglelevels, firstind, lastind)
end
ProductSection(::Tuple{}, ::AbstractUnitRange) = throw(ArgumentError("need at least one iterator"))

"""
    ProductSplit{T, N, Q<:NTuple{N,AbstractRange}}

Iterator that loops over a section of the outer product of ranges.
If the ranges are strictly increasing, the iteration is in reverse - lexicographic order.
Given `N` ranges, each element returned by the iterator will be
a tuple of length `N` with one element from each range.

See also: [`ProductSection`](@ref)
"""
struct ProductSplit{T, N, Q<:NTuple{N, AbstractRange}} <: AbstractConstrainedProduct{T, N, Q}
    ps :: ProductSection{T, N, Q}
    np :: Int
    p :: Int

    function ProductSplit(ps::ProductSection{T, N, Q}, np::Integer, p::Integer) where {T, N, Q}
        1 <= p <= np || throw(ArgumentError("processor rank out of range"))
        new{T, N, Q}(ps, np, p)
    end
end

function nelementsdroptake(len, np, p)
    d, r = divrem(len, np)
    drop = d*(p - 1) + min(r, p - 1)
    lastind = d*p + min(r, p)
    take = lastind - drop
    drop, take
end

"""
    ProductSplit(iterators::Tuple{Vararg{AbstractRange}}, np::Integer, p::Integer)

Construct a `ProductSplit` iterator that represents the outer product
of the iterators split over `np` workers, with this instance reprsenting
the values on the `p`-th worker.

!!! note
    `p` here refers to the rank of the worker, and is unrelated to the worker ID obtained by
    executing `myid()` on that worker.

# Examples
```jldoctest
julia> ParallelUtilities.ProductSplit((1:2, 4:5), 2, 1) |> collect
2-element $(Vector{Tuple{Int, Int}}):
 (1, 4)
 (2, 4)

julia> ParallelUtilities.ProductSplit((1:2, 4:5), 2, 2) |> collect
2-element $(Vector{Tuple{Int, Int}}):
 (1, 5)
 (2, 5)
```
"""
function ProductSplit(iterators::Tuple{Vararg{AbstractRange}}, np::Integer, p::Integer)
    # d, r = divrem(prod(length, iterators), np)
    # firstind = d*(p - 1) + min(r, p - 1) + 1
    # lastind = d*p + min(r, p)
    drop, take = nelementsdroptake(prod(length, iterators), np, p)
    firstind = drop + 1
    lastind = drop + take
    ProductSplit(ProductSection(iterators, firstind:lastind), np, p)
end
ProductSplit(::Tuple{}, ::Integer, ::Integer) = throw(ArgumentError("Need at least one iterator"))

takedrop(ps::ProductSplit) = takedrop(ProductSection(ps))

workerrank(ps::ProductSplit) = ps.p
Distributed.nworkers(ps::ProductSplit) = ps.np

ProductSection(ps::ProductSection) = ps
ProductSection(ps::ProductSplit) = ps.ps

getiterators(ps::AbstractConstrainedProduct) = ProductSection(ps).iterators
togglelevels(ps::AbstractConstrainedProduct) = ProductSection(ps).togglelevels

function Base.summary(io::IO, ps::AbstractConstrainedProduct)
    print(io, length(ps), "-element ", string(nameof(typeof(ps))))
end
function Base.show(io::IO, ps::AbstractConstrainedProduct)
    summary(io, ps)
    if !isempty(ps)
        print(io, " [", repr(first(ps)) * ", ... , " * repr(last(ps)), "]")
    end
end

Base.isempty(ps::AbstractConstrainedProduct) = (firstindexglobal(ps) > lastindexglobal(ps))

function Base.first(ps::AbstractConstrainedProduct)
    isempty(ps) && throw(ArgumentError("collection must be non - empty"))
    _first(getiterators(ps), childindex(ps, firstindexglobal(ps))...)
end

function _first(t::Tuple, ind::Integer, rest::Integer...)
    (1 <= ind <= length(first(t))) || throw(BoundsError(first(t), ind))
    (first(t)[ind], _first(Base.tail(t), rest...)...)
end
_first(::Tuple{}) = ()

function Base.last(ps::AbstractConstrainedProduct)
    isempty(ps) && throw(ArgumentError("collection must be non - empty"))
    _last(getiterators(ps), childindex(ps, lastindexglobal(ps))...)
end

function _last(t::Tuple, ind::Integer, rest::Integer...)
    (1 <= ind <= length(first(t))) || throw(BoundsError(first(t), ind))
    (first(t)[ind], _last(Base.tail(t), rest...)...)
end
_last(::Tuple{}) = ()

Base.length(ps::AbstractConstrainedProduct) = lastindex(ps)

Base.firstindex(ps::AbstractConstrainedProduct) = 1
Base.lastindex(ps::AbstractConstrainedProduct) = lastindexglobal(ps) - firstindexglobal(ps) + 1

firstindexglobal(ps::AbstractConstrainedProduct) = ProductSection(ps).firstind
lastindexglobal(ps::AbstractConstrainedProduct) = ProductSection(ps).lastind

# SplittablesBase interface
function SplittablesBase.halve(ps::AbstractConstrainedProduct)
    iter = getiterators(ps)
    firstind = firstindexglobal(ps)
    lastind = lastindexglobal(ps)
    nleft = length(ps) ÷ 2
    firstindleft = firstind
    lastindleft = firstind + nleft - 1
    firstindright = lastindleft + 1
    lastindright = lastind
    tl = togglelevels(ps)
    ProductSection(iter, tl, firstindleft, lastindleft),
    ProductSection(iter, tl, firstindright, lastindright)
end

"""
    childindex(ps::AbstractConstrainedProduct, ind)

Return a tuple containing the indices of the individual `AbstractRange`s
corresponding to the element that is present at index `ind` in the
outer product of the ranges.

!!! note
    The index `ind` corresponds to the outer product of the ranges, and not to `ps`.

# Examples
```jldoctest
julia> iters = (1:5, 2:4, 1:3);

julia> ps = ParallelUtilities.ProductSplit(iters, 7, 1);

julia> ind = 6;

julia> cinds = ParallelUtilities.childindex(ps, ind)
(1, 2, 1)

julia> v = collect(Iterators.product(iters...));

julia> getindex.(iters, cinds) == v[ind]
true
```

See also: [`childindexshifted`](@ref)
"""
function childindex(ps::AbstractConstrainedProduct, ind)
    tl = reverse(Base.tail(togglelevels(ps)))
    reverse(childindex(tl, ind))
end

function childindex(tl::Tuple, ind)
    t = first(tl)
    k = div(ind - 1, t)
    (k + 1, childindex(Base.tail(tl), ind - k*t)...)
end

# First iterator gets the final remainder
childindex(::Tuple{}, ind) = (ind,)

"""
    childindexshifted(ps::AbstractConstrainedProduct, ind)

Return a tuple containing the indices in the individual iterators
given an index of `ps`.

If the iterators `(r1, r2, ...)` are used to generate
`ps`, then return `(i1, i2, ...)` such that `ps[ind] == (r1[i1], r2[i2], ...)`.

# Examples
```jldoctest
julia> iters = (1:5, 2:4, 1:3);

julia> ps = ParallelUtilities.ProductSplit(iters, 7, 3);

julia> psind = 4;

julia> cinds = ParallelUtilities.childindexshifted(ps, psind)
(3, 1, 2)

julia> getindex.(iters, cinds) == ps[psind]
true
```

See also: [`childindex`](@ref)
"""
function childindexshifted(ps::AbstractConstrainedProduct, ind)
    childindex(ps, (ind - 1) + firstindexglobal(ps))
end

function Base.getindex(ps::AbstractConstrainedProduct, ind)
    1 <= ind <= length(ps) || throw(BoundsError(ps, ind))
    _getindex(ps, childindexshifted(ps, ind)...)
end
# This needs to be a separate function to deal with the case of a single child iterator, in which case
# it's not clear if the single index is for the ProductSplit or the child iterator

# This method asserts that the number of indices is correct
function _getindex(ps::AbstractConstrainedProduct{<:Any, N}, inds::Vararg{Integer, N}) where {N}
    _getindex(getiterators(ps), inds...)
end

function _getindex(t::Tuple, ind::Integer, rest::Integer...)
    (1 <= ind <= length(first(t))) || throw(BoundsError(first(t), ind))
    (first(t)[ind], _getindex(Base.tail(t), rest...)...)
end
_getindex(::Tuple{}, ::Integer...) = ()

function Base.iterate(ps::AbstractConstrainedProduct, state...)
    iterate(takedrop(ps), state...)
end

function _firstlastalongdim(ps::AbstractConstrainedProduct, dims,
    firstindchild::Tuple = childindex(ps, firstindexglobal(ps)),
    lastindchild::Tuple = childindex(ps, lastindexglobal(ps)))

    iter = getiterators(ps)[dims]

    fic = firstindchild[dims]
    lic = lastindchild[dims]

    first_iter = iter[fic]
    last_iter = iter[lic]

    (first_iter, last_iter)
end

function _checkrollover(ps::AbstractConstrainedProduct, dims,
    firstindchild::Tuple = childindex(ps, firstindexglobal(ps)),
    lastindchild::Tuple = childindex(ps, lastindexglobal(ps)))

    _checkrollover(getiterators(ps), dims, firstindchild, lastindchild)
end

function _checkrollover(t::Tuple, dims, firstindchild::Tuple, lastindchild::Tuple)
    if dims > 0
        return _checkrollover(Base.tail(t), dims - 1, Base.tail(firstindchild), Base.tail(lastindchild))
    end

    !_checknorollover(reverse(t), reverse(firstindchild), reverse(lastindchild))
end

function _checknorollover(t, firstindchild, lastindchild)
    iter = first(t)
    first_iter = iter[first(firstindchild)]
    last_iter = iter[first(lastindchild)]

    (last_iter == first_iter) &
        _checknorollover(Base.tail(t), Base.tail(firstindchild), Base.tail(lastindchild))
end
_checknorollover(::Tuple{}, ::Tuple{}, ::Tuple{}) = true

function _nrollovers(ps::AbstractConstrainedProduct, dims::Integer)
    dims == _niterators(ps) && return 0
    nelements(ps; dims = dims + 1) - 1
end

"""
    nelements(ps::AbstractConstrainedProduct{T, N, <:NTuple{N,AbstractUnitRange}}; dims::Integer) where {T,N}

Compute the number of unique values in the section of the `dims`-th range contained in `ps`.

The function is defined currently only for iterator products of `AbstractUnitRange`s.

# Examples
```jldoctest
julia> ps = ParallelUtilities.ProductSplit((1:5, 2:4, 1:3), 7, 3);

julia> collect(ps)
7-element $(Vector{Tuple{Int, Int, Int}}):
 (5, 4, 1)
 (1, 2, 2)
 (2, 2, 2)
 (3, 2, 2)
 (4, 2, 2)
 (5, 2, 2)
 (1, 3, 2)

julia> ParallelUtilities.nelements(ps, dims = 1)
5

julia> ParallelUtilities.nelements(ps, dims = 2)
3

julia> ParallelUtilities.nelements(ps, dims = 3)
2
```
"""
function nelements(ps::IncreasingAbstractConstrainedProduct; dims::Integer)
    1 <= dims <= _niterators(ps) || throw(ArgumentError("1 ⩽ dims ⩽ N=$(_niterators(ps)) not satisfied for dims=$dims"))

    iter = getiterators(ps)[dims]

    if _nrollovers(ps, dims) == 0
        st = first(ps)[dims]
        en = last(ps)[dims]
        stind = findfirst(isequal(st), iter)
        enind = findfirst(isequal(en), iter)
        nel = length(stind:enind)
    elseif _nrollovers(ps, dims) > 1
        nel = length(iter)
    else
        st = first(ps)[dims]
        en = last(ps)[dims]
        stind = findfirst(isequal(st), iter)
        enind = findfirst(isequal(en), iter)
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
    maximumelement(ps::AbstractConstrainedProduct; dims::Integer)

Compute the maximum value of the section of the range number `dims` contained in `ps`.

# Examples
```jldoctest
julia> ps = ParallelUtilities.ProductSplit((1:2, 4:5), 2, 1);

julia> collect(ps)
2-element $(Vector{Tuple{Int, Int}}):
 (1, 4)
 (2, 4)

julia> ParallelUtilities.maximumelement(ps, dims = 1)
2

julia> ParallelUtilities.maximumelement(ps, dims = 2)
4
```
"""
function maximumelement(ps::IncreasingAbstractConstrainedProduct; dims::Integer)
    isempty(ps) && throw(ArgumentError("collection must be non - empty"))

    firstindchild = childindex(ps, firstindexglobal(ps))
    lastindchild = childindex(ps, lastindexglobal(ps))

    _, last_iter = _firstlastalongdim(ps, dims, firstindchild, lastindchild)

    v = last_iter

    # The last index will not roll over so this can be handled easily
    if dims == _niterators(ps)
        return v
    end

    if _checkrollover(ps, dims, firstindchild, lastindchild)
        iter = getiterators(ps)[dims]
        v = maximum(iter)
    end

    return v
end

function maximumelement(ps::IncreasingAbstractConstrainedProduct{<:Any, 1})
    isempty(ps) && throw(ArgumentError("range must be non - empty"))
    lastindchild = childindex(ps, lastindexglobal(ps))
    lic_dim = lastindchild[1]
    iter = getiterators(ps)[1]
    iter[lic_dim]
end

"""
    minimumelement(ps::AbstractConstrainedProduct; dims::Integer)

Compute the minimum value of the section of the range number `dims` contained in `ps`.

# Examples
```jldoctest
julia> ps = ParallelUtilities.ProductSplit((1:2, 4:5), 2, 1);

julia> collect(ps)
2-element $(Vector{Tuple{Int, Int}}):
 (1, 4)
 (2, 4)

julia> ParallelUtilities.minimumelement(ps, dims = 1)
1

julia> ParallelUtilities.minimumelement(ps, dims = 2)
4
```
"""
function minimumelement(ps::IncreasingAbstractConstrainedProduct; dims::Integer)
    isempty(ps) && throw(ArgumentError("collection must be non - empty"))

    firstindchild = childindex(ps, firstindexglobal(ps))
    lastindchild = childindex(ps, lastindexglobal(ps))

    first_iter, last_iter = _firstlastalongdim(ps, dims, firstindchild, lastindchild)

    v = first_iter

    # The last index will not roll over so this can be handled easily
    if dims == _niterators(ps)
        return v
    end

    if _checkrollover(ps, dims, firstindchild, lastindchild)
        iter = getiterators(ps)[dims]
        v = minimum(iter)
    end

    return v
end

function minimumelement(ps::IncreasingAbstractConstrainedProduct{<:Any, 1})
    isempty(ps) && throw(ArgumentError("range must be non - empty"))
    firstindchild = childindex(ps, firstindexglobal(ps))
    fic_dim = firstindchild[1]
    iter = getiterators(ps)[1]
    iter[fic_dim]
end

"""
    extremaelement(ps::AbstractConstrainedProduct; dims::Integer)

Compute the `extrema` of the section of the range number `dims` contained in `ps`.

# Examples
```jldoctest
julia> ps = ParallelUtilities.ProductSplit((1:2, 4:5), 2, 1);

julia> collect(ps)
2-element $(Vector{Tuple{Int, Int}}):
 (1, 4)
 (2, 4)

julia> ParallelUtilities.extremaelement(ps, dims = 1)
(1, 2)

julia> ParallelUtilities.extremaelement(ps, dims = 2)
(4, 4)
```
"""
function extremaelement(ps::IncreasingAbstractConstrainedProduct; dims::Integer)
    isempty(ps) && throw(ArgumentError("collection must be non - empty"))

    firstindchild = childindex(ps, firstindexglobal(ps))
    lastindchild = childindex(ps, lastindexglobal(ps))

    first_iter, last_iter = _firstlastalongdim(ps, dims, firstindchild, lastindchild)

    v = (first_iter, last_iter)
    # The last index will not roll over so this can be handled easily
    if dims == _niterators(ps)
        return v
    end

    if _checkrollover(ps, dims, firstindchild, lastindchild)
        iter = getiterators(ps)[dims]
        v = extrema(iter)
    end

    return v
end

function extremaelement(ps::IncreasingAbstractConstrainedProduct{<:Any, 1})
    isempty(ps) && throw(ArgumentError("collection must be non - empty"))
    firstindchild = childindex(ps, firstindexglobal(ps))
    lastindchild = childindex(ps, lastindexglobal(ps))
    fic_dim = firstindchild[1]
    lic_dim = lastindchild[1]
    iter = getiterators(ps)[1]

    (iter[fic_dim], iter[lic_dim])
end

for (f, g) in [(:maximumelement, :maximum), (:minimumelement, :minimum), (:extremaelement, :extrema)]
    @eval $f(ps::AbstractConstrainedProduct{<:Any, 1}) = $g(first, takedrop(ps))
    @eval $f(ps::AbstractConstrainedProduct; dims::Integer) = $g(x -> x[dims], takedrop(ps))
end

"""
    extremadims(ps::AbstractConstrainedProduct)

Compute the extrema of the sections of all the ranges contained in `ps`.
Functionally this is equivalent to

```julia
map(i -> extrema(ps, dims = i), 1:_niterators(ps))
```

but it is implemented more efficiently.

Returns a `Tuple` containing the `(min, max)` pairs along each
dimension, such that the `i`-th index of the result contains the `extrema` along the section of the `i`-th range
contained locally.

# Examples
```jldoctest
julia> ps = ParallelUtilities.ProductSplit((1:2, 4:5), 2, 1);

julia> collect(ps)
2-element $(Vector{Tuple{Int, Int}}):
 (1, 4)
 (2, 4)

julia> ParallelUtilities.extremadims(ps)
((1, 2), (4, 4))
```
"""
function extremadims(ps::AbstractConstrainedProduct)
    _extremadims(ps, 1, getiterators(ps))
end

function _extremadims(ps::AbstractConstrainedProduct, dims::Integer, iterators::Tuple)
    (extremaelement(ps; dims = dims), _extremadims(ps, dims + 1, Base.tail(iterators))...)
end
_extremadims(::AbstractConstrainedProduct, ::Integer, ::Tuple{}) = ()

"""
    extrema_commonlastdim(ps::AbstractConstrainedProduct{T, N, <:NTuple{N,AbstractUnitRange}}) where {T,N}

Return the reverse - lexicographic extrema of values taken from
ranges contained in `ps`, where the pairs of ranges are constructed
by concatenating the ranges along each dimension with the last one.

For two ranges this simply returns `([first(ps)], [last(ps)])`.

# Examples
```jldoctest
julia> ps = ParallelUtilities.ProductSplit((1:3, 4:7, 2:7), 10, 2);

julia> collect(ps)
8-element $(Vector{Tuple{Int, Int, Int}}):
 (3, 6, 2)
 (1, 7, 2)
 (2, 7, 2)
 (3, 7, 2)
 (1, 4, 3)
 (2, 4, 3)
 (3, 4, 3)
 (1, 5, 3)

julia> ParallelUtilities.extrema_commonlastdim(ps)
$((Tuple{Int,Int}[(1, 2), (6, 2)], Tuple{Int,Int}[(3, 3), (5, 3)]))
```
"""
function extrema_commonlastdim(ps::IncreasingAbstractConstrainedProduct)
    isempty(ps) && return nothing

    m = extremadims(ps)
    lastvar_min, lastvar_max = last(m)

    val_first = first(ps)
    val_last = last(ps)
    min_vals = collect(Base.front(val_first))
    max_vals = collect(Base.front(val_last))

    for val in ps
        val_rev = reverse(val)
        lastvar = first(val_rev)
        (lastvar_min < lastvar < lastvar_max) && continue

        for (ind, vi) in enumerate(Base.tail(val_rev))
            if lastvar == lastvar_min
                min_vals[_niterators(ps) - ind] = min(min_vals[_niterators(ps) - ind], vi)
            end
            if lastvar == lastvar_max
                max_vals[_niterators(ps) - ind] = max(max_vals[_niterators(ps) - ind], vi)
            end
        end
    end

    [(m, lastvar_min) for m in min_vals], [(m, lastvar_max) for m in max_vals]
end

_infullrange(val::T, ps::AbstractConstrainedProduct{T}) where {T} = _infullrange(val, getiterators(ps))

function _infullrange(val, t::Tuple)
    first(val) in first(t) && _infullrange(Base.tail(val), Base.tail(t))
end
_infullrange(::Tuple{}, ::Tuple{}) = true

"""
    indexinproduct(iterators::NTuple{N, AbstractRange}, val::NTuple{N, Any}) where {N}

Return the index of `val` in the outer product of `iterators`.
Return nothing if `val` is not present.

# Examples
```jldoctest
julia> iterators = (1:4, 1:3, 3:5);

julia> val = (2, 2, 4);

julia> ind = ParallelUtilities.indexinproduct(iterators, val)
18

julia> collect(Iterators.product(iterators...))[ind] == val
true
```
"""
function indexinproduct(iterators::NTuple{N, AbstractRange}, val::Tuple{Vararg{Any, N}}) where {N}
    all(map(in, val, iterators)) || return nothing

    ax = map(x -> 1:length(x), iterators)
    individual_inds = map((it, val) -> findfirst(isequal(val), it), iterators, val)

    LinearIndices(ax)[individual_inds...]
end

indexinproduct(::Tuple{}, ::Tuple{}) = throw(ArgumentError("need at least one iterator"))

function Base.in(val::T, ps::AbstractConstrainedProduct{T}) where {T}
    _infullrange(val, ps) || return false

    ind = indexinproduct(getiterators(ps), val)
    firstindexglobal(ps) <= ind <= lastindexglobal(ps)
end

function Base.in(val::T, ps::IncreasingAbstractConstrainedProduct{T}) where {T}
    _infullrange(val, ps) || return false
    ReverseLexicographicTuple(first(ps)) <= ReverseLexicographicTuple(val) <= ReverseLexicographicTuple(last(ps))
end

# This struct is just a wrapper to flip the tuples before comparing
struct ReverseLexicographicTuple{T<:Tuple}
    t :: T
end

Base.isless(a::ReverseLexicographicTuple{T}, b::ReverseLexicographicTuple{T}) where {T} = reverse(a.t) < reverse(b.t)
Base.isequal(a::ReverseLexicographicTuple, b::ReverseLexicographicTuple) = a.t == b.t

"""
    whichproc(iterators::Tuple{Vararg{AbstractRange}}, val::Tuple, np::Integer)

Return the processor rank that will contain `val` if the outer
product of the ranges contained in `iterators` is split evenly
across `np` processors.

# Examples
```jldoctest
julia> iters = (1:4, 2:3);

julia> np = 2;

julia> ParallelUtilities.ProductSplit(iters, np, 2) |> collect
4-element $(Vector{Tuple{Int, Int}}):
 (1, 3)
 (2, 3)
 (3, 3)
 (4, 3)

julia> ParallelUtilities.whichproc(iters, (2, 3), np)
2
```
"""
function whichproc(iterators::Tuple{AbstractRange, Vararg{AbstractRange}}, val, np::Integer)
    _infullrange(val, iterators) || return nothing
    np >= 1 || throw(ArgumentError("np must be >= 1"))
    np  == 1 && return 1

    # We may carry out a binary search as the iterators are sorted
    left, right = 1, np

    val_t = ReverseLexicographicTuple(val)

    while left < right
        mid = div(left + right, 2)
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

    return left
end

whichproc(ps::ProductSplit, val) = whichproc(getiterators(ps), val, ps.np)

# This function tells us the range of processors that would be involved
# if we are to compute the tasks contained in the list ps on np_new processors.
# The total list of tasks is contained in iterators, and might differ from
# getiterators(ps) (eg if ps contains a subsection of the parameter set)
"""
    procrange_recast(iterators::Tuple{Vararg{AbstractRange}}, ps, np_new::Integer)

Return the range of processor ranks that would contain the values in `ps` if
the outer produce of the ranges in `iterators` is split across `np_new`
workers.

The values contained in `ps` should be a subsection of the outer product of
the ranges in `iterators`.

# Examples
```jldoctest
julia> iters = (1:10, 4:6, 1:4);

julia> ps = ParallelUtilities.ProductSplit(iters, 5, 2);

julia> ParallelUtilities.procrange_recast(iters, ps, 10)
3:4
```
"""
function procrange_recast(iterators::Tuple{AbstractRange, Vararg{AbstractRange}}, ps::AbstractConstrainedProduct, np_new::Integer)
    isempty(ps) && return nothing

    procid_start = whichproc(iterators, first(ps), np_new)
    if procid_start === nothing
        throw(TaskNotPresentError(iterators, first(ps)))
    end
    if length(ps) == 1
        procid_end = procid_start
    else
        procid_end = whichproc(iterators, last(ps), np_new)
        if procid_end === nothing
            throw(TaskNotPresentError(iterators, last(ps)))
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

julia> ps = ParallelUtilities.ProductSplit(iters, 5, 2); # split across 5 processes initially

julia> ParallelUtilities.procrange_recast(ps, 10) # If `iters` were spread across 10 processes
3:4
```
"""
function procrange_recast(ps::AbstractConstrainedProduct, np_new::Integer)
    procrange_recast(getiterators(ps), ps, np_new)
end

"""
    localindex(ps::AbstractConstrainedProduct{T}, val::T) where {T}

Return the index of `val` in `ps`. Return `nothing` if the value
is not found.

# Examples
```jldoctest
julia> ps = ParallelUtilities.ProductSplit((1:3, 4:5:20), 3, 2);

julia> collect(ps)
4-element $(Vector{Tuple{Int, Int}}):
 (2, 9)
 (3, 9)
 (1, 14)
 (2, 14)

julia> ParallelUtilities.localindex(ps, (3, 9))
2
```
"""
function localindex(ps::AbstractConstrainedProduct{T}, val::T) where {T}
    (isempty(ps) || val ∉ ps) && return nothing

    indflat = indexinproduct(getiterators(ps), val)
    indflat - firstindexglobal(ps) + 1
end

"""
    whichproc_localindex(iterators::Tuple{Vararg{AbstractRange}}, val::Tuple, np::Integer)

Return `(rank, ind)`, where `rank` is the
rank of the worker that `val` will reside on if the outer product
of the ranges in `iterators` is spread over `np` workers, and `ind` is
the index of `val` in the local section on that worker.

# Examples
```jldoctest
julia> iters = (1:4, 2:8);

julia> np = 10;

julia> ParallelUtilities.whichproc_localindex(iters, (2, 4), np)
(4, 1)

julia> ParallelUtilities.ProductSplit(iters, np, 4) |> collect
3-element $(Vector{Tuple{Int, Int}}):
 (2, 4)
 (3, 4)
 (4, 4)
```
"""
function whichproc_localindex(iterators::Tuple{Vararg{AbstractRange}}, val::Tuple, np::Integer)
    procid = whichproc(iterators, val, np)
    procid === nothing && return nothing
    index = localindex(ProductSplit(iterators, np, procid), val)
    index === nothing && return nothing
    return procid, index
end

#################################################################

"""
    dropleading(ps::AbstractConstrainedProduct{T, N, NTuple{N,AbstractUnitRange}}) where {T,N}

Return a `ProductSection` leaving out the first iterator contained in `ps`.
The range of values of the remaining iterators in the
resulting `ProductSection` will be the same as in `ps`.

# Examples
```jldoctest
julia> ps = ParallelUtilities.ProductSplit((1:5, 2:4, 1:3), 7, 3);

julia> collect(ps)
7-element $(Vector{Tuple{Int, Int, Int}}):
 (5, 4, 1)
 (1, 2, 2)
 (2, 2, 2)
 (3, 2, 2)
 (4, 2, 2)
 (5, 2, 2)
 (1, 3, 2)

julia> ParallelUtilities.dropleading(ps) |> collect
3-element $(Vector{Tuple{Int, Int}}):
 (4, 1)
 (2, 2)
 (3, 2)
```
"""
function dropleading(ps::IncreasingAbstractConstrainedProduct)
    isempty(ps) && throw(ArgumentError("need at least one iterator"))
    iterators = Base.tail(getiterators(ps))
    first_element = Base.tail(first(ps))
    last_element = Base.tail(last(ps))
    firstind = indexinproduct(iterators, first_element)
    lastind = indexinproduct(iterators, last_element)
    ProductSection(iterators, firstind:lastind)
end
