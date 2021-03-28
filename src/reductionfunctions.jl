"""
    Commutative

Declare a reduction operator to be commutative in its arguments.
No check is performed to ascertain if the operator is indeed commutative.
"""
struct Commutative{F} <: Function
    f :: F
end

(c::Commutative)(x, y) = c.f(x, y)

"""
    BroadcastFunction(f)

Construct a binary function that evaluates `f.(x, y)` given the arguments `x` and `y`.

!!! note
    The function `BroadcastFunction(f)` is equivalent to `Base.BroadcastFunction(f)` on Julia versions
    1.6 and above.

# Examples
```jldoctest
julia> ParallelUtilities.BroadcastFunction(+)(ones(3), ones(3))
3-element Vector{Float64}:
 2.0
 2.0
 2.0
```
"""
struct BroadcastFunction{V, F} <: Function
    f :: F
end

BroadcastFunction{V}(f) where {V} = BroadcastFunction{V, typeof(f)}(f)
BroadcastFunction(f::Function) = BroadcastFunction{Nothing, typeof(f)}(f)

(o::BroadcastFunction{Nothing})(x, y) = o.f.(x, y)

(o::BroadcastFunction{1})(x, y) = broadcast!(o.f, x, x, y)
(o::BroadcastFunction{2})(x, y) = broadcast!(o.f, y, x, y)

"""
    broadcastinplace(f, ::Val{N}) where {N}

Construct a binary operator that evaluates `f.(x, y)` and overwrites the `N`th argument with the result.
For `N == 1` this evaluates `x .= f.(x, y)`, whereas for `N == 2` this evaluates `y .= f.(x, y)`.

# Examples

```jldoctest
julia> op = ParallelUtilities.broadcastinplace(+, Val(1));

julia> x = ones(3); y = ones(3);

julia> op(x, y)
3-element Vector{Float64}:
 2.0
 2.0
 2.0

julia> x # overwritten
3-element Vector{Float64}:
 2.0
 2.0
 2.0
```
"""
function broadcastinplace(f, v::Val{N}) where {N}
    BroadcastFunction{N}(f)
end

"""
    elementwisesum!(x, y)

Binary reduction operator that performs an elementwise product and stores the result inplace in `x`.
The value of `x` is overwritten in the process.

Functionally `elementwisesum!(x, y)` is equivalent to `x .= x .+ y`.

!!! note
    The operator is assumed to be commutative.
"""
const elementwisesum! = Commutative(broadcastinplace(+, Val(1)))

"""
    elementwiseproduct!(x, y)

Binary reduction operator that performs an elementwise product and stores the result inplace in `x`.
The value of `x` is overwritten in the process.

Functionally `elementwiseproduct!(x, y)` is equivalent to `x .= x .* y`.

!!! note
    The operator is assumed to be commutative.
"""
const elementwiseproduct! = Commutative(broadcastinplace(*, Val(1)))

"""
    elementwisemin!(x, y)

Binary reduction operator that performs an elementwise `min` and stores the result inplace in `x`.
The value of `x` is overwritten in the process.

Functionally `elementwisemin!(x, y)` is equivalent to `x .= min.(x, y)`.

!!! note
    The operator is assumed to be commutative.
"""
const elementwisemin! = Commutative(broadcastinplace(min, Val(1)))

"""
    elementwisemax!(x, y)

Binary reduction operator that performs an elementwise `max` and stores the result inplace in `x`.
The value of `x` is overwritten in the process.

Functionally `elementwisemax!(x, y)` is equivalent to `x .= max.(x, y)`.

!!! note
    The operator is assumed to be commutative.
"""
const elementwisemax! = Commutative(broadcastinplace(max, Val(1)))

"""
    BroadcastStack(f, dims)(x::AbstractArray, y::AbstractArray)

Construct a binary function that stacks its arguments along `dims`, with overlapping indices `I` being replaced by
`f(x[I], y[I])`. The arguments `x` and `y` must both be `n`-dimensional arrays that have identical axes along all dimensions
aside from those specified by `dims`. The axes of the result along each dimensions `d`
in `dims` would be `union(axes(x, d), axes(y, d))`.
Along the other dimensions the result has the same axes as `x` and `y`.

!!! note
    If the resulting axes along the concatenated dimensions are not 1-based, one might require an offset array package
    such as [`OffsetArrays.jl`](https://github.com/JuliaArrays/OffsetArrays.jl).

# Examples
```jldoctest
julia> A = ones(2)*2
2-element Vector{Float64}:
 2.0
 2.0

julia> B = ones(3)*3
3-element Vector{Float64}:
 3.0
 3.0
 3.0

julia> ParallelUtilities.BroadcastStack(min, 1)(A, B)
3-element Vector{Float64}:
 2.0
 2.0
 3.0

julia> A = ones(2,2)*2
2×2 Matrix{Float64}:
 2.0  2.0
 2.0  2.0

julia> B = ones(2,3)*3
2×3 Matrix{Float64}:
 3.0  3.0  3.0
 3.0  3.0  3.0

julia> ParallelUtilities.BroadcastStack(+, 2)(A, B)
2×3 Matrix{Float64}:
 5.0  5.0  3.0
 5.0  5.0  3.0
```
"""
struct BroadcastStack{F, D} <: Function
    f :: F
    dims :: D
end

(s::BroadcastStack)(x, y) = broadcaststack(x, y, s.f, s.dims)

function _union(axes_x_dim::AbstractUnitRange, axes_y_dim::AbstractUnitRange)
    axes_dim_min = min(minimum(axes_x_dim), minimum(axes_y_dim))
    axes_dim_max = max(maximum(axes_x_dim), maximum(axes_y_dim))
    axes_dim = axes_dim_min:axes_dim_max
end
_union(axes_x_dim::Base.OneTo, axes_y_dim::Base.OneTo) = axes_x_dim ∪ axes_y_dim

_maybeUnitRange(ax::AbstractUnitRange) = UnitRange(ax)
_maybeUnitRange(ax::Base.OneTo) = ax

function _subsetaxes(f, axes_x, axes_y, dims)
    ax = collect(_maybeUnitRange.(axes_x))
    for dim in dims
        ax[dim] = f(axes_x[dim], axes_y[dim])
    end
    ntuple(i -> ax[i], length(axes_x))
end

function broadcaststack(x::AbstractArray, y::AbstractArray, f, dims)
    ndims(x) == ndims(y) || throw(DimensionMismatch("arrays must have the same number of dimensions"))

    for dim in 1:ndims(x)
        if dim ∈ dims
            if dim > ndims(x)
                throw(ArgumentError("dim must lie in 1 <= dim <= ndims(x)"))
            end
        else
            axes(x, dim) == axes(y, dim) || throw(DimensionMismatch("non-concatenated axes must be identical"))
        end
    end

    axes_cat = _subsetaxes(_union, axes(x), axes(y), dims)

    xy_cat = similar(x, promote_type(eltype(x), eltype(y)), axes_cat)
    eltype(xy_cat) <: Number && fill!(xy_cat, zero(eltype(xy_cat)))

    common_ax = CartesianIndices(_subsetaxes(intersect, axes(x), axes(y), dims))

    for arr in (x, y)
        @inbounds for I in CartesianIndices(arr)
            I in  common_ax && continue
            xy_cat[I] = arr[I]
        end
    end

    @inbounds for I in common_ax
        xy_cat[I] = f(x[I], y[I])
    end

    xy_cat
end

"""
    Flip(f)

Flip the arguments of a binary function `f`, so that `Flip(f)(x, y) == f(y,x)`.

# Examples
```jldoctest flip
julia> flip1 = ParallelUtilities.Flip(vcat);

julia> flip1(2, 3)
2-element Vector{Int64}:
 3
 2
```

Two flips pop the original function back:

```jldoctest flip
julia> flip2 = ParallelUtilities.Flip(flip1);

julia> flip2(2, 3)
2-element Vector{Int64}:
 2
 3
```
"""
struct Flip{F} <: Function
    f :: F
end

(o::Flip)(x, y) = o.f(y, x)

Flip(o::Flip) = o.f

# Perserve the commutative tag
Flip(c::Commutative) = Commutative(Flip(c.f))
Flip(b::BroadcastFunction{1}) = BroadcastFunction{2}(Flip(b.f))
Flip(b::BroadcastFunction{2}) = BroadcastFunction{1}(Flip(b.f))
