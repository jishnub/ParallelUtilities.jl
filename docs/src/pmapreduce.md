```@meta
DocTestSetup  = quote
    using ParallelUtilities
end
```

# Parallel mapreduce

There are two modes of evaluating a parallel mapreduce that vary only in the arguments that the mapping function accepts.

1. Iterated zip, where one element from the zipped iterators is splatted and passed as arguments to the mapping function. In this case the function must accept as many arguments as the number of iterators passed to mapreduce. This is analogous to a serial `mapreduce`

2. Non-iterated product, in which case the iterator product of the arguments is distributed evenly across the workers. The mapping function in this case should accept one argument that is a collection of `Tuple`s of values. It may iterate over the argument to obtain the individual `Tuple`s.

Each process involved in a `pmapreduce` operation carries out a local `mapreduce`, followed by a reduction across processes. The reduction is carried out in the form of a binary tree. The reduction happens in three stages:

1. A local reduction as a part of `mapreduce`
2. A reduction on the host across the workers on the same host. Typically on an HPC system there is an independent reduction on each node across the processes on that node.
3. A global reduction across hosts.

The reduction operator is assumed to be associative, and reproducibility of floating-point operations is not guaranteed. For associative reductions look into various `mapfold*` methods provided by other packages, such as [`Transducers`](https://github.com/JuliaFolds/Transducers.jl). The reduction operator is not assumed to be commutative.

A `pmapreduce` might only benefit in performance if the mapping function runs for longer than the communication overhead across processes, or if each process has dedicated memory and returns large arrays that may not be collectively stored on one process.

## Iterated Zip

The syntax for a parallel map-reduce operation is quite similar to the serial `mapreduce`, with the replacement of `mapreduce` by `pmapreduce`.

Serial:

```julia
julia> mapreduce(x -> x^2, +, 1:100_000)
333338333350000
```

Parallel:

```julia
julia> pmapreduce(x -> x^2, +, 1:100_000)
333338333350000
```

We may check that parallel evaluation helps in performance for a long-running process.

```julia
julia> nworkers()
2

julia> @time mapreduce(x -> (sleep(1); x^2), +, 1:6);
  6.079191 seconds (54.18 k allocations: 3.237 MiB, 1.10% compilation time)

julia> @time pmapreduce(x -> (sleep(1); x^2), +, 1:6);
  3.365979 seconds (91.57 k allocations: 5.473 MiB, 0.87% compilation time)
```

## Non-iterated product

The second mode of usage is similar to MPI, where each process evaluates the same function once for different arguments. This is called using

```julia
pmapreduce_productsplit(f, op, iterators...)
```

In this function, the iterator product of the argument `iterators` is split evenly across the workers, and
the function `f` on each process receives one such section according to its rank. The argument is an iterator similar to an iterator product, and looping over it would produce Tuples `(iterators[1][i], iterators[2][i], ...)` where the index `i` depends on the rank of the worker as well as the local loop index.

As an example, we run this with 2 workers:

```julia
julia> pmapreduce_productsplit(ps -> (@show collect(ps)), vcat, 1:4)
      From worker 2:    collect(ps) = [(1,), (2,)]
      From worker 3:    collect(ps) = [(3,), (4,)]
4-element Vector{Tuple{Int64}}:
 (1,)
 (2,)
 (3,)
 (4,)

julia> pmapreduce_productsplit(ps -> (@show collect(ps)), vcat, 1:3, 1:2)
      From worker 2:    collect(ps) = [(1, 1), (2, 1), (3, 1)]
      From worker 3:    collect(ps) = [(1, 2), (2, 2), (3, 2)]
6-element Vector{Tuple{Int64, Int64}}:
 (1, 1)
 (2, 1)
 (3, 1)
 (1, 2)
 (2, 2)
 (3, 2)
```

Note that in each case the mapping function receives the entire collection of arguments in one go, unlike a standard `mapreduce` where the function receives the arguments individually. This is chosen so that the function may perform any one-time compute-intensive task for the entire range before looping over the argument values.

Each process might return one or more values that are subsequently reduced in parallel.

!!! note
    At present the `iterators` passed as arguments to `pmapreduce_productsplit` may only be strictly increasing ranges. This might be relaxed in the future.

The argument `ps` passed on to each worker is a [`ParallelUtilities.ProductSplit`](@ref) object. This has several methods defined for it that might aid in evaluating the mapping function locally.

### ProductSplit

A `ProductSplit` object `ps` holds the section of the iterator product that is assigned to the worker. It also encloses the worker rank and the size of the worker pool, similar to MPI's `Comm_rank` and `Comm_size`. These may be accessed as `workerrank(ps)` and `nworkers(ps)`. Unlike MPI though, the rank goes from `1` to `np`. An example where the worker rank is used (on 2 workers) is

```julia
julia> pmapreduce_productsplit(ps -> ones(2) * workerrank(ps), hcat, 1:nworkers())
2×2 Matrix{Float64}:
 1.0  2.0
 1.0  2.0
```

The way to construct a `ProductSplit` object is `ParallelUtilities.ProductSplit(tuple_of_iterators, nworkers, worker_rank)`

```jldoctest productsplit; setup=:(using ParallelUtilities)
julia> ps = ParallelUtilities.ProductSplit((1:2, 3:4), 2, 1)
2-element ProductSplit [(1, 3), ... , (2, 3)]

julia> ps |> collect
2-element Vector{Tuple{Int64, Int64}}:
 (1, 3)
 (2, 3)
```

A `ProductSplit` that wraps `AbstractUnitRange`s has several efficient functions defined for it, such as `length`, `minimumelement`, `maximumelement` and `getindex`, each of which returns in `O(1)` without iterating over the object.

```jldoctest productsplit
julia> ps[1]
(1, 3)
```

The function `maximumelement`, `minimumelement` and `extremaelement` treat the `ProductSplit` object as a linear view of an `n`-dimensional iterator product. These functions look through the elements in the `dim`-th dimension of the iterator product, and if possible, return the corresponding extremal element in `O(1)` time. Similarly, for a `ProductSplit` object that wraps `AbstractUnitRange`s, it's possible to know if a value is contained in the iterator in `O(1)` time.

```julia productsplit
julia> ps = ParallelUtilities.ProductSplit((1:100_000, 1:100_000, 1:100_000), 25000, 1500)
40000000000-element ProductSplit [(1, 1, 5997), ... , (100000, 100000, 6000)]

julia> @btime (3,3,5998) in $ps
  111.399 ns (0 allocations: 0 bytes)
true

julia> @btime ParallelUtilities.maximumelement($ps, dims = 1)
  76.534 ns (0 allocations: 0 bytes)
100000

julia> @btime ParallelUtilities.minimumelement($ps, dims = 2)
  73.724 ns (0 allocations: 0 bytes)
1

julia> @btime ParallelUtilities.extremaelement($ps, dims = 2)
  76.332 ns (0 allocations: 0 bytes)
(1, 100000)
```

The number of unique elements along a particular dimension may be obtained as

```julia productsplit
julia> @btime ParallelUtilities.nelements($ps, dims = 3)
  118.441 ns (0 allocations: 0 bytes)
4
```

It's also possible to drop the leading dimension of a `ProductSplit` that wraps `AbstractUnitRange`s to obtain an analogous operator that contains the unique elements along the remaining dimension. This is achieved using `ParallelUtilities.dropleading`.

```jldoctest productsplit
julia> ps = ParallelUtilities.ProductSplit((1:3, 1:3, 1:2), 4, 2)
5-element ProductSplit [(3, 2, 1), ... , (1, 1, 2)]

julia> collect(ps)
5-element Vector{Tuple{Int64, Int64, Int64}}:
 (3, 2, 1)
 (1, 3, 1)
 (2, 3, 1)
 (3, 3, 1)
 (1, 1, 2)

julia> ps2 = ParallelUtilities.dropleading(ps)
3-element ProductSection [(2, 1), ... , (1, 2)]

julia> collect(ps2)
3-element Vector{Tuple{Int64, Int64}}:
 (2, 1)
 (3, 1)
 (1, 2)
```

The process may be repeated multiple times:

```jldoctest productsplit
julia> collect(ParallelUtilities.dropleading(ps2))
2-element Vector{Tuple{Int64}}:
 (1,)
 (2,)
```

# Reduction Operators

Any standard Julia reduction operator may be passed to `pmapreduce`. Aside from this, this package defines certain operators that may be used as well in a reduction.

## Broadcasted elementwise operators

The general way to construct an elementwise operator using this package is using [`ParallelUtilities.BroadcastFunction`](@ref).

For example, a broadcasted sum operator may be constructed using
```jldoctest
julia> ParallelUtilities.BroadcastFunction(+);
```

The function call `ParallelUtilities.BroadcastFunction(op)(x, y)` perform the fused elementwise operation `op.(x, y)`.

!!! note "Julia 1.6 and above"
    Julia versions above `v"1.6"` provide a function `Base.BroadcastFunction` which is equivalent to `ParallelUtilities.BroadcastFunction`.

# Inplace assignment

The function [`ParallelUtilities.broadcastinplace`](@ref) may be used to construct a binary operator that broadcasts a function over its arguments and stores the result inplace in one of the arguments. This is particularly useful if the results in intermediate evaluations are not important, as this cuts down on allocations in the reduction.

Several operators for common functions are pre-defined for convenience.

1. [`ParallelUtilities.elementwisesum!`](@ref)
2. [`ParallelUtilities.elementwiseproduct!`](@ref)
3. [`ParallelUtilities.elementwisemin!`](@ref)
4. [`ParallelUtilities.elementwisemax!`](@ref)

Each of these functions overwrites the first argument with the result.

!!! warn
    The pre-defined elementwise operators are assumed to be commutative, so, if used in `pmapreduce`, the order of arguments passed to the function is not guaranteed. In particular this might not be in order of the `workerrank`. These functions should only be used if both the arguments support the inplace assignment, eg. if they have identical axes.

## Flip

The [`ParallelUtilities.Flip`](@ref) function may be used to wrap a binary function to flips the order of arguments. For example

```jldoctest
julia> vcat(1,2)
2-element Vector{Int64}:
 1
 2

julia> ParallelUtilities.Flip(vcat)(1,2)
2-element Vector{Int64}:
 2
 1
```

`Flip` may be combined with inplace assignment operators to change the argument that is overwritten.

```jldoctest
julia> x = ones(3); y = ones(3);

julia> op1 = ParallelUtilities.elementwisesum!; # overwrites the first argument

julia> op1(x, y); x
3-element Vector{Float64}:
 2.0
 2.0
 2.0

julia> x = ones(3); y = ones(3);

julia> op2 = ParallelUtilities.Flip(op1); # ovrewrites the second argument

julia> op2(x, y); y
3-element Vector{Float64}:
 2.0
 2.0
 2.0
```

## BroadcastStack

This function may be used to combine arrays having overlapping axes to obtain a new array that spans the union of axes of the arguments. The overlapping section is computed by applying the reduction function to that section.

We construct a function that concatenates arrays along the first dimension with overlapping indices summed.
```jldoctest broadcaststack
julia> f = ParallelUtilities.BroadcastStack(+, 1);
```

We apply this to two arrays having different indices
```jldoctest broadcaststack
julia> f(ones(2), ones(4))
4-element Vector{Float64}:
 2.0
 2.0
 1.0
 1.0
```

This function is useful to reduce [`OffsetArray`s](https://github.com/JuliaArrays/OffsetArrays.jl) where each process evaluates a potentially overlapping section of the entire array.

!!! note
    A `BroadcastStack` function requires its arguments to have the same dimensionality, and identical axes along non-concatenated dimensions. In particular it is not possible to block-concatenate arrays using this function.

!!! note
    A `BroadcastStack` function does not operate in-place.

## Commutative

In general this package does not assume that a reduction operator is commutative. It's possible to declare an operator to be commutative in its arguments by wrapping it in the tag [`ParallelUtilities.Commutative`](@ref). 
