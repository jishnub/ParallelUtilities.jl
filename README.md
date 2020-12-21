# ParallelUtilities.jl

[![Build status](https://github.com/jishnub/ParallelUtilities.jl/workflows/CI/badge.svg)](https://github.com/jishnub/ParallelUtilities.jl/actions)
[![codecov](https://codecov.io/gh/jishnub/ParallelUtilities.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jishnub/ParallelUtilities.jl)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jishnub.github.io/ParallelUtilities.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jishnub.github.io/ParallelUtilities.jl/dev)

Parallel mapreduce and other helpful functions for HPC, meant primarily for embarassingly parallel operations that often require one to split up a list of tasks into subsections that may be processed on individual cores.

# Installation

Install the package using

```julia
pkg> add ParallelUtilities
julia> using ParallelUtilities 
```
# Quick start

```julia
julia> addprocs(2)
2-element Array{Int64,1}:
 2
 3

julia> @everywhere using ParallelUtilities

julia> pmapreduce(x -> ones(2) .* myid(), x -> hcat(x...), 1:nworkers())
2×2 Array{Float64,2}:
 2.0  3.0
 2.0  3.0

julia> pmapreduce_commutative(x -> ones(2) .* myid(), sum, 1:nworkers())
2-element Array{Float64,1}:
 5.0
 5.0

julia> pmapsum(x -> ones(2) .* myid(), 1:nworkers())
2-element Array{Float64,1}:
 5.0
 5.0
```

# Performance

The `pmapreduce`-related functions are expected to be more performant than `@distributed` for loops. As an example, running the following on a Slurm cluster using 2 nodes with 28 cores on each leads to

```julia
julia> @time @distributed (+) for i=1:nworkers()
           ones(10_000, 1_000)
       end;
 22.355047 seconds (7.05 M allocations: 8.451 GiB, 6.73% gc time)

julia> @time pmapsum(x -> ones(10_000, 1_000), 1:nworkers());
  2.672838 seconds (52.83 k allocations: 78.295 MiB, 0.53% gc time)
```

The difference becomes more apparent as larger data needs to be communicated across workers. This is because `ParallelUtilities.pmapreduce*` perform local reductions on each node before communicating across nodes.

# Usage

The package splits up a collection of ranges into subparts of roughly equal length, so that all the cores are approximately equally loaded. This is best understood using an example: let's say that we have a function `f` that is defined as   

```julia
julia> @everywhere begin 
       f(x,y,z) = x+y+z
       end
```

where each parameter takes up values in a range, and we would like to sample the entire parameter space. As an example, we choose the ranges to be 

```julia
julia> xrange, yrange, zrange = 1:3, 2:4, 3:6 # ranges should be strictly increasing
```

There are a total of 36 possible `(x,y,z)` combinations possible given these ranges. Let's say that we would like to split the evaluation of the function over 10 processors. We describe the simple way to evaluate this and then explain how this is achieved.

The set of parameters may be split up using the function `ProductSplit`. In this example each of the 10 processors receive a chunk as listed below

```julia
julia> [collect(ProductSplit((xrange,yrange,zrange),10,i)) for i=1:10]
10-element Array{Array{Tuple{Int64,Int64,Int64},1},1}:
 [(1, 2, 3), (2, 2, 3), (3, 2, 3), (1, 3, 3)]
 [(2, 3, 3), (3, 3, 3), (1, 4, 3), (2, 4, 3)]
 [(3, 4, 3), (1, 2, 4), (2, 2, 4), (3, 2, 4)]
 [(1, 3, 4), (2, 3, 4), (3, 3, 4), (1, 4, 4)]
 [(2, 4, 4), (3, 4, 4), (1, 2, 5), (2, 2, 5)]
 [(3, 2, 5), (1, 3, 5), (2, 3, 5), (3, 3, 5)]
 [(1, 4, 5), (2, 4, 5), (3, 4, 5)]           
 [(1, 2, 6), (2, 2, 6), (3, 2, 6)]           
 [(1, 3, 6), (2, 3, 6), (3, 3, 6)]           
 [(1, 4, 6), (2, 4, 6), (3, 4, 6)] 
```

The first six processors receive 4 tuples of parameters each and the final four receive 3 each. This is the splitting used by the various functions described next.

## pmap-related functions

The package provides versions of `pmap` with an optional reduction. These differ from the one provided by `Distributed` in a few key aspects: firstly, the iterator product of the argument is what is passed to the function and not the arguments by elementwise, so the i-th task will be `Iterators.product(args...)[i]` and not `[x[i] for x in args]`. Specifically the second set of parameters in the example above will be `(2,2,3)` and not `(2,3,4)`.

Secondly, the iterator is passed to the function in batches and not elementwise, and it is left to the function to iterate over the collection. Thirdly, the tasks are passed on to processors sorted by rank, so the first task is passed to the first processor and the last to the last active worker. The tasks are also approximately evenly distributed across processors. The exported function `pmapbatch_elementwise` passes the elements to the function one-by-one as splatted tuples. This produces the same result as `pmap` for a single range as the argument.

### pmapbatch and pmapbatch_elementwise

As an example we demonstrate how to evaluate the function `f` for the ranges of parameters listed above:

```julia
julia> p = pmapbatch_elementwise(f, (xrange,yrange,zrange));

julia> Tuple(p)
(6, 7, 8, 7, 8, 9, 8, 9, 10, 7, 8, 9, 8, 9, 10, 9, 10, 11, 8, 9, 10, 9, 10, 11, 10, 11, 12, 9, 10, 11, 10, 11, 12, 11, 12, 13)
```

There is also a function `pmapbatch` that deals with batches of parameters that are passed to each processor, and `pmap_elementwise` calls this function under the hood to process the parameters one by one. We may use this directly as well if we need the entire batch for some reason (eg. reading values off a disk, which needs to be done once for the entire set and not for every parameter). As an example we demonstrate how to obtain the same result as above using `pmapbatch`:

```julia
julia> p = pmapbatch(x->[f(i...) for i in x], (xrange,yrange,zrange));

julia> Tuple(p)
(6, 7, 8, 7, 8, 9, 8, 9, 10, 7, 8, 9, 8, 9, 10, 9, 10, 11, 8, 9, 10, 9, 10, 11, 10, 11, 12, 9, 10, 11, 10, 11, 12, 11, 12, 13)
```

### pmapsum and pmapreduce

Often a parallel execution is followed by a reduction (eg. a sum over the results). A reduction may be commutative (in which case the order of results do not matter), or non-commutative (in which the order does matter). There are two functions that are exported that carry out these tasks: `pmapreduce_commutative` and `pmapreduce`, where the former does not preserve ordering and the latter does. For convenience, the package also provides the function `pmapsum` that chooses `sum` as the reduction operator. The map-reduce operation is similar in many ways to the distributed `for` loop provided by julia, but the main difference is that the reduction operation is not binary for the functions in this package (eg. we need `sum` and not `(+)`to add the results). There is also the difference as above that the function gets the parameters in batches, with functions having the suffix `_elementwise` taking on parameters individually as splatted `Tuple`s. The function `pmapreduce` does not take on parameters elementwise at this point, although this might be implemented in the future.

As an example, to sum up a list of numbers in parallel we may call
```julia
julia> pmapsum_elementwise(identity, 1:1000)
500500
```

Here the mapped function is taken to by `identity` which just returns its argument. To sum the squares of the numbers in a list we may use 

```julia
julia> pmapsum_elementwise(x -> x^2, 1:1000)
333833500
```

We may choose an arbitrary reduction operator in the function `pmapreduce` and `pmapreduce_commutative`, and the elementwise function `pmapreduce_commutative_elementwise`. The reductions are carried out as a binary tree across all workers.

```julia
# Compute 1^2 * 2^2 * 3^2 in parallel
julia> pmapreduce_commutative_elementwise(x -> x^2, prod, 1:3)
36
```

The function `pmapreduce` sorts the results obtained from each processor, so it is useful for concatenations.

```julia
julia> workers()
2-element Array{Int64,1}:
 2
 3

# The signature is pmapreduce(fmap, freduce, range_or_tuple_of_ranges)
julia> pmapreduce(x -> ones(2).*myid(), x -> hcat(x...), 1:nworkers())
2×2 Array{Float64,2}:
 2.0  3.0
 2.0  3.0
```

The functions `pmapreduce` produces the same result as `pmapreduce_commutative` if the reduction operator is commutative (ie. the order of results received from the children workers does not matter).

The function `pmapsum` sets the reduction function to `sum`.

```julia
julia> sum(workers())
5

# We compute ones(2).*sum(workers()) in parallel
julia> pmapsum(x -> ones(2).*myid(), 1:nworkers())
2-element Array{Float64,1}:
 5.0
 5.0
```

It is possible to specify the return types of the map and reduce operations in these functions. To specify the return types use the following variants:

```julia
# Signature is pmapreduce(fmap, Tmap, freduce, Treduce, range_or_tuple_of_ranges)
julia> pmapreduce(x -> ones(2).*myid(), Vector{Float64}, x -> hcat(x...), Matrix{Float64}, 1:nworkers())
2×2 Array{Float64,2}:
 2.0  3.0
 2.0  3.0

# Signature is pmapsum(fmap, Tmap, range_or_tuple_of_ranges)
julia> pmapsum(x -> ones(2).*myid(), Vector{Float64}, 1:nworkers())
2-element Array{Float64,1}:
 5.0
 5.0
```

Specifying the types would lead to a type coercion if possible, or an error if a conversion is not possible. This might help in asserting the correctness of the result obtained. For example:

```julia
# The result is converted from Vector{Float64} to Vector{Int}. 
# Conversion works as the numbers are integers
julia> pmapsum(x -> ones(2).*myid(), Vector{Int}, 1:nworkers())
2-element Array{Int64,1}:
 5
 5

# Conversion fails here as the numbers aren't integers
julia> pmapsum(x -> rand(2), Vector{Int}, 1:nworkers())
ERROR: On worker 2:
InexactError: Int64(0.7742577217010362)
```

### Progress bar

The progress of the map-reduce operation might be tracked by setting the keyword argument `showprogress` to true. This might be useful in case certain workers have a heavier load than others.

```julia
# Running on 8 workers, artificially induce load using sleep
julia> pmapreduce(x -> (sleep(myid()); myid()), x -> hcat(x...), 1:nworkers(), showprogress=true)
Progress in pmapreduce : 100%|██████████████████████████████████████████████████| Time: 0:00:09
1×8 Array{Int64,2}:
 2  3  4  5  6  7  8  9

julia> pmapreduce(x -> (sleep(myid()); myid()), x -> hcat(x...), 1:nworkers(), showprogress=true, progressdesc="Progress : ")
Progress : 100%|████████████████████████████████████████████████████████████████| Time: 0:00:09
1×8 Array{Int64,2}:
 2  3  4  5  6  7  8  9
```

Note that this does not track the progress of the individual maps, it merely tracks how many are completed. The progress of the individual maps may be tracked by explicitly passing a `RemoteChannel` to the mapping function and pushing the progress status to it from the workers.

### Why two mapreduce functions?

The two separate functions `pmapreduce` and `pmapreduce_commutative` exist for historical reasons. They use different binary tree structures for reduction. The commutative one might be removed in the future in favour of `pmapreduce`.

## ProductSplit

In the above examples we have talked about the tasks being distributed approximately equally among the workers without going into details about the distribution, which is what we describe here. The package provides an iterator `ProductSplit` that lists that ranges of parameters that would be passed on to each core. This may equivalently be achieved using an

```Iterators.Take{Iterators.Drop{Iterators.ProductIterator}}```

with appropriately chosen parameters, and in many ways a `ProductSplit` behaves similarly. However a `ProductSplit` supports several extra features such as `O(1)` indexing, which eliminates the need to actually iterate over it in many scenarios.

The signature of the constructor is 

```julia 
ProductSplit(tuple_of_ranges, number_of_processors, processor_rank)
```

where `processor_rank` takes up values in `1:number_of_processors`. Note that this is different from MPI where the rank starts from 0. For example, we check the tasks that are passed on to the processor number 4:

```julia
julia> ps = ProductSplit((xrange, yrange, zrange), 10, 4);

julia> collect(ps)
4-element Array{Tuple{Int64,Int64,Int64},1}:
 (1, 3, 4)
 (2, 3, 4)
 (3, 3, 4)
 (1, 4, 4)
```

where the object loops over values of `(x,y,z)`, and the values are sorted in reverse lexicographic order (the last index increases the slowest while the first index increases the fastest). The ranges roll over as expected. The tasks are evenly distributed with the remainders being split among the first few processors. In this example the first six processors receive 4 tasks each and the last four receive 3 each. We can see this by evaluating the length of the `ProductSplit` operator on each processor

```julia
julia> Tuple(length(ProductSplit((xrange,yrange,zrange), 10, i)) for i=1:10)
(4, 4, 4, 4, 4, 4, 3, 3, 3, 3)
```

### Indexing

The iterator supports fast indexing
```julia
julia> ps[3]
(3, 3, 4)

julia> @btime $ps[3]
  9.493 ns (0 allocations: 0 bytes)
(3, 3, 4)
```

This is useful if we have a large number of parameters to analyze on each processor.

```julia
julia> xrange_long,yrange_long,zrange_long = 1:3000,1:3000,1:3000
(1:3000, 1:3000, 1:3000)

julia> params_long = (xrange_long,yrange_long,zrange_long);

julia> ps = ProductSplit(params_long, 10, 3)
2700000000-element ProductSplit((1:3000, 1:3000, 1:3000), 10, 3)
[(1, 1, 601), ... , (3000, 3000, 900)]

# Evaluate length using random ranges to avoid compiler optimizations
julia> @btime length(p) setup = (n = rand(3000:4000); p = ProductSplit((1:n,1:n,1:n), 200, 2));
  2.674 ns (0 allocations: 0 bytes)

julia> @btime $ps_long[1000000] # also fast, does not iterate
  32.530 ns (0 allocations: 0 bytes)
(1000, 334, 901)

julia> @btime first($ps_long)
  31.854 ns (0 allocations: 0 bytes)
(1, 1, 901)

julia> @btime last($ps_long)
  31.603 ns (0 allocations: 0 bytes)
(3000, 3000, 1200)
```

We may evaluate whether or not a value exists in the list and its index in `O(1)` time.

```julia
julia> val = (3,3,4)
(3, 3, 4)

julia> val in ps
true

julia> localindex(ps, val)
3

julia> val = (10,2,901);

julia> @btime $val in $ps_long
  50.183 ns (0 allocations: 0 bytes)
true

julia> @btime localindex($ps_long, $val)
  104.513 ns (0 allocations: 0 bytes)
3010
```

Another useful function is `whichproc` that returns the rank of the processor a specific set of parameters will be on, given the total number of processors. This is also computed using a binary search.

```julia
julia> whichproc(params_long, val, 10)
4

julia> @btime whichproc($params_long, $val, 10);
  353.706 ns (0 allocations: 0 bytes)
```

### Extrema

We may compute the ranges of each variable on any processor in `O(1)` time. 

```julia
julia> extrema(ps, dim = 2) # extrema of the second parameter on this processor
(3, 4)

julia> Tuple(extrema(ps, dim = i) for i in 1:3)
((1, 3), (3, 4), (4, 4))

# Minimum and maximum work similarly

julia> (minimum(ps, dim = 2), maximum(ps, dim = 2))
(3, 4)

julia> @btime extrema($ps_long, dim=2)
  52.813 ns (0 allocations: 0 bytes)
(1, 3000)
```
