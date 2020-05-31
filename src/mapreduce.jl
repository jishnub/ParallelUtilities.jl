# Store the rank with the value, necessary for collecting values in order`
struct pval{T}
	rank :: Int
	value :: T
end

# Function to obtain the value of pval types
@inline value(p::pval) = p.value
@inline value(p::Any) = p

Base.convert(::Type{pval{T}},p::pval) where {T} = pval{T}(p.rank,convert(T,value(p)))

Base.:(==)(p1::pval,p2::pval) = (p1.rank == p2.rank) && (value(p1) == value(p2))

############################################################################################
# Map
############################################################################################

# Wrap a pval around the mapped value if sorting is necessary
@inline function maybepvalput!(pipe::BranchChannel{T},rank,val) where {T}
	put!(pipe.selfchannels.out,val)
end
@inline function maybepvalput!(pipe::BranchChannel{T},rank,val) where {T<:pval}
	valT = T(rank,value(val))
	put!(pipe.selfchannels.out,valT)
end

function indicatemapprogress!(::Nothing,rank) end
function indicatemapprogress!(progress::RemoteChannel,rank)
	put!(progress,(true,false,rank))
end

function indicatefailure!(::Nothing,rank) end
function indicatefailure!(progress::RemoteChannel,rank)
	put!(progress,(false,false,rank))
end

function mapTreeNode(fmap::Function,iterator,rank,pipe::BranchChannel,
	progress::Union{Nothing,RemoteChannel},args...;kwargs...)
	# Evaluate the function
	# Store the error flag locally
	# If there are no errors then store the result locally
	# No communication with other nodes happens here
	try
		res = fmap(iterator,args...;kwargs...)
		maybepvalput!(pipe,rank,res)
		put!(pipe.selfchannels.err,false)
		indicatemapprogress!(progress,rank)
	catch
		put!(pipe.selfchannels.err,true)
		indicatefailure!(progress,rank)
		rethrow()
	end
end

############################################################################################
# Reduction
############################################################################################

abstract type Ordering end
struct Sorted <: Ordering end
struct Unsorted <: Ordering end

abstract type ReductionNode end
struct TopTreeNode <: ReductionNode
	rank :: Int
end
struct SubTreeNode <: ReductionNode
	rank :: Int
end

function reducedvalue(freduce::Function,rank,
	pipe::BranchChannel,ifsorted::Ordering)

	reducedvalue(freduce,
		rank > 0 ? SubTreeNode(rank) : TopTreeNode(rank),
		pipe,ifsorted)
end

function reducedvalue(freduce::Function,node::SubTreeNode,
	pipe::BranchChannel{Tmap,Tred},::Unsorted) where {Tmap,Tred}

	self = take!(pipe.selfchannels.out) :: Tmap
	N = nchildren(pipe)
	vals = Vector{Tred}(undef,N+1)
	@sync begin
		@async vals[1] = freduce([self]) :: Tred
		@async for i=1:N
			vals[i+1] = take!(pipe.childrenchannels.out)::Tred
		end
	end
	
	freduce(vals)
end
function reducedvalue(freduce::Function,node::TopTreeNode,
	pipe::BranchChannel{<:Any,Tred},::Unsorted) where {Tred}

	N = nchildren(pipe)
	if N == 0
		# shouldn't reach this
		error("Nodes on the top tree must have children")
	end
	vals = Vector{Tred}(undef,N)
	for i=1:N
		vals[i] = take!(pipe.childrenchannels.out)::Tred
	end

	freduce(vals)
end

function reducedvalue(freduce::Function,node::SubTreeNode,
	pipe::BranchChannel{Tmap,Tred},::Sorted) where {Tmap,Tred}

	rank = node.rank
	N = nchildren(pipe)
	leftchild = N > 0
	vals = Vector{Tred}(undef,N + 1)
	@sync begin
		@async begin
			selfval = take!(pipe.selfchannels.out)::Tmap
			selfvalred = freduce((value(selfval),))
			pv = pval(rank,selfvalred)
			ind = leftchild + 1
			vals[ind] = pv
		end
		@async for i=1:N 
			pv = take!(pipe.childrenchannels.out) :: Tred
			shift = pv.rank > rank ? 1 : -1
			ind = shift + leftchild + 1
			vals[ind] = pv
		end
	end

	Tred(rank,freduce(value(v) for v in vals))
end
function reducedvalue(freduce::Function,node::TopTreeNode,
	pipe::BranchChannel{<:Any,Tred},::Sorted) where {Tred}

	rank = node.rank
	N = nchildren(pipe)
	leftchild = N > 0
	@assert leftchild "Nodes on the top tree must have children"
	vals = Vector{Tred}(undef,N)
	for i=1:N
		pv = take!(pipe.childrenchannels.out) :: Tred
		vals[i] = pv
	end
	sort!(vals,by=pv->pv.rank)

	Tred(rank,freduce(value(v) for v in vals))
end

function indicatereduceprogress!(::Nothing,rank) end
function indicatereduceprogress!(progress::RemoteChannel,rank)
	put!(progress,(false,true,rank))
end

function reduceTreeNode(freduce::Function,rank,pipe::BranchChannel,
	ifsort::Ordering,progress)
	
	reduceTreeNode(freduce,
		rank > 0 ? SubTreeNode(rank) : TopTreeNode(rank),
		pipe,ifsort,progress)
end

function checkerror(::SubTreeNode,pipe::BranchChannel)
	selferr = take!(pipe.selfchannels.err)
	childrenerr = any(take!(pipe.childrenchannels.err) for i=1:nchildren(pipe))
	selferr || childrenerr
end
function checkerror(::TopTreeNode,pipe::BranchChannel)
	any(take!(pipe.childrenchannels.err) for i=1:nchildren(pipe))
end

function reduceTreeNode(freduce::Function,node::ReductionNode,
	pipe::BranchChannel{<:Any,Tred},
	ifsort::Ordering,progress::Union{Nothing,RemoteChannel}) where {Tred}
	# This function that communicates with the parent and children

	# Start by checking if there is any error locally in the map,
	# and if there's none then check if there are any errors on the children
	anyerr = checkerror(node,pipe)
	rank = node.rank
	# Evaluate the reduction only if there's no error
	# In either case push the error flag to the parent
	if !anyerr
		try
			res = reducedvalue(freduce,node,pipe,ifsort) :: Tred
			put!(pipe.parentchannels.out,res)
			put!(pipe.parentchannels.err,false)
			indicatereduceprogress!(progress,rank)
		catch e
			put!(pipe.parentchannels.err,true)
			indicatefailure!(progress,rank)
			rethrow()
		end
	else
		put!(pipe.parentchannels.err,true)
		indicatefailure!(progress,rank)
	end

	finalize(pipe)
end

function return_unless_error(r::RemoteChannelContainer)
	anyerror = take!(r.err)
	if !anyerror
		return value(take!(r.out))
	end
end

function return_unless_error(b::BranchChannel)
	return_unless_error(b.parentchannels)
end

function pmapreduceworkers(fmap::Function,freduce::Function,iterators::Tuple,
	tree,branches,ord::Ordering,args...;
	showprogress::Bool = false, progressdesc = "Progress in pmapreduce : ",
	kwargs...)

	num_workers_active = nworkersactive(iterators)
	Nmaptotal = num_workers_active
	Nreducetotal = length(branches)
	extrareducenodes = Nreducetotal - Nmaptotal
	
	Nprogress = Nmaptotal+Nreducetotal
	progresschannel = RemoteChannel(()->Channel{Tuple{Bool,Bool,Int}}(
						ifelse(showprogress,Nprogress,0)))
	progressbar = Progress(Nprogress,1,progressdesc)

	@sync begin

		for (ind,mypipe) in enumerate(branches)
			p = mypipe.p
			ind_reduced = ind - extrareducenodes
			rank = ind_reduced
			if ind_reduced > 0
				iterable_on_proc = ProductSplit(iterators,num_workers_active,rank)

				@spawnat p mapTreeNode(fmap,iterable_on_proc,rank,mypipe,
					showprogress ? progresschannel : nothing,
					args...;kwargs...)

				@spawnat p reduceTreeNode(freduce,SubTreeNode(rank),
					mypipe,ord,
					showprogress ? progresschannel : nothing)
			else
				@spawnat p reduceTreeNode(freduce,TopTreeNode(rank),
					mypipe,ord,
					showprogress ? progresschannel : nothing)
			end
		end

		if showprogress

			mapdone,reducedone = 0,0

			for i = 1:Nprogress
				mapflag,redflag,rank = take!(progresschannel)
				# both flags are false in case of an error
				mapflag || redflag || break

				mapdone += mapflag
				reducedone += redflag

				if mapdone != Nmaptotal && reducedone != Nreducetotal
					showvalues = [
					(:map,string(mapdone)*"/"*string(Nmaptotal)),
					(:reduce,string(reducedone)*"/"*string(Nreducetotal))
					]

				elseif reducedone != Nreducetotal
					showvalues = [
					(:reduce,string(reducedone)*"/"*string(Nreducetotal))
					]
				else
					showvalues = []
				end

				next!(progressbar;showvalues=showvalues)
			end
		end
	end

	return_unless_error(topbranch(tree,branches))
end

"""
	pmapreduce_commutative(fmap, freduce, iterators, [mapargs...]; 
		<keyword arguments>, [mapkwargs...])

Evaluate a parallel mapreduce over the range spanned by 
the outer product of the iterators. The operation
is assumed to be commutative, results obtained may be incorrect otherwise.

The argument  `iterators` must be a strictly-increasing range of integers, 
or a tuple of such ranges. 
The outer product of the ranges is split evenly across the workers. 
The function `fmap` receives a `ProductSplit` iterator as its first argument
that acts as a collection of tuples. One may index into a `ProductSplit` 
or iterate over one to access individual tuples of integers.

The reduction function `freduce` is expected to accept a collection of mapped values.
Note that this is different from the standard `mapreduce` operation in julia that 
expects a binary reduction operator. For example, `fmap` should be 
`sum` and not `+`. In case a binary operator `op` is to be passed, one may wrap it in
an anonymous function as `x->reduce(op,x)`, or as `x->op(x...)` in case the operator
accepts multiple arguments that are processed in pairs.

Arguments `mapargs` and keyword arguments `mapkwargs` — if provided — are 
passed on to the mapping function `fmap`.

	pmapreduce_commutative(fmap, Tmap, freduce, Treduce, iterators, [mapargs...]; 
		<keyword arguments>, [mapkwargs...])

The types `Tmap` and `Treduce` are the return types of the map and 
reduce operations respectively. The returned values will be coerced to 
the specified types if possible, throwing an error otherwise. 

# Keyword Arguments

- `showprogress::Bool = false` : Displays a progress-bar if set to true
- `progressdesc = "Progress in pmapreduce : "` : Leading text in the progress-bar

# Examples
```jldoctest
julia> pmapreduce_commutative(x->ones(2), x->hcat(x...), 1:4)
2×4 Array{Float64,2}:
 1.0  1.0  1.0  1.0
 1.0  1.0  1.0  1.0

julia> pmapreduce_commutative(x->ones(2), Vector{Int64}, x->hcat(x...), Matrix{Int64}, 1:4)
2×4 Array{Int64,2}:
 1  1  1  1
 1  1  1  1
```

See also: [`pmapreduce_commutative_elementwise`](@ref), [`pmapreduce`](@ref), [`pmapsum`](@ref)
"""
function pmapreduce_commutative(fmap::Function,Tmap::Type,
	freduce::Function,Tred::Type,iterators::Tuple,args...;
	kwargs...)
	
	tree,branches = createbranchchannels(Tmap,Tred,iterators,
		SegmentedSequentialBinaryTree)
	pmapreduceworkers(fmap,freduce,iterators,tree,
		branches,Unsorted(),args...;kwargs...)
end

function pmapreduce_commutative(fmap::Function,freduce::Function,
	iterators::Tuple,args...;kwargs...)

	pmapreduce_commutative(fmap,Any,freduce,Any,iterators,args...;kwargs...)
end

function pmapreduce_commutative(fmap::Function,Tmap::Type,
	freduce::Function,Tred::Type,
	iterable,args...;kwargs...)
	pmapreduce_commutative(fmap,Tmap,freduce,Tred,(iterable,),args...;kwargs...)
end

function pmapreduce_commutative(fmap::Function,freduce::Function,iterable,args...;kwargs...)
	pmapreduce_commutative(fmap,freduce,(iterable,),args...;kwargs...)
end

"""
	pmapreduce_commutative_elementwise(fmap, freduce, iterators, [mapargs...]; 
		<keyword arguments>, [mapkwargs...])

Evaluate a parallel mapreduce over the range spanned by 
the outer product of the iterators. 
The argument `iterators` must be a strictly-increasing range of integers, 
or a tuple of such ranges. The map is evaluated elementwise 
over the entire range of parameters.
The reduction is assumed to be commutative, 
results obtained may be incorrect otherwise.

The reduction function `freduce` is expected to accept a collection of mapped values.
Note that this is different from the standard `mapreduce` operation in julia that 
expects a binary reduction operator. For example, `fmap` should be 
`sum` and not `+`. In case a binary operator `op` is to be passed, one may wrap it in
an anonymous function as `x->reduce(op,x)`, or as `x->op(x...)` in case the operator
accepts multiple arguments that are processed in pairs.

Arguments `mapargs` and keyword arguments `mapkwargs` — if provided — are 
passed on to the mapping function `fmap`.

	pmapreduce_commutative_elementwise(fmap, Tmap, freduce, Treduce, iterators, [mapargs...]; 
		<keyword arguments>, [mapkwargs...])

The types `Tmap` and `Treduce` are the return types of the map and 
reduce operations respectively. The returned values will be coerced to 
the specified types if possible, throwing an error otherwise. 

# Keyword Arguments

- `showprogress::Bool = false` : Displays a progress-bar if set to true
- `progressdesc = "Progress in pmapreduce : "` : Leading text in the progress-bar

# Examples
```jldoctest
julia> pmapreduce_commutative_elementwise(x->x^2,prod,1:5)
14400

julia> pmapreduce_commutative_elementwise((x,y)->x+y,sum,(1:2,1:2))
12

julia> pmapreduce_commutative_elementwise(x->x^2,Int,prod,Float64,1:5)
14400.0
```

See also: [`pmapsum_commutative_elementwise`](@ref), [`pmapreduce_commutative`](@ref)
"""
function pmapreduce_commutative_elementwise(fmap::Function,Tmap::Type,
	freduce::Function,Tred::Type,iterable,args...;
	showprogress::Bool = false, progressdesc = "Progress in pmapreduce : ",
	kwargs...)
	
	pmapreduce_commutative(
		plist->freduce((fmap(x...,args...;kwargs...) for x in plist)),
		Tred,freduce,Tred,iterable,
		showprogress = showprogress, progressdesc = progressdesc)
end

function pmapreduce_commutative_elementwise(fmap::Function,freduce::Function,iterable,args...;
	showprogress::Bool = false, progressdesc = "Progress in pmapreduce : ",
	kwargs...)

	pmapreduce_commutative(
		plist->freduce((fmap(x...,args...;kwargs...) for x in plist)),
		freduce,iterable,
		showprogress = showprogress, progressdesc = progressdesc)
end

"""
	pmapsum(fmap, iterators, [mapargs...]; 
		<keyword arguments>, [mapkwargs...])

Evaluate a parallel mapreduce over the range spanned by 
the outer product of the iterators, where the reduction operation
is a sum.

The argument `iterators` must be a strictly-increasing range of integers, 
or a tuple of such ranges.
The outer product of the ranges is split evenly across the workers. 
The function `fmap` receives a `ProductSplit` iterator as its first argument
that acts as a collection of tuples. One may index into a `ProductSplit` 
or iterate over one to access individual tuples of integers.

Arguments `mapargs` and keyword arguments `mapkwargs` — if provided — are 
passed on to the mapping function `fmap`.

	pmapsum(fmap, Tmap, iterators, [mapargs...]; 
		<keyword arguments>, [mapkwargs...])

The types `Tmap` is the return types of the map. 
The returned values will be coerced to 
the specified type if possible, throwing an error otherwise.

# Keyword Arguments

- `showprogress::Bool = false` : Displays a progress-bar if set to true
- `progressdesc = "Progress in pmapsum : "` : Leading text in the progress-bar

# Examples
```jldoctest
julia> pmapsum(x->ones(2), 1:4)
2-element Array{Float64,1}:
 4.0
 4.0

julia> pmapsum(x->ones(2), Vector{Int64}, 1:4)
2-element Array{Int64,1}:
 4
 4
```

See also: [`pmapreduce`](@ref), [`pmapreduce_commutative`](@ref)
"""
function pmapsum(fmap::Function,T::Type,iterable,args...;kwargs...)
	pmapreduce_commutative(fmap,T,sum,T,iterable,args...;
		progressdesc = "Progress in pmapsum : ",kwargs...)
end

function pmapsum(fmap::Function,iterable,args...;kwargs...)
	pmapreduce_commutative(fmap,sum,iterable,args...;
		progressdesc = "Progress in pmapsum : ",kwargs...)
end

"""
	pmapsum_elementwise(fmap, iterators, [mapargs...]; 
		<keyword arguments>, [mapkwargs...])

Evaluate a parallel mapreduce over the range spanned by 
the outer product of the iterators, where the reduction operation is a sum. 
The argument `iterators` must be a strictly-increasing range of integers, 
or a tuple of such ranges. The map is evaluated elementwise 
over the entire range of parameters.

Arguments `mapargs` and keyword arguments `mapkwargs` — if provided — are 
passed on to the mapping function `fmap`.

	pmapsum_elementwise(fmap, Tmap, iterators, [mapargs...]; 
		<keyword arguments>, [mapkwargs...])

The type `Tmap` is the return type of the map. 
The returned values will be coerced to 
the specified type if possible, throwing an error otherwise.

# Keyword Arguments

- `showprogress::Bool = false` : Displays a progress-bar if set to true
- `progressdesc = "Progress in pmapreduce : "` : Leading text in the progress-bar

# Examples
```jldoctest
julia> pmapsum_elementwise(x->x^2,1:200)
2686700

julia> pmapsum_elementwise((x,y)-> x+y, (1:5,1:2))
45

julia> pmapsum_elementwise(x->x^2, Float64, 1:5)
55.0
```

See also: [`pmapreduce_commutative_elementwise`](@ref), [`pmapsum`](@ref)
"""
function pmapsum_elementwise(fmap::Function,T::Type,iterable,args...;
	showprogress::Bool = false, progressdesc = "Progress in pmapsum : ",
	kwargs...)

	pmapsum(plist->sum(x->fmap(x...,args...;kwargs...),plist),T,iterable,
		showprogress = showprogress, progressdesc = progressdesc)
end

function pmapsum_elementwise(fmap::Function,iterable,args...;
	showprogress::Bool = false, progressdesc = "Progress in pmapsum : ",
	kwargs...)

	pmapsum(plist->sum(x->fmap(x...,args...;kwargs...),plist),iterable,
		showprogress = showprogress, progressdesc = progressdesc)
end

"""
	pmapreduce(fmap, freduce, iterators, [mapargs...]; 
		<keyword arguments>, [mapkwargs...])

Evaluate a parallel mapreduce over the range spanned by 
the outer product of the iterators.

`iterators` must be a strictly-increasing range of integers, 
or a tuple of such ranges. 
The outer product of the ranges is split evenly across the workers. 
The function `fmap` receives a `ProductSplit` iterator as its first argument
that acts as a collection of tuples. One may index into a `ProductSplit` 
or iterate over one to access individual tuples of integers.

The reduction function `freduce` is expected to accept a collection of mapped values.
Note that this is different from the standard `mapreduce` operation in julia that 
expects a binary reduction operator. For example, `fmap` should be 
`sum` and not `+`. In case a binary operator `op` is to be passed, one may wrap it in
an anonymous function as `x->reduce(op,x)`, or as `x->op(x...)` in case the operator
accepts multiple arguments that are processed in pairs.

Arguments `mapargs` and keyword arguments `mapkwargs` — if provided — are 
passed on to the mapping function `fmap`.

	pmapreduce(fmap, Tmap, freduce, Treduce, iterators, [mapargs...]; 
		<keyword arguments>, [mapkwargs...])

The types `Tmap` and `Treduce` are the return types of the map and 
reduce operations respectively. The returned values will be coerced to 
the specified types if possible, throwing an error otherwise. 

# Keyword Arguments

- `showprogress::Bool = false` : Displays a progress-bar if set to true
- `progressdesc = "Progress in pmapreduce : "` : Leading text in the progress-bar

# Examples
```jldoctest
julia> pmapreduce(x->ones(2).*myid(), x->hcat(x...), 1:4)
2×4 Array{Float64,2}:
 2.0  3.0  4.0  5.0
 2.0  3.0  4.0  5.0

julia> pmapreduce(x->ones(2).*myid(), Vector{Int64}, x->hcat(x...), Matrix{Int64}, 1:4)
2×4 Array{Int64,2}:
 2  3  4  5
 2  3  4  5
```

See also: [`pmapreduce_commutative`](@ref), [`pmapsum`](@ref)
"""
function pmapreduce(fmap::Function,Tmap::Type,freduce::Function,Tred::Type,
	iterators::Tuple,args...;kwargs...)

	tree,branches = createbranchchannels(pval{Tmap},pval{Tred},
		iterators, SegmentedOrderedBinaryTree)
	pmapreduceworkers(fmap,freduce,iterators,tree,
		branches,Sorted(),args...;kwargs...)
end

function pmapreduce(fmap::Function,freduce::Function,iterators::Tuple,args...;
	kwargs...)

	pmapreduce(fmap,Any,freduce,Any,iterators,args...;kwargs...)
end

function pmapreduce(fmap::Function,Tmap::Type,freduce::Function,Tred::Type,
	iterable,args...;kwargs...)
	
	pmapreduce(fmap,Tmap,freduce,Tred,(iterable,),args...;kwargs...)
end

function pmapreduce(fmap::Function,freduce::Function,iterable,args...;kwargs...)
	pmapreduce(fmap,freduce,(iterable,),args...;kwargs...)
end

############################################################################################
# pmap in batches without reduction
############################################################################################

"""
	pmapbatch(f, iterators, [mapargs...]; 
		num_workers::Int = nworkersactive(iterators), [mapkwargs...])

Evaluate the function `f` in parallel, where each worker gets a 
part of the entire parameter space sequentially. The argument 
`iterators` needs to be a strictly-increasing range of intergers,
or a tuple of such ranges. The outer product of these ranges forms the 
entire range of parameters that is processed in batches on 
the workers.

Arguments `mapargs` and keyword arguments `mapkwargs` — if provided — are 
passed on to the function `f`. 
Additionally, the number of workers to be used may be specified using the 
keyword argument `num_workers`. In this case the first `num_workers` available
workers are used in the evaluation.

# Examples
```jldoctest
julia> pmapbatch(x->[(y,sum(y)) for y in x], (1:2,1:3))
4-element Array{Array{Tuple{Tuple{Int64,Int64},Int64},1},1}:
 [((1, 1), 2), ((2, 1), 3)]
 [((1, 2), 3), ((2, 2), 4)]
 [((1, 3), 4)]
 [((2, 3), 5)]

julia> pmapbatch(x->myid(),1:nworkers(), num_workers=2)
2-element Array{Int64,1}:
 2
 3
```

See also: [`pmapreduce`](@ref), [`pmapsum`](@ref)
"""
function pmapbatch(f::Function,iterators::Tuple,args...;
	num_workers = nworkersactive(iterators),kwargs...)

	procs_used = workersactive(iterators)

	if num_workers < length(procs_used)
		procs_used = procs_used[1:num_workers]
	end
	num_workers = length(procs_used)

	futures = Vector{Future}(undef,num_workers)
	@sync for (rank,p) in enumerate(procs_used)
		@async begin
			iterable_on_proc = ProductSplit(iterators,num_workers,rank)
			futures[rank] = @spawnat p f(iterable_on_proc,args...;kwargs...)
		end
	end
	vcat(asyncmap(fetch,futures)...)
end

function pmapbatch(f::Function,iterable,args...;kwargs...)
	pmapbatch(f,(iterable,),args...;kwargs...)
end

"""
	pmapbatch_elementwise(f, iterators, [mapargs...]; 
		num_workers::Int = nworkersactive(iterators), [mapkwargs...])

Evaluate the function `f` in parallel, where each worker gets a 
part of the entire parameter space sequentially. The argument 
`iterators` needs to be a strictly-increasing range of intergers,
or a tuple of such ranges. The outer product of these ranges forms the 
entire range of parameters that is processed elementwise by the function `f`.
Given `n` ranges in `iterators`, the function `f` will receive `n` integers 
at a time.

Arguments `mapargs` and keyword arguments `mapkwargs` — if provided — are 
passed on to the function `f`. 
Additionally, the number of workers to be used may be specified using the 
keyword argument `num_workers`. In this case the first `num_workers` available
workers are used in the evaluation.

# Examples
```jldoctest
julia> pmapbatch_elementwise((x,y)->x+y, (1:2,1:3))
4-element Array{Array{Int64,1},1}:
 [2, 3]
 [3, 4]
 [4]
 [5]

julia> pmapbatch_elementwise((x,y)->x+y, (1:2,1:3), num_workers=2)
2-element Array{Array{Int64,1},1}:
 [2, 3, 3]
 [4, 4, 5]
```

See also: [`pmapbatch`](@ref)
"""
function pmapbatch_elementwise(f::Function,iterators,args...;
	num_workers = nworkersactive(iterators), kwargs...)

	pmapbatch(plist->asyncmap(x->f(x...,args...;kwargs...),plist),
		iterators,num_workers=num_workers)
end