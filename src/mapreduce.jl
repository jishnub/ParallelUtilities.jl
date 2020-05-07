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

function indicatemapprogress!(::Nothing) end
function indicatemapprogress!(progress::RemoteChannel)
	put!(progress,(true,false))
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
	catch
		put!(pipe.selfchannels.err,true)
		rethrow()
	finally
		indicatemapprogress!(progress)
	end
end

############################################################################################
# Reduction
############################################################################################

abstract type Ordering end
struct Sorted <: Ordering end
struct Unsorted <: Ordering end

function reducedvalue(freduce::Function,rank,
	pipe::BranchChannel{Tmap,Tred},::Unsorted) where {Tmap,Tred}

	self = take!(pipe.selfchannels.out) :: Tmap

	N = nchildren(pipe)
	res = if N > 0
			reducechildren = freduce(take!(pipe.childrenchannels.out)::Tred for i=1:N)::Tred
			freduce((reducechildren,self)) :: Tred
		else
			freduce((self,)) :: Tred
		end
end

function reducedvalue(freduce::Function,rank,
	pipe::BranchChannel{Tmap,Tred},::Sorted) where {Tmap,Tred}

	N = nchildren(pipe)
	leftchild = N > 0
	vals = Vector{Tred}(undef,N+1)
	@sync begin
		@async begin
			selfval = take!(pipe.selfchannels.out)::Tmap
			selfvalred = freduce((value(selfval),))
			ind = 1 + leftchild
			v = pval(rank,selfvalred)
			vals[ind] = v
		end
		@async for i=2:N+1
			pv = take!(pipe.childrenchannels.out) :: Tred
			shift = pv.rank > rank ? 1 : -1
			ind = shift + leftchild + 1
			vals[ind] = pv
		end
	end

	Tred(rank,freduce(value(v) for v in vals))
end

function indicatereduceprogress!(::Nothing) end
function indicatereduceprogress!(progress::RemoteChannel)
	put!(progress,(false,true))
end

function reduceTreeNode(freduce::Function,rank,pipe::BranchChannel{Tmap,Tred},
	ifsort::Ordering,progress::Union{Nothing,RemoteChannel}) where {Tmap,Tred}
	# This function that communicates with the parent and children

	# Start by checking if there is any error locally in the map,
	# and if there's none then check if there are any errors on the children
	anyerr = take!(pipe.selfchannels.err) || 
				any(take!(pipe.childrenchannels.err) for i=1:nchildren(pipe))

	# Evaluate the reduction only if there's no error
	# In either case push the error flag to the parent
	if !anyerr
		try
			res = reducedvalue(freduce,rank,pipe,ifsort) :: Tred
			put!(pipe.parentchannels.out,res)
			put!(pipe.parentchannels.err,false)
		catch e
			put!(pipe.parentchannels.err,true)
			rethrow()
		finally
			indicatereduceprogress!(progress)
		end
	else
		put!(pipe.parentchannels.err,true)
		indicatereduceprogress!(progress)
	end

	finalize(pipe)
end

function return_unless_error(r::RemoteChannelContainer)
	anyerror = take!(r.err)
	if !anyerror
		return value(take!(r.out))
	end
end

@inline function return_unless_error(b::BranchChannel)
	return_unless_error(b.parentchannels)
end

function pmapreduceworkers(fmap::Function,freduce::Function,iterators::Tuple,
	tree,branches,ord::Ordering,args...;
	showprogress::Bool = false, progressdesc = "Progress in pmapreduce : ",
	kwargs...)

	num_workers_active = nworkersactive(iterators)

	nmap,nred = 0,0
	progresschannel = RemoteChannel(()->Channel{Tuple{Bool,Bool}}(
						ifelse(showprogress,2num_workers_active,0)))
	progressbar = Progress(2num_workers_active,1,progressdesc)

	# Run the function on each processor and compute the reduction at each node
	@sync begin
		for (rank,mypipe) in enumerate(branches)
			@async begin
				p = mypipe.p
				iterable_on_proc = evenlyscatterproduct(iterators,num_workers_active,rank)

				@spawnat p mapTreeNode(fmap,iterable_on_proc,rank,mypipe,
					ifelse(showprogress,progresschannel,nothing),
					args...;kwargs...)
				@spawnat p reduceTreeNode(freduce,rank,mypipe,ord,
					ifelse(showprogress,progresschannel,nothing))
			end
		end
		
		if showprogress
			for i = 1:2num_workers_active
				mapdone,reddone = take!(progresschannel)
				nmap += mapdone
				nred += reddone

				next!(progressbar;showvalues=[(:map,nmap),(:reduce,nred)])
			end
			finish!(progressbar)
		end
	end

	return_unless_error(topnode(tree,branches))
end

# This function does not sort the values, so it might be faster
function pmapreduce_commutative(fmap::Function,::Type{Tmap},
	freduce::Function,::Type{Tred},iterators::Tuple,args...;
	kwargs...) where {Tmap,Tred}
	
	tree,branches = createbranchchannels(Tmap,Tred,iterators,
		SequentialBinaryTree)
	pmapreduceworkers(fmap,freduce,iterators,tree,
		branches,Unsorted(),args...;kwargs...)
end

function pmapreduce_commutative(fmap::Function,freduce::Function,
	iterators::Tuple,args...;kwargs...)

	pmapreduce_commutative(fmap,Any,freduce,Any,iterators,args...;kwargs...)
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
	freduce::Function,::Type{Tred},iterable,args...;
	showprogress::Bool = false, progressdesc = "Progress in pmapreduce : ",
	kwargs...) where {Tmap,Tred}
	
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

function pmapsum(fmap::Function,::Type{T},iterable,args...;kwargs...) where {T}
	pmapreduce_commutative(fmap,T,sum,T,iterable,args...;
		progressdesc = "Progress in pmapsum : ",kwargs...)
end

function pmapsum(fmap::Function,iterable,args...;kwargs...)
	pmapreduce_commutative(fmap,sum,iterable,args...;
		progressdesc = "Progress in pmapsum : ",kwargs...)
end

function pmapsum_elementwise(fmap::Function,::Type{T},iterable,args...;
	showprogress::Bool = false, progressdesc = "Progress in pmapsum : ",
	kwargs...) where {T}

	pmapsum(plist->sum(x->fmap(x...,args...;kwargs...),plist),T,iterable,
		showprogress = showprogress, progressdesc = progressdesc)
end

function pmapsum_elementwise(fmap::Function,iterable,args...;
	showprogress::Bool = false, progressdesc = "Progress in pmapsum : ",
	kwargs...)

	pmapsum(plist->sum(x->fmap(x...,args...;kwargs...),plist),iterable,
		showprogress = showprogress, progressdesc = progressdesc)
end

function pmapreduce(fmap::Function,::Type{Tmap},freduce::Function,::Type{Tred},
	iterators::Tuple,args...;kwargs...) where {Tmap,Tred}

	tree,branches = createbranchchannels(pval{Tmap},pval{Tred},
		iterators,OrderedBinaryTree)
	pmapreduceworkers(fmap,freduce,iterators,tree,
		branches,Sorted(),args...;kwargs...)
end

function pmapreduce(fmap::Function,freduce::Function,iterators::Tuple,args...;
	kwargs...)

	pmapreduce(fmap,Any,freduce,Any,iterators,args...;kwargs...)
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