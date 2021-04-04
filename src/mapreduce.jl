# Store the rank with the value, necessary for collecting values in order
struct pval{T}
    rank :: Int
    errorstatus :: Bool
    value :: T
end
pval{T}(p::pval{T}) where {T} = p
pval{T}(p::pval) where {T} = pval{T}(p.rank, p.errorstatus, convert(T, p.value))

errorpval(rank) = pval(rank, true, nothing)

errorstatus(p::pval) = p.errorstatus

# Function to obtain the value of pval types
value(p::pval) = p.value
value(p::Any) = p

struct Product{I}
    iterators :: I
end

getiterators(p::Product) = p.iterators

Base.length(p::Product) = length(Iterators.product(p.iterators...))
Base.iterate(p::Product, st...) = iterate(Iterators.product(p.iterators...), st...)

function product(iter...)
    any(x -> x isa Product, iter) && throw(ArgumentError("the iterators should not be Products"))
    Product(iter)
end

struct Hold{T}
    iterators :: T
end

getiterators(h::Hold) = getiterators(h.iterators)

Base.length(h::Hold) = length(h.iterators)

check_knownsize(iterators::Tuple) = _check_knownsize(first(iterators)) & check_knownsize(Base.tail(iterators))
check_knownsize(::Tuple{}) = true
function _check_knownsize(iterator)
    itsz = Base.IteratorSize(iterator)
    itsz isa Base.HasLength || itsz isa Base.HasShape
end

function zipsplit(iterators::Tuple, np::Integer, p::Integer)
    check_knownsize(iterators)
    itzip = zip(iterators...)
    d,r = divrem(length(itzip), np)
    skipped_elements = d*(p-1) + min(r,p-1)
    lastind = d*p + min(r,p)
    elements_on_proc = lastind - skipped_elements
    Iterators.take(Iterators.drop(itzip, skipped_elements), elements_on_proc)
end

_split_iterators(iterators, np, p) = (zipsplit(iterators, np, p),)
function _split_iterators(iterators::Tuple{Hold{<:Product}}, np, p)
    it_hold = first(iterators)
    (ProductSplit(getiterators(it_hold), np, p), )
end

############################################################################################
# Local mapreduce
############################################################################################

struct NoSplat <: Function
    f :: Function
end
NoSplat(u::NoSplat) = u

_maybesplat(f) = Base.splat(f)
_maybesplat(f::NoSplat) = f

_mapreduce(f, op, iterators...; reducekw...) = mapreduce(f, op, iterators...; reducekw...)
function _mapreduce(fun::NoSplat, op, iterators...; reducekw...)
    mapval = fun.f(iterators...)
    reduce(op, (mapval,); reducekw...)
end

function mapreducenode(f, op, rank, pipe::BranchChannel, selfoutchannel, iterators...; reducekw...)
    # Evaluate the function
    # No communication with other nodes happens here
    try
        fmap = _maybesplat(f)
        if rank == 1
            res = _mapreduce(fmap, op, iterators...; reducekw...)
        else
            # init should only be used once on the first rank
            # remove it from the kwargs on other workers
            kwdict = Dict(reducekw)
            pop!(kwdict, :init, nothing)
            res = _mapreduce(fmap, op, iterators...; kwdict...)
        end
        val = pval(rank, false, res)
        put!(selfoutchannel, val)
    catch
        put!(selfoutchannel, errorpval(rank))
        rethrow()
    end
end

############################################################################################
# Reduction across workers
############################################################################################

abstract type ReductionNode end
struct TopTreeNode <: ReductionNode
    rank :: Int
end
struct SubTreeNode <: ReductionNode
    rank :: Int
end

_maybesort(op::Commutative, vals) = vals
_maybesort(op, vals) = sort!(vals, by = pv -> pv.rank)

function reducechannel(op, c, N; reducekw...)
    vals = [take!(c) for i = 1:N]
    vals = _maybesort(op, vals)
    v = [value(v) for v in vals]
    reduce(op, v; reducekw...)
end

seterrorflag(c, val) = put!(c, take!(c) | val)

function reducedvalue(op, node::SubTreeNode, pipe::BranchChannel, selfoutchannel; reducekw...)
    rank = node.rank

    N = nchildren(pipe) + 1
    err_ch = Channel{Bool}(1)
    put!(err_ch, false)

    self_pval = take!(selfoutchannel)
    if errorstatus(self_pval)
        return errorpval(rank)
    else
        put!(selfoutchannel, self_pval)
    end

    @sync for i = 1:nchildren(pipe)
        @async begin
            child_pval = take!(pipe.childrenchannel)
            if errorstatus(child_pval)
                seterrorflag(err_ch, true)
            else
                put!(selfoutchannel, child_pval)
                seterrorflag(err_ch, false)
            end
        end
    end

    take!(err_ch) && return errorpval(rank)

    redval = reducechannel(op, selfoutchannel, N; reducekw...)

    return pval(rank, false, redval)
end
function reducedvalue(op, node::TopTreeNode, pipe::BranchChannel, ::Any; reducekw...)
    rank = node.rank

    N = nchildren(pipe)
    c = Channel(N)
    err_ch = Channel{Bool}(1)
    put!(err_ch, false)

    @sync for i in 1:N
        @async begin
            child_pval = take!(pipe.childrenchannel)
            if errorstatus(child_pval)
                seterrorflag(err_ch, true)
            else
                put!(c, child_pval)
                seterrorflag(err_ch, false)
            end
        end
    end

    take!(err_ch) && return errorpval(rank)

    redval = reducechannel(op, c, N; reducekw...)

    return pval(rank, false, redval)
end

function reducenode(op, node::ReductionNode, pipe::BranchChannel, selfoutchannel = nothing; kwargs...)
    # This function that communicates with the parent and children
    rank = node.rank
    try
        kwdict = Dict(kwargs)
        pop!(kwdict, :init, nothing)
        res = reducedvalue(op, node, pipe, selfoutchannel; kwdict...)
        put!(pipe.parentchannel, res)
    catch
        put!(pipe.parentchannel, errorpval(rank))
        rethrow()
    finally
        GC.gc()
    end

    return nothing
end

function pmapreduceworkers(f, op, tree_branches, iterators; reducekw...)

    tree, branches = tree_branches

    nworkerstree = nworkers(tree)
    extrareducenodes = length(tree) - nworkerstree

    @sync for (ind, mypipe) in enumerate(branches)
        p = mypipe.p
        ind_reduced = ind - extrareducenodes
        rank = ind_reduced
        if ind_reduced > 0
            iterable_on_proc = _split_iterators(iterators, nworkerstree, rank)
            @spawnat p begin
                selfoutchannel = Channel(nchildren(mypipe) + 1)
                @sync begin
                    @async mapreducenode(f, op, rank, mypipe, selfoutchannel, iterable_on_proc...; reducekw...)
                    @async reducenode(op, SubTreeNode(rank), mypipe, selfoutchannel; reducekw...)
                end
            end
        else
            @spawnat p reducenode(op, TopTreeNode(rank), mypipe; reducekw...)
        end
    end

    tb = topbranch(tree, branches)
    value(take!(tb.parentchannel))
end

"""
    pmapreduce(f, op, [pool::AbstractWorkerPool], iterators...; reducekw...)

Evaluate a parallel `mapreduce` over the elements from `iterators`.
For multiple iterators, apply `f` elementwise.

The keyword arguments `reducekw` are passed on to the reduction.

See also: [`pmapreduce_productsplit`](@ref)
"""
function pmapreduce(f, op, pool::AbstractWorkerPool, iterators...; reducekw...)
    N = length(zip(iterators...))

    if N <= 1 || nworkers(pool) == 1
        iterable_on_proc = _split_iterators(iterators, 1, 1)
        fmap = _maybesplat(f)
        if nprocs() == 1 # no workers added
            return _mapreduce(fmap, op, iterable_on_proc...; reducekw...)
        else # one worker or single-valued iterator
            return @fetchfrom workers(pool)[1] _mapreduce(fmap, op, iterable_on_proc...; reducekw...)
        end
    end

    tree_branches = createbranchchannels(pool, N)
    pmapreduceworkers(f, op, tree_branches, iterators; reducekw...)
end

function pmapreduce(f, op, iterators...; reducekw...)
    N = length(zip(iterators...))
    pool = maybetrimmedworkerpool(workers(), N)
    pmapreduce(f, op, pool, iterators...; reducekw...)
end

"""
    pmapreduce_productsplit(f, op, [pool::AbstractWorkerPool], iterators...; reducekw...)

Evaluate a parallel mapreduce over the outer product of elements from `iterators`.
The product of `iterators` is split over the workers available, and each worker is assigned a section
of the product. The function `f` should accept a single argument that is a collection of `Tuple`s.

The keyword arguments `reducekw` are passed on to the reduction.

See also: [`pmapreduce`](@ref)
"""
pmapreduce_productsplit(f, op, pool::AbstractWorkerPool, iterators...; reducekw...) =
    pmapreduce(NoSplat(f), op, pool, Hold(product(iterators...)); reducekw...)

function pmapreduce_productsplit(f, op, iterators...; reducekw...)
    N = length(product(iterators...))
    pool = maybetrimmedworkerpool(workers(), N)
    pmapreduce_productsplit(f, op, pool, iterators...; reducekw...)
end

"""
    pmapbatch(f, [pool::AbstractWorkerPool], iterators...)

Carry out a `pmap` with the `iterators` divided evenly among the available workers.

See also: [`pmapreduce`](@ref)
"""
function pmapbatch(f, pool::AbstractWorkerPool, iterators...)
    pmapreduce((x...) -> [f(x...)], vcat, pool, iterators...)
end

function pmapbatch(f, iterators...)
    N = length(zip(iterators...))
    pool = maybetrimmedworkerpool(workers(), N)
    pmapbatch(f, pool, iterators...)
end

"""
    pmapbatch_productsplit(f, [pool::AbstractWorkerPool], iterators...)

Carry out a `pmap` with the outer product of `iterators` divided evenly among the available workers.
The function `f` must accept a collection of `Tuple`s.

See also: [`pmapbatch`](@ref), [`pmapreduce_productsplit`](@ref)
"""
function pmapbatch_productsplit(f, pool::AbstractWorkerPool, iterators...)
    pmapreduce_productsplit(x -> [f(x)], vcat, pool, iterators...)
end

function pmapbatch_productsplit(f, iterators...)
    N = length(product(iterators...))
    pool = maybetrimmedworkerpool(workers(), N)
    pmapbatch_productsplit(f, pool, iterators...)
end
