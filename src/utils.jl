"""
    nworkersactive(iterators::Tuple)

Number of workers required to contain the outer product of the iterators.
"""
function nworkersactive(iterators::Tuple)
    min(nworkers(), prod(length, iterators))
end
nworkersactive(ps::AbstractConstrainedProduct) = nworkersactive(getiterators(ps))
nworkersactive(args::AbstractRange...) = nworkersactive(args)

"""
    workersactive(iterators::Tuple)

Workers required to split the outer product of the iterators. 
If `prod(length, iterators) < nworkers()` then the first `prod(length, iterators)`
workers are chosen.
"""
workersactive(iterators::Tuple) = workers()[1:nworkersactive(iterators)]
workersactive(ps::AbstractConstrainedProduct) = workersactive(getiterators(ps))
workersactive(args::AbstractRange...) = workersactive(args)