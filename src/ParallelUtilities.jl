module ParallelUtilities
using ProgressMeter
using DataStructures
using Reexport
@reexport using Distributed

export  ProductSplit,
	ntasks,
	whichproc,
	procrange_recast,
	localindex,
	whichproc_localindex,
	extremadims,
	extrema_commonlastdim,
	nodenames,
	gethostnames,
	nprocs_node,
	pmapbatch,
	pmapbatch_elementwise,
	pmapsum,
	pmapsum_elementwise,
	pmapreduce,
	pmapreduce_commutative,
	pmapreduce_commutative_elementwise

include("errors.jl")
include("productsplit.jl")
include("utils.jl")
include("trees.jl")
include("mapreduce.jl")

end # module
