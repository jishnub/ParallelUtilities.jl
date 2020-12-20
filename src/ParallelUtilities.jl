module ParallelUtilities
using ProgressMeter
using Reexport
using OffsetArrays

@reexport using Distributed

export ProductSplit,
    ntasks,
    whichproc,
    procrange_recast,
    localindex,
    whichproc_localindex,
    extremadims,
    extrema_commonlastdim,
    pmapbatch,
    pmapbatch_elementwise,
    pmapsum,
    pmapsum_elementwise,
    pmapreduce,
    pmapreduce_commutative,
    pmapreduce_commutative_elementwise

include("errors.jl")
include("productsplit.jl")

include("clusterquery.jl")
@reexport using .ClusterQueryUtils

include("utils.jl")
include("trees.jl")
include("mapreduce.jl")
include("reductionfunctions.jl")

end # module
