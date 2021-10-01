module ParallelUtilities

using Distributed
using SplittablesBase

export pmapreduce
export pmapreduce_productsplit
export pmapbatch
export pmapbatch_productsplit
export workerrank

include("productsplit.jl")

include("clusterquery.jl")
using .ClusterQueryUtils

include("trees.jl")
include("reductionfunctions.jl")
include("mapreduce.jl")

end # module
