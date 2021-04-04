module ParallelUtilities

using Distributed

export pmapreduce
export pmapreduce_productsplit
export pmapbatch
export pmapbatch_productsplit
export workerrank

include("productsplit.jl")

include("clusterquery.jl")
using .ClusterQueryUtils: procs_node

include("trees.jl")
include("reductionfunctions.jl")
include("mapreduce.jl")

end # module
