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
using .ClusterQueryUtils: procs_node
using .ClusterQueryUtils: workerpool_nodes
using .ClusterQueryUtils: workers_myhost
using .ClusterQueryUtils: workersactive
using .ClusterQueryUtils: maybetrimmedworkerpool

include("trees.jl")
include("reductionfunctions.jl")
include("mapreduce.jl")

end # module
