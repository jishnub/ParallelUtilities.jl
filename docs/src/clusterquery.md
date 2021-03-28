```@meta
DocTestSetup  = quote
    using ParallelUtilities
    using ParallelUtilities.ClusterQueryUtils
end
```

# Cluster Query Utilities

These are a collection of helper functions that are used in `ParallelUtilities`, but may be used independently as well to obtain information about the cluster on which codes are being run.

To use these functions run

```jldoctest cqu
julia> using ParallelUtilities.ClusterQueryUtils
```

The functions defined in this module are:

```@autodocs
Modules = [ParallelUtilities.ClusterQueryUtils]
```
