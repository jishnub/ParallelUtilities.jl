var documenterSearchIndex = {"docs":
[{"location":"","page":"API","title":"API","text":"CurrentModule = ParallelUtilities","category":"page"},{"location":"#ParallelUtilities.jl","page":"API","title":"ParallelUtilities.jl","text":"","category":"section"},{"location":"","page":"API","title":"API","text":"Modules = [ParallelUtilities, ParallelUtilities.ClusterQueryUtils]","category":"page"},{"location":"#ParallelUtilities.AbstractConstrainedProduct","page":"API","title":"ParallelUtilities.AbstractConstrainedProduct","text":"AbstractConstrainedProduct{T,N}\n\nSupertype of ProductSplit and ProductSection.\n\n\n\n\n\n","category":"type"},{"location":"#ParallelUtilities.ProductSection","page":"API","title":"ParallelUtilities.ProductSection","text":"ProductSection{T,N,Q}\n\nIterator that loops over a specified section of the  outer product of the ranges provided in  reverse-lexicographic order. The ranges need to be strictly increasing. Given N ranges,  each element returned by the iterator will be  a tuple of length N with one element from each range.\n\nSee also: ProductSplit\n\n\n\n\n\n","category":"type"},{"location":"#ParallelUtilities.ProductSection-Tuple{Tuple{Vararg{AbstractRange,N} where N},AbstractUnitRange}","page":"API","title":"ParallelUtilities.ProductSection","text":"ProductSection(iterators::Tuple{Vararg{AbstractRange}}, inds::AbstractUnitRange)\n\nConstruct a ProductSection iterator that represents a 1D view of the outer product of the ranges provided in iterators, with the range of indices in the view being specified by inds.\n\nExamples\n\njulia> p = ParallelUtilities.ProductSection((1:3,4:6), 5:8);\n\njulia> collect(p)\n4-element Array{Tuple{Int64,Int64},1}:\n (2, 5)\n (3, 5)\n (1, 6)\n (2, 6)\n\njulia> collect(p) == collect(Iterators.product(1:3, 4:6))[5:8]\ntrue\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.ProductSplit","page":"API","title":"ParallelUtilities.ProductSplit","text":"ProductSplit{T,N,Q}\n\nIterator that loops over the outer product of ranges in  reverse-lexicographic order. The ranges need to be strictly increasing. Given N ranges,  each element returned by the iterator will be  a tuple of length N with one element from each range.\n\nSee also: ProductSection\n\n\n\n\n\n","category":"type"},{"location":"#ParallelUtilities.ProductSplit-Tuple{Tuple{Vararg{AbstractRange,N} where N},Integer,Integer}","page":"API","title":"ParallelUtilities.ProductSplit","text":"ProductSplit(iterators::Tuple{Vararg{AbstractRange}}, np::Integer, p::Integer)\n\nConstruct a ProductSplit iterator that represents the outer product  of the iterators split over np workers, with this instance reprsenting  the values on the p-th worker.\n\nExamples\n\njulia> ProductSplit((1:2,4:5), 2, 1) |> collect\n2-element Array{Tuple{Int64,Int64},1}:\n (1, 4)\n (2, 4)\n\njulia> ProductSplit((1:2,4:5), 2, 2) |> collect\n2-element Array{Tuple{Int64,Int64},1}:\n (1, 5)\n (2, 5)\n\n\n\n\n\n","category":"method"},{"location":"#Base.extrema-Tuple{ParallelUtilities.AbstractConstrainedProduct}","page":"API","title":"Base.extrema","text":"extrema(ps::AbstractConstrainedProduct; dim::Integer)\n\nCompute the extrema of the section of the range number dim contained in ps.\n\nExamples\n\njulia> ps = ProductSplit((1:2, 4:5), 2, 1);\n\njulia> collect(ps)\n2-element Array{Tuple{Int64,Int64},1}:\n (1, 4)\n (2, 4)\n\njulia> extrema(ps, dim = 1)\n(1, 2)\n\njulia> extrema(ps, dim = 2)\n(4, 4)\n\n\n\n\n\n","category":"method"},{"location":"#Base.maximum-Tuple{ParallelUtilities.AbstractConstrainedProduct}","page":"API","title":"Base.maximum","text":"maximum(ps::AbstractConstrainedProduct; dim::Integer)\n\nCompute the maximum value of the section of the range number dim contained in ps.\n\nExamples\n\njulia> ps = ProductSplit((1:2,4:5),2,1);\n\njulia> collect(ps)\n2-element Array{Tuple{Int64,Int64},1}:\n (1, 4)\n (2, 4)\n\njulia> maximum(ps, dim = 1)\n2\n\njulia> maximum(ps, dim = 2)\n4\n\n\n\n\n\n","category":"method"},{"location":"#Base.minimum-Tuple{ParallelUtilities.AbstractConstrainedProduct}","page":"API","title":"Base.minimum","text":"minimum(ps::AbstractConstrainedProduct; dim::Integer)\n\nCompute the minimum value of the section of the range number dim contained in ps.\n\nExamples\n\njulia> ps = ProductSplit((1:2, 4:5), 2, 1);\n\njulia> collect(ps)\n2-element Array{Tuple{Int64,Int64},1}:\n (1, 4)\n (2, 4)\n\njulia> minimum(ps, dim = 1)\n1\n\njulia> minimum(ps, dim = 2)\n4\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.childindex-Tuple{ParallelUtilities.AbstractConstrainedProduct,Any}","page":"API","title":"ParallelUtilities.childindex","text":"childindex(ps::AbstractConstrainedProduct, ind)\n\nReturn a tuple containing the indices of the individual iterators  corresponding to the element that is present at index ind in the  outer product of the iterators.\n\nExamples\n\njulia> ps = ProductSplit((1:5, 2:4, 1:3), 7, 1);\n\njulia> ParallelUtilities.childindex(ps, 6)\n(1, 2, 1)\n\njulia> v = collect(Iterators.product(1:5, 2:4, 1:3));\n\njulia> getindex.(ps.iterators, ParallelUtilities.childindex(ps,6)) == v[6]\ntrue\n\nSee also: childindexshifted\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.childindexshifted-Tuple{ParallelUtilities.AbstractConstrainedProduct,Any}","page":"API","title":"ParallelUtilities.childindexshifted","text":"childindexshifted(ps::AbstractConstrainedProduct, ind)\n\nReturn a tuple containing the indices in the individual iterators  given an index of a AbstractConstrainedProduct.\n\nExamples\n\njulia> ps = ProductSplit((1:5, 2:4, 1:3), 7, 3);\n\njulia> cinds = ParallelUtilities.childindexshifted(ps, 3)\n(2, 1, 2)\n\njulia> getindex.(ps.iterators, cinds) == ps[3]\ntrue\n\nSee also: childindex\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.dropleading-Tuple{ParallelUtilities.AbstractConstrainedProduct}","page":"API","title":"ParallelUtilities.dropleading","text":"dropleading(ps::AbstractConstrainedProduct)\n\nReturn a ProductSection leaving out the first iterator contained in ps.  The range of values of the remaining iterators in the  resulting ProductSection will be the same as in ps.\n\nExamples\n\njulia> ps = ProductSplit((1:5, 2:4, 1:3), 7, 3);\n\njulia> collect(ps)\n7-element Array{Tuple{Int64,Int64,Int64},1}:\n (5, 4, 1)\n (1, 2, 2)\n (2, 2, 2)\n (3, 2, 2)\n (4, 2, 2)\n (5, 2, 2)\n (1, 3, 2)\n\njulia> ParallelUtilities.dropleading(ps) |> collect\n3-element Array{Tuple{Int64,Int64},1}:\n (4, 1)\n (2, 2)\n (3, 2)\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.extrema_commonlastdim-Union{Tuple{ParallelUtilities.AbstractConstrainedProduct{var\"#s27\",N} where var\"#s27\"}, Tuple{N}} where N","page":"API","title":"ParallelUtilities.extrema_commonlastdim","text":"extrema_commonlastdim(ps::AbstractConstrainedProduct)\n\nReturn the reverse-lexicographic extrema of values taken from  ranges contained in ps, where the pairs of ranges are constructed  by concatenating the ranges along each dimension with the last one.\n\nFor two ranges this simply returns ([first(ps)], [last(ps)]).\n\nExamples\n\njulia> ps = ProductSplit((1:3,4:7,2:7), 10, 2);\n\njulia> collect(ps)\n8-element Array{Tuple{Int64,Int64,Int64},1}:\n (3, 6, 2)\n (1, 7, 2)\n (2, 7, 2)\n (3, 7, 2)\n (1, 4, 3)\n (2, 4, 3)\n (3, 4, 3)\n (1, 5, 3)\n\njulia> extrema_commonlastdim(ps)\n([(1, 2), (6, 2)], [(3, 3), (5, 3)])\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.extremadims-Tuple{ParallelUtilities.AbstractConstrainedProduct}","page":"API","title":"ParallelUtilities.extremadims","text":"extremadims(ps::AbstractConstrainedProduct)\n\nCompute the extrema of the sections of all the ranges contained in ps.  Functionally this is equivalent to \n\nmap(i -> extrema(ps, dim = i), 1:ndims(ps))\n\nbut it is implemented more efficiently. \n\nReturns a Tuple containing the (min, max) pairs along each  dimension, such that the i-th index of the result contains the extrema along the section of the i-th range contained locally.\n\nExamples\n\njulia> ps = ProductSplit((1:2, 4:5), 2, 1);\n\njulia> collect(ps)\n2-element Array{Tuple{Int64,Int64},1}:\n (1, 4)\n (2, 4)\n\njulia> extremadims(ps)\n((1, 2), (4, 4))\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.indexinproduct-Union{Tuple{N}, Tuple{Tuple{Vararg{AbstractRange,N}},Tuple{Vararg{Any,N}}}} where N","page":"API","title":"ParallelUtilities.indexinproduct","text":"indexinproduct(iterators::NTuple{N,AbstractRange}, val::NTuple{N,Any}) where {N}\n\nReturn the index of val in the outer product of iterators,  where iterators is a Tuple of increasing AbstractRanges.  Return nothing if val is not present.\n\nExamples\n\njulia> iterators = (1:4, 1:3, 3:5);\n\njulia> val = (2, 2, 4);\n\njulia> ind = ParallelUtilities.indexinproduct(iterators,val)\n18\n\njulia> collect(Iterators.product(iterators...))[ind] == val\ntrue\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.localindex-Union{Tuple{T}, Tuple{ParallelUtilities.AbstractConstrainedProduct{T,N} where N,T}} where T","page":"API","title":"ParallelUtilities.localindex","text":"localindex(ps::AbstractConstrainedProduct{T}, val::T) where {T}\n\nReturn the index of val in ps. Return nothing if the value is not found.\n\nExamples\n\njulia> ps = ProductSplit((1:3, 4:5:20), 3, 2);\n\njulia> collect(ps)\n4-element Array{Tuple{Int64,Int64},1}:\n (2, 9)\n (3, 9)\n (1, 14)\n (2, 14)\n\njulia> localindex(ps, (3,9))\n2\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.nelements-Tuple{ParallelUtilities.AbstractConstrainedProduct,Integer}","page":"API","title":"ParallelUtilities.nelements","text":"nelements(ps::AbstractConstrainedProduct; dim::Integer)\n\nCompute the number of unique values in the section of the dim-th range contained in ps.\n\nExamples\n\njulia> ps = ProductSplit((1:5, 2:4, 1:3), 7, 3);\n\njulia> collect(ps)\n7-element Array{Tuple{Int64,Int64,Int64},1}:\n (5, 4, 1)\n (1, 2, 2)\n (2, 2, 2)\n (3, 2, 2)\n (4, 2, 2)\n (5, 2, 2)\n (1, 3, 2)\n\njulia> ParallelUtilities.nelements(ps, dim = 1)\n5\n\njulia> ParallelUtilities.nelements(ps, dim = 2)\n3\n\njulia> ParallelUtilities.nelements(ps, dim = 3)\n2\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.ntasks","page":"API","title":"ParallelUtilities.ntasks","text":"ntasks(iterators::Tuple)\n\nThe total number of elements in the outer product of the ranges contained in  iterators, equal to prod(length, iterators)\n\n\n\n\n\n","category":"function"},{"location":"#ParallelUtilities.nworkersactive-Tuple{Tuple}","page":"API","title":"ParallelUtilities.nworkersactive","text":"nworkersactive(iterators::Tuple)\n\nNumber of workers required to contain the outer product of the iterators.\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.pmapbatch-Tuple{Function,Tuple,Vararg{Any,N} where N}","page":"API","title":"ParallelUtilities.pmapbatch","text":"pmapbatch(f, iterators, [mapargs...]; \n    [num_workers::Int = nworkersactive(iterators)], [mapkwargs...])\n\nEvaluate the function f in parallel, where each worker gets a  part of the entire parameter space sequentially. The argument  iterators needs to be a strictly-increasing range, or a tuple of such ranges. The outer product of these ranges forms the  entire range of parameters that is processed in batches on  the workers. Arguments mapargs and keyword arguments mapkwargs — if provided — are  passed on to the function f. \n\nAdditionally, the number of workers to be used may be specified using the  keyword argument num_workers. In this case the first num_workers available workers are used in the evaluation.\n\npmapbatch(f, T::Type, iterators, [mapargs...];\n    [num_workers::Int = nworkersactive(iterators)], [mapkwargs...])\n\nEvaluate f in parallel, and convert the returned value to type T.  The method is type stable if T is concrete. Values returned by f will be type-coerced if possible, and an error will be raised otherwise.\n\nSee also: pmapreduce, pmapsum\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.pmapbatch_elementwise-Tuple{Function,Any,Vararg{Any,N} where N}","page":"API","title":"ParallelUtilities.pmapbatch_elementwise","text":"pmapbatch_elementwise(f, iterators, [mapargs...]; \n    [num_workers::Int = nworkersactive(iterators)], [mapkwargs...])\n\nEvaluate the function f in parallel, where each worker gets a  part of the entire parameter space sequentially. The argument  iterators needs to be a strictly-increasing range of intergers, or a tuple of such ranges. The outer product of these ranges forms the  entire range of parameters that is processed elementwise by the function f. The individual tuples are splatted and passed as arguments to f. Given n ranges in iterators, the function f will receive n values  at a time.\n\nArguments mapargs and keyword arguments mapkwargs — if provided — are  passed on to the function f.  Additionally, the number of workers to be used may be specified using the  keyword argument num_workers. In this case the first num_workers available workers are used in the evaluation.\n\nSee also: pmapbatch\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.pmapreduce-Tuple{Function,Type,Function,Type,Tuple,Vararg{Any,N} where N}","page":"API","title":"ParallelUtilities.pmapreduce","text":"pmapreduce(fmap, freduce, iterators, [mapargs...]; \n    <keyword arguments>, [mapkwargs...])\n\nEvaluate a parallel mapreduce over the range spanned by  the outer product of the iterators.\n\niterators must be a strictly-increasing range of integers,  or a tuple of such ranges.  The outer product of the ranges is split evenly across the workers.  The function fmap receives a ProductSplit iterator as its first argument that acts as a collection of tuples. One may index into a ProductSplit  or iterate over one to access individual tuples of integers.\n\nThe reduction function freduce is expected to accept a collection of mapped values. Note that this is different from the standard mapreduce operation in julia that  expects a binary reduction operator. For example, fmap should be  sum and not +. In case a binary operator op is to be used in the reduction, one may pass it  as Base.splat(op) or wrap it in an anonymous function as x -> op(x...).\n\nArguments mapargs and keyword arguments mapkwargs — if provided — are  passed on to the mapping function fmap.\n\npmapreduce(fmap, Tmap, freduce, Treduce, iterators, [mapargs...]; \n    <keyword arguments>, [mapkwargs...])\n\nThe types Tmap and Treduce are the return types of the map and  reduce operations respectively. The returned values will be coerced to  the specified types if possible, throwing an error otherwise. \n\nKeyword Arguments\n\nshowprogress::Bool = false : Displays a progress-bar if set to true\nprogressdesc = \"Progress in pmapreduce : \" : Leading text in the progress-bar\n\nSee also: pmapreduce_commutative, pmapsum\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.pmapreduce_commutative-Tuple{Function,Type,Function,Type,Tuple,Vararg{Any,N} where N}","page":"API","title":"ParallelUtilities.pmapreduce_commutative","text":"pmapreduce_commutative(fmap, freduce, iterators, [mapargs...]; \n    <keyword arguments>, [mapkwargs...])\n\nEvaluate a parallel mapreduce over the range spanned by  the outer product of the iterators. The operation is assumed to be commutative, results obtained may be incorrect otherwise.\n\nThe argument  iterators must be a strictly-increasing range of integers,  or a tuple of such ranges.  The outer product of the ranges is split evenly across the workers.  The function fmap receives a ProductSplit iterator as its first argument that acts as a collection of tuples. One may index into a ProductSplit  or iterate over one to access individual tuples of integers.\n\nThe reduction function freduce is expected to accept a collection of mapped values. Note that this is different from the standard mapreduce operation in julia that  expects a binary reduction operator. For example, freduce should be  sum and not +. In case a binary operator op is to be used in the reduction, one may pass it  as Base.splat(op) or wrap it in an anonymous function as x -> op(x...).\n\nArguments mapargs and keyword arguments mapkwargs — if provided — are  passed on to the mapping function fmap.\n\npmapreduce_commutative(fmap, Tmap, freduce, Treduce, iterators, [mapargs...]; \n    <keyword arguments>, [mapkwargs...])\n\nThe types Tmap and Treduce are the return types of the map and  reduce operations respectively. The returned values will be coerced to  the specified types if possible, throwing an error otherwise. \n\nKeyword Arguments\n\nshowprogress::Bool = false : Displays a progress-bar if set to true\nprogressdesc = \"Progress in pmapreduce : \" : Leading text in the progress-bar\n\nSee also: pmapreduce_commutative_elementwise, pmapreduce, pmapsum\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.pmapreduce_commutative_elementwise-Tuple{Function,Type,Function,Type,Any,Vararg{Any,N} where N}","page":"API","title":"ParallelUtilities.pmapreduce_commutative_elementwise","text":"pmapreduce_commutative_elementwise(fmap, freduce, iterators, [mapargs...]; \n    <keyword arguments>, [mapkwargs...])\n\nEvaluate a parallel mapreduce over the range spanned by  the outer product of the iterators.  The argument iterators must be a strictly-increasing range of integers,  or a tuple of such ranges. The map is evaluated elementwise  over the entire range of parameters. The reduction is assumed to be commutative,  results obtained may be incorrect otherwise.\n\nThe reduction function freduce is expected to accept a collection of mapped values. Note that this is different from the standard mapreduce operation in julia that  expects a binary reduction operator. For example, freduce should be  sum and not +. In case a binary operator op is to be used in the reduction, one may pass it  as Base.splat(op) or wrap it in an anonymous function as x -> op(x...).\n\nArguments mapargs and keyword arguments mapkwargs — if provided — are  passed on to the mapping function fmap.\n\npmapreduce_commutative_elementwise(fmap, Tmap, freduce, Treduce, iterators, [mapargs...]; \n    <keyword arguments>, [mapkwargs...])\n\nThe types Tmap and Treduce are the return types of the map and  reduce operations respectively. The returned values will be coerced to  the specified types if possible, throwing an error otherwise. \n\nKeyword Arguments\n\nshowprogress::Bool = false : Displays a progress-bar if set to true\nprogressdesc = \"Progress in pmapreduce : \" : Leading text in the progress-bar\n\nSee also: pmapreduce_commutative\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.pmapsum-Tuple{Function,Type,Any,Vararg{Any,N} where N}","page":"API","title":"ParallelUtilities.pmapsum","text":"pmapsum(fmap, iterators, [mapargs...]; \n    <keyword arguments>, [mapkwargs...])\n\nEvaluate a parallel mapreduce over the range spanned by  the outer product of the iterators, where the reduction operation is a sum.\n\nThe argument iterators must be a strictly-increasing range of integers,  or a tuple of such ranges. The outer product of the ranges is split evenly across the workers.  The function fmap receives a ProductSplit iterator as its first argument that acts as a collection of tuples. One may index into a ProductSplit  or iterate over one to access individual tuples of integers.\n\nArguments mapargs and keyword arguments mapkwargs — if provided — are  passed on to the mapping function fmap.\n\npmapsum(fmap, Tmap, iterators, [mapargs...]; \n    <keyword arguments>, [mapkwargs...])\n\nThe types Tmap is the return types of the map.  The returned values will be coerced to  the specified type if possible, throwing an error otherwise.\n\nKeyword Arguments\n\nshowprogress::Bool = false : Displays a progress-bar if set to true\nprogressdesc = \"Progress in pmapsum : \" : Leading text in the progress-bar\n\nSee also: pmapreduce, pmapreduce_commutative\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.pmapsum_elementwise-Tuple{Function,Type,Any,Vararg{Any,N} where N}","page":"API","title":"ParallelUtilities.pmapsum_elementwise","text":"pmapsum_elementwise(fmap, iterators, [mapargs...]; \n    <keyword arguments>, [mapkwargs...])\n\nEvaluate a parallel mapreduce over the range spanned by  the outer product of the iterators, where the reduction operation is a sum.  The argument iterators must be a strictly-increasing range of integers,  or a tuple of such ranges. The map is evaluated elementwise  over the entire range of parameters.\n\nArguments mapargs and keyword arguments mapkwargs — if provided — are  passed on to the mapping function fmap.\n\npmapsum_elementwise(fmap, Tmap, iterators, [mapargs...]; \n    <keyword arguments>, [mapkwargs...])\n\nThe type Tmap is the return type of the map.  The returned values will be coerced to  the specified type if possible, throwing an error otherwise.\n\nKeyword Arguments\n\nshowprogress::Bool = false : Displays a progress-bar if set to true\nprogressdesc = \"Progress in pmapreduce : \" : Leading text in the progress-bar\n\nSee also: pmapreduce_commutative_elementwise, pmapsum\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.procrange_recast-Tuple{ParallelUtilities.AbstractConstrainedProduct,Integer}","page":"API","title":"ParallelUtilities.procrange_recast","text":"procrange_recast(ps::AbstractConstrainedProduct, np_new::Integer)\n\nReturn the range of processor ranks that would contain the values in ps if the  iterators used to construct ps were split across np_new processes.\n\nExamples\n\njulia> iters = (1:10, 4:6, 1:4);\n\njulia> ps = ProductSplit(iters, 5, 2); # split across 5 processes initially\n\njulia> procrange_recast(ps, 10) # If `iters` were spread across 10 processes\n3:4\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.procrange_recast-Tuple{Tuple,ParallelUtilities.AbstractConstrainedProduct,Integer}","page":"API","title":"ParallelUtilities.procrange_recast","text":"procrange_recast(iterators::Tuple, ps::ProductSplit, np_new::Integer)\n\nReturn the range of processor ranks that would contain the values in ps if  the outer produce of the ranges in iterators is split across np_new  workers.\n\nThe values contained in ps should be a subsection of the outer product of  the ranges in iterators.\n\nExamples\n\njulia> iters = (1:10, 4:6, 1:4);\n\njulia> ps = ProductSplit(iters, 5, 2);\n\njulia> procrange_recast(iters, ps, 10)\n3:4\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.sumcat_aligned-Union{Tuple{Vararg{AbstractArray{T,N},N1} where N1}, Tuple{N}, Tuple{T}} where N where T","page":"API","title":"ParallelUtilities.sumcat_aligned","text":"sumcat_aligned(A::AbstractArray{T,N}...; dims) where {T,N}\n\nConcatenate the arrays along the dimensions dims according to their axes,  with overlapping sections being summed over. Returns an OffsetArray with the minimal  axis span encompassing all the arrays.\n\ndims may be an Integer or a collection of Integers, but all elements of dims must be from the range 1:N.\n\nExamples\n\njulia> ParallelUtilities.sumcat_aligned(ones(1:2), ones(4:5), dims=1)\n5-element OffsetArray(::Array{Float64,1}, 1:5) with eltype Float64 with indices 1:5:\n 1.0\n 1.0\n 0.0\n 1.0\n 1.0\n\njulia> ParallelUtilities.sumcat_aligned(ones(1:2, 1:2), ones(2:3, 2:3), dims=(1,2))\n3×3 OffsetArray(::Array{Float64,2}, 1:3, 1:3) with eltype Float64 with indices 1:3×1:3:\n 1.0  1.0  0.0\n 1.0  2.0  1.0\n 0.0  1.0  1.0\n\njulia> ParallelUtilities.sumcat_aligned(ones(1:2, 1:2), ones(3:4, 3:4), dims=(1,2))\n4×4 OffsetArray(::Array{Float64,2}, 1:4, 1:4) with eltype Float64 with indices 1:4×1:4:\n 1.0  1.0  0.0  0.0\n 1.0  1.0  0.0  0.0\n 0.0  0.0  1.0  1.0\n 0.0  0.0  1.0  1.0\n\nSee also: sumhcat_aligned, sumvcat_aligned\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.sumhcat_aligned-Union{Tuple{Vararg{AbstractArray{T,N},N1} where N1}, Tuple{N}, Tuple{T}} where N where T","page":"API","title":"ParallelUtilities.sumhcat_aligned","text":"sumhcat_aligned(A::AbstractArray{T,N}...) where {T,N}\n\nConcatenate the arrays along the second dimension according to their axes,  with overlapping sections being summed over. Returns an OffsetArray with the minimal  axis span encompassing all the arrays. \n\nThe input arrays must be at least two-dimensional.\n\nExamples\n\njulia> ParallelUtilities.sumhcat_aligned(ones(2, 1:2), ones(2, 4:5))\n2×5 OffsetArray(::Array{Float64,2}, 1:2, 1:5) with eltype Float64 with indices 1:2×1:5:\n 1.0  1.0  0.0  1.0  1.0\n 1.0  1.0  0.0  1.0  1.0\n\njulia> ParallelUtilities.sumhcat_aligned(ones(1:2, 1:2), ones(1:2, 2:3))\n2×3 OffsetArray(::Array{Float64,2}, 1:2, 1:3) with eltype Float64 with indices 1:2×1:3:\n 1.0  2.0  1.0\n 1.0  2.0  1.0\n\nSee also: sumcat_aligned, sumvcat_aligned\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.sumvcat_aligned-Union{Tuple{Vararg{AbstractArray{T,N},N1} where N1}, Tuple{N}, Tuple{T}} where N where T","page":"API","title":"ParallelUtilities.sumvcat_aligned","text":"sumvcat_aligned(A::AbstractArray{T,N}...) where {T,N}\n\nConcatenate the arrays along the first dimension according to their axes,  with overlapping sections being summed over. Returns an OffsetArray with the minimal  axis span encompassing all the arrays.\n\nThe input arrays must be at least one-dimensional.\n\nExamples\n\njulia> ParallelUtilities.sumvcat_aligned(ones(1:2), ones(4:5))\n5-element OffsetArray(::Array{Float64,1}, 1:5) with eltype Float64 with indices 1:5:\n 1.0\n 1.0\n 0.0\n 1.0\n 1.0\n\njulia> ParallelUtilities.sumvcat_aligned(ones(1:2, 1:2), ones(2:3, 1:2))\n3×2 OffsetArray(::Array{Float64,2}, 1:3, 1:2) with eltype Float64 with indices 1:3×1:2:\n 1.0  1.0\n 2.0  2.0\n 1.0  1.0\n\nSee also: sumcat_aligned, sumhcat_aligned\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.whichproc-Tuple{Any,Any,Integer}","page":"API","title":"ParallelUtilities.whichproc","text":"whichproc(iterators::Tuple, val::Tuple, np::Integer)\n\nReturn the processor rank that will contain val if the outer  product of the ranges contained in iterators is split evenly  across np processors.\n\nExamples\n\njulia> iters = (1:4, 2:3);\n\njulia> np = 2;\n\njulia> ProductSplit(iters, np, 2) |> collect\n4-element Array{Tuple{Int64,Int64},1}:\n (1, 3)\n (2, 3)\n (3, 3)\n (4, 3)\n\njulia> whichproc(iters, (2,3), np)\n2\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.whichproc_localindex-Tuple{Tuple,Tuple,Integer}","page":"API","title":"ParallelUtilities.whichproc_localindex","text":"whichproc_localindex(iterators::Tuple, val::Tuple, np::Integer)\n\nReturn (rank,ind), where rank is the rank of the worker that val will reside on if the outer product  of the ranges in iterators is spread over np workers, and ind is the index of val in the local section on that worker.\n\nExamples\n\njulia> iters = (1:4, 2:8);\n\njulia> np = 10;\n\njulia> whichproc_localindex(iters, (2,4), np)\n(4, 1)\n\njulia> ProductSplit(iters, np, 4) |> collect\n3-element Array{Tuple{Int64,Int64},1}:\n (2, 4)\n (3, 4)\n (4, 4)\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.workersactive-Tuple{Tuple}","page":"API","title":"ParallelUtilities.workersactive","text":"workersactive(iterators::Tuple)\n\nWorkers required to split the outer product of the iterators.  If prod(length, iterators) < nworkers() then the first prod(length, iterators) workers are chosen.\n\n\n\n\n\n","category":"method"},{"location":"#ParallelUtilities.ClusterQueryUtils.gethostnames","page":"API","title":"ParallelUtilities.ClusterQueryUtils.gethostnames","text":"gethostnames(procs = workers())\n\nReturn the hostname of each worker in procs. This is obtained by evaluating  Libc.gethostname() on each worker asynchronously.\n\nwarn: Warn\ngethostnames is deprecated in favor of hostnames    \n\n\n\n\n\n","category":"function"},{"location":"#ParallelUtilities.ClusterQueryUtils.hostnames","page":"API","title":"ParallelUtilities.ClusterQueryUtils.hostnames","text":"hostnames(procs = workers())\n\nReturn the hostname of each worker in procs. This is obtained by evaluating  Libc.gethostname() on each worker asynchronously.\n\n\n\n\n\n","category":"function"},{"location":"#ParallelUtilities.ClusterQueryUtils.nodenames","page":"API","title":"ParallelUtilities.ClusterQueryUtils.nodenames","text":"nodenames(procs = workers())\n\nReturn the unique hostnames that the workers in procs lie on.  On an HPC system these are usually the hostnames of the nodes involved.\n\n\n\n\n\n","category":"function"},{"location":"#ParallelUtilities.ClusterQueryUtils.nprocs_node","page":"API","title":"ParallelUtilities.ClusterQueryUtils.nprocs_node","text":"nprocs_node(procs = workers())\n\nReturn the number of workers on each host. On an HPC system this would return the number of workers on each node.\n\n\n\n\n\n","category":"function"},{"location":"#ParallelUtilities.ClusterQueryUtils.procs_node","page":"API","title":"ParallelUtilities.ClusterQueryUtils.procs_node","text":"procs_node(procs = workers())\n\nReturn the worker ids on each host of the cluster. On an HPC system this would return the workers on each node.\n\n\n\n\n\n","category":"function"}]
}
