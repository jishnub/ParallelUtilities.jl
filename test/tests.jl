using DataStructures
using Test

@everywhere begin
    using ParallelUtilities
    import ParallelUtilities: BinaryTreeNode, RemoteChannelContainer, BranchChannel, 
	Sorted, Unsorted, Ordering, pval, value, reducedvalue, reduceTreeNode, mapTreeNode,
    SequentialBinaryTree, OrderedBinaryTree, SegmentedSequentialBinaryTree,
    SegmentedOrderedBinaryTree,
    parentnoderank, nchildren,
    maybepvalput!, createbranchchannels, nworkersactive, workersactive,
    procs_node, leafrankfoldedtree
end

macro testsetwithinfo(str,ex)
    quote
        @info "Testing "*$str
        @testset $str begin $(esc(ex)); end;
    end
end

function showworkernumber(ind,nw)
    # Cursor starts off at the beginning of the line
    print("\u1b[K") # clear till end of line
    print("Testing on worker $ind of $nw")
    # return the cursor to the beginning of the line
    endchar = ind == nw ? "\n" : "\r"
    print(endchar)
end

@testsetwithinfo "ProductSplit" begin

	various_iters = [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6),
		    	(1:2,Base.OneTo(4),1:3:10)]

	function split_across_processors_iterators(arr::Iterators.ProductIterator,num_procs,proc_id)

	    num_tasks = length(arr);

	    num_tasks_per_process,num_tasks_leftover = divrem(num_tasks,num_procs)

	    num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
	    task_start = num_tasks_per_process*(proc_id-1) + min(num_tasks_leftover,proc_id-1) + 1;

	    Iterators.take(Iterators.drop(arr,task_start-1),num_tasks_on_proc)
	end

	function split_product_across_processors_iterators(arrs_tuple,num_procs,proc_id)
		split_across_processors_iterators(Iterators.product(arrs_tuple...),num_procs,proc_id)
	end

    @testset "Constructor" begin

	    function checkPSconstructor(iters,npmax=10)
	    	ntasks_total = prod(length.(iters))
			for np = 1:npmax, p = 1:np
		        ps = ProductSplit(iters,np,p)
		        @test collect(ps) == collect(split_product_across_processors_iterators(iters,np,p))
		        @test ntasks(ps) == ntasks_total
		        @test ntasks(ps.iterators) == ntasks_total
		        @test eltype(ps) == Tuple{map(eltype,iters)...}
		    end

		    @test_throws ParallelUtilities.ProcessorNumberError ProductSplit(iters,npmax,npmax+1)
		end

		@testset "0D" begin
		    @test_throws ArgumentError ProductSplit((),2,1)
		end

		@testset "cumprod" begin
		    @test ParallelUtilities._cumprod(1,()) == ()
		    @test ParallelUtilities._cumprod(1,(2,)) == (1,)
		    @test ParallelUtilities._cumprod(1,(2,3)) == (1,2)
		    @test ParallelUtilities._cumprod(1,(2,3,4)) == (1,2,6)
		end

    	@testset "1D" begin
	    	iters = (1:10,)
	    	checkPSconstructor(iters)
    	end
    	@testset "2D" begin
	    	iters = (1:10,4:6)
	    	checkPSconstructor(iters)
    	end
    	@testset "3D" begin
	    	iters = (1:10,4:6,1:4)
	    	checkPSconstructor(iters)
    	end
    	@testset "steps" begin
	    	iters = (1:2:10,4:1:6)
	    	checkPSconstructor(iters)
	    	iters = (10:-1:10,6:-2:0)
	    	@test_throws ParallelUtilities.DecreasingIteratorError ProductSplit(iters,3,2)
    	end
    	@testset "mixed" begin
    	    for iters in [(1:2,4:2:6),(1:2,Base.OneTo(4),1:3:10)]
	    		checkPSconstructor(iters)
	    	end
    	end

    	@testset "empty" begin
    	    iters = (1:1,)
    	    ps = ProductSplit(iters,10,2)
    	    @test isempty(ps)
    	    @test length(ps) == 0
    	end

    	@testset "first and last ind" begin
    	    for iters in [(1:10,),(1:2,Base.OneTo(4),1:3:10)]
	    	    ps = ProductSplit(iters,2,1)
	    	    @test firstindex(ps) == 1
	    	    @test ps.firstind == 1
	    	    @test ps.lastind == div(ntasks(iters),2)
	    	    @test lastindex(ps) == div(ntasks(iters),2)
	    	    @test lastindex(ps) == length(ps)
	    	    ps = ProductSplit(iters,2,2)
	    	    @test ps.firstind == div(ntasks(iters),2) + 1
	    	    @test firstindex(ps) == 1
	    	    @test ps.lastind == ntasks(iters)
	    	    @test lastindex(ps) == length(ps)

	    	    for np in ntasks(iters)+1:ntasks(iters)+10,
	    	    	p in ntasks(iters)+1:np

		    	    ps = ProductSplit(iters,np,p)
		    	    @test ps.firstind == ntasks(iters) + 1
		    	    @test ps.lastind == ntasks(iters)
		    	end
		    end
    	end

        @testset "summary" begin
            ps = ProductSplit((1:3, 4:5:19),3,2)
            reprstr = "ProductSplit("*repr((1:3, 4:5:19))*",3,2)"
            @test ParallelUtilities.mwerepr(ps) == reprstr

            summarystr = "$(length(ps))-element "*reprstr
            @test ParallelUtilities.summary(ps) == summarystr

            io = IOBuffer()
            summary(io,ps)
            @test String(take!(io)) == summarystr
        end
    end

    @testset "firstlast" begin
        @testset "first" begin

        	@test ParallelUtilities._first(()) == ()

            for iters in various_iters,np=1:5ntasks(iters)

	            ps = ProductSplit(iters,np,1)
	            @test first(ps) == ( isempty(ps) ? nothing : map(first,iters) )
	        end

	        iters = (1:1,)
	        ps = ProductSplit(iters,2ntasks(iters),ntasks(iters)+1) # must be empty
	        @test first(ps) === nothing
        end
        @testset "last" begin

        	@test ParallelUtilities._last(()) == ()

            for iters in various_iters,np=1:5ntasks(iters)

	            ps = ProductSplit(iters,np,np)
	            @test last(ps) == ( isempty(ps) ? nothing : map(last,iters) )
	        end

	        iters = (1:1,)
	        ps = ProductSplit(iters,2length(iters[1]),length(iters[1])+1) # must be empty
	        @test last(ps) === nothing
        end
    end

    @testset "extrema" begin

    	@testset "min max extrema" begin
	    	function checkPSextrema(iters,fn::Function,npmax=10)
				for np = 1:npmax, p = 1:np
			        ps = ProductSplit(iters,np,p)
			        pcol = collect(ps)
			        for dim in 1:length(iters)
			        	@test begin
			        		res = fn(ps,dim=dim) == fn(x[dim] for x in pcol)
			        		if !res
			        			println(summary(ps))
			        		end
			        		res
			        	end
			        end
			    end
			end

		    for iters in various_iters,	fn in [maximum,minimum,extrema]
		        checkPSextrema(iters,fn)
		    end

            @test minimum(ProductSplit((1:5,),2,1)) == 1
            @test maximum(ProductSplit((1:5,),2,1)) == 3
            @test extrema(ProductSplit((1:5,),2,1)) == (1,3)

            @test minimum(ProductSplit((1:5,),2,2)) == 4
            @test maximum(ProductSplit((1:5,),2,2)) == 5
            @test extrema(ProductSplit((1:5,),2,2)) == (4,5)
    	end

    	@testset "extremadims" begin
    		ps = ProductSplit((1:10,),2,1)
    		@test ParallelUtilities._extremadims(ps,1,()) == ()
    		for iters in various_iters

    			dims = length(iters)
	    		for np = 1:5ntasks(iters), proc_id = 1:np
	    	    	ps = ProductSplit(iters,np,proc_id)
	    	    	if isempty(ps)
	    	    		@test extremadims(ps) == Tuple(nothing for i=1:dims)
	    	    	else
		    	    	ext = Tuple(map(extrema,zip(collect(ps)...)))
		    	    	@test extremadims(ps) == ext
		    	    end
	    	    end
	    	end
    	end

    	@testset "extrema_commonlastdim" begin
    	    iters = (1:10,4:6,1:4)
    	    ps = ProductSplit(iters,37,8)
    	    @test extrema_commonlastdim(ps) == ([(9,1),(6,1)],[(2,2),(4,2)])
    	    ps = ProductSplit(iters,ntasks(iters)+1,ntasks(iters)+1)
    	    @test extrema_commonlastdim(ps) === nothing
    	end
    end

    @testset "in" begin

    	function checkifpresent(iters,npmax=10)
    		for np = 1:npmax, p = 1:np
		        ps = ProductSplit(iters,np,p)
		        pcol = collect(ps)

		        for el in pcol
		        	# It should be contained in this iterator
		        	@test el in ps
		        	for p2 in 1:np
		        		# It should not be contained anywhere else
		        		p2 == p && continue
		        		ps2 = ProductSplit(iters,np,p2)
		        		@test !(el in ps2)
		        	end
		        end
		    end
    	end

        for iters in various_iters
	        checkifpresent(iters)
	    end

	    @test ParallelUtilities._infullrange((),())
    end

    @testset "whichproc + procrange_recast" begin
        np,proc_id = 5,5
        iters = (1:10,4:6,1:4)
        ps = ProductSplit(iters,np,proc_id)
        @test whichproc(iters,first(ps),1) == 1
        @test whichproc(iters,(100,100,100),1) === nothing
        @test procrange_recast(iters,ps,1) == 1:1
        @test procrange_recast(ps,1) == 1:1

        smalleriter = (1:1,1:1,1:1)
        err = ParallelUtilities.TaskNotPresentError(smalleriter,first(ps))
        @test_throws err procrange_recast(smalleriter,ps,1)
        smalleriter = (7:9,4:6,1:4)
        err = ParallelUtilities.TaskNotPresentError(smalleriter,last(ps))
        @test_throws err procrange_recast(smalleriter,ps,1)

        iters = (1:1,2:2)
        ps = ProductSplit(iters,np,proc_id)
        @test whichproc(iters,first(ps),np) === nothing
        @test whichproc(iters,nothing,np) === nothing
        @test procrange_recast(iters,ps,2) == (0:-1)
        @test procrange_recast(ps,2) == (0:-1)

        iters = (1:1,2:2)
        ps = ProductSplit(iters,1,1)
        @test procrange_recast(iters,ps,2) == 1:1
        @test procrange_recast(ps,2) == 1:1

        iters = (Base.OneTo(2),2:4)
        ps = ProductSplit(iters,2,1)
        @test procrange_recast(iters,ps,1) == 1:1
        @test procrange_recast(iters,ps,2) == 1:1
        @test procrange_recast(iters,ps,ntasks(iters)) == 1:length(ps)

        for np_new in 1:5ntasks(iters)
        	for proc_id_new=1:np_new
	        	ps_new = ProductSplit(iters,np_new,proc_id_new)

	        	for val in ps_new
	        		# Should loop only if ps_new is non-empty
	        		@test whichproc(iters,val,np_new) == proc_id_new
	        	end
	        end
	        procid_new_first = whichproc(iters,first(ps),np_new)
	        proc_new_last = whichproc(iters,last(ps),np_new)
        	@test procrange_recast(iters,ps,np_new) == (isempty(ps) ? (0:-1) : (procid_new_first:proc_new_last))
        	@test procrange_recast(ps,np_new) == (isempty(ps) ? (0:-1) : (procid_new_first:proc_new_last))
        end

        @testset "different set" begin
	        iters = (1:100,1:4000)
	        ps = ProductSplit((20:30,1:1),2,1)
	        @test procrange_recast(iters,ps,700) == 1:1
	        ps = ProductSplit((20:30,1:1),2,2)
	        @test procrange_recast(iters,ps,700) == 1:1

	        iters = (1:1,2:2)
	        ps = ProductSplit((20:30,2:2),2,1)
	        @test_throws ParallelUtilities.TaskNotPresentError procrange_recast(iters,ps,3)
	        ps = ProductSplit((1:30,2:2),2,1)
	        @test_throws ParallelUtilities.TaskNotPresentError procrange_recast(iters,ps,3)
        end
    end

    @testset "localindex" begin
        
        for iters in various_iters
	        for np=1:5ntasks(iters),proc_id=1:np
	        	ps = ProductSplit(iters,np,proc_id)
	        	for (ind,val) in enumerate(ps)
	        		@test localindex(ps,val) == ind
	        		@test localindex(iters,val,np,proc_id) == ind
	        	end
	        	if isempty(ps)
	        		@test localindex(ps,first(ps)) === nothing
	        	end
	        end
	    end
    end

    @testset "whichproc_localindex" begin
        for iters in various_iters
	        for np=1:ntasks(iters),proc_id=1:np
	        	ps_col = collect(ProductSplit(iters,np,proc_id))
	        	ps_col_rev = [reverse(t) for t in ps_col] 
	        	for val in ps_col
	        		p,ind = whichproc_localindex(iters,val,np)
	        		@test p == proc_id
	        		ind_in_arr = searchsortedfirst(ps_col_rev,reverse(val))
	        		@test ind == ind_in_arr
	        	end
	        end
	    end
    end

    @testset "getindex" begin
    	
    	@test ParallelUtilities._getindex((),1) == ()
    	@test ParallelUtilities._getindex((),1,2) == ()

    	@test ParallelUtilities.childindex((),1) == (1,)

        for iters in various_iters
            for np=1:ntasks(iters),p=1:np
            	ps = ProductSplit(iters,np,p)
            	ps_col = collect(ps)
            	for i in 1:length(ps)
            		@test ps[i] == ps_col[i]
            	end
            	@test ps[end] == ps[length(ps)]
                for ind in [0,length(ps)+1]
            	   @test_throws ParallelUtilities.BoundsError(ps,ind) ps[ind]
                end
            end
        end
    end
end;

@testset "ReverseLexicographicTuple" begin
    @testset "isless" begin
    	a = ParallelUtilities.ReverseLexicographicTuple((1,2,3))
        b = ParallelUtilities.ReverseLexicographicTuple((2,2,3))
        @test a < b
        @test a <= b
        b = ParallelUtilities.ReverseLexicographicTuple((1,1,3))
        @test b < a
        @test b <= a
        b = ParallelUtilities.ReverseLexicographicTuple((2,1,3))
        @test b < a
        @test b <= a
        b = ParallelUtilities.ReverseLexicographicTuple((2,1,4))
        @test a < b
        @test a <= b
    end
    @testset "equal" begin
        a = ParallelUtilities.ReverseLexicographicTuple((1,2,3))
        @test a == a
        @test isequal(a,a)
        @test a <= a
        b = ParallelUtilities.ReverseLexicographicTuple(a.t)
        @test a == b
        @test isequal(a,b)
        @test a <= b
    end
end;

@testset "utilities" begin
    @testset "workers active" begin
        @test nworkersactive((1:1,)) == 1
        @test nworkersactive((1:2,)) == min(2,nworkers())
        @test nworkersactive((1:1,1:2)) == min(2,nworkers())
        @test nworkersactive(1:2) == min(2,nworkers())
        @test nworkersactive(1:1,1:2) == min(2,nworkers())
        @test nworkersactive((1:nworkers()+1,)) == nworkers()
        @test nworkersactive(1:nworkers()+1) == nworkers()
    	@test workersactive((1:1,)) == workers()[1:1]
    	@test workersactive(1:1) == workers()[1:1]
    	@test workersactive(1:1,1:1) == workers()[1:1]
        @test workersactive((1:2,)) == workers()[1:min(2,nworkers())]
        @test workersactive((1:1,1:2)) == workers()[1:min(2,nworkers())]
        @test workersactive(1:1,1:2) == workers()[1:min(2,nworkers())]
        @test workersactive((1:nworkers()+1,)) == workers()
        @test workersactive(1:nworkers()+1) == workers()

        ps = ProductSplit((1:10,),nworkers(),1)
        @test nworkersactive(ps) == min(10,nworkers())

        iters = (1:1,1:2)
        ps = ProductSplit(iters,2,1)
        @test nworkersactive(ps) == nworkersactive(iters)
        @test workersactive(ps) == workersactive(iters)
    end

    @testset "hostnames" begin
    	hostnames = gethostnames()
    	nodes = unique(hostnames)
        @test hostnames == [@fetchfrom p Libc.gethostname() for p in workers()]
        @test nodenames() == nodes
        @test nodenames(hostnames) == nodes
        np1 = nprocs_node(hostnames,nodes)
        np2 = nprocs_node(hostnames)
        np3 = nprocs_node()
        @test np1 == np2 == np3
        for node in nodes
            npnode = count(isequal(node),hostnames)
            @test np1[node] == npnode
        end
        p1 = procs_node()
        p2 = procs_node(workers(),hostnames,nodes)
        @test p1 == p2
        for node in nodes
            pnode = workers()[findall(isequal(node),hostnames)]
            @test p1[node] == pnode
        end
        np4 = nprocs_node(p1)
        @test np1 == np4
    end
end;

@testset "BinaryTree" begin
    @testsetwithinfo "BinaryTreeNode" begin
    	@testset "Constructor" begin
	    	p = workers()[1]
	    	b = BinaryTreeNode(p,p,0)
	        @test nchildren(b) == 0
	        b = BinaryTreeNode(p,p,1)
	        @test nchildren(b) == 1
	        b = BinaryTreeNode(p,p,2)
	        @test nchildren(b) == 2

	        @test_throws DomainError BinaryTreeNode(p,p,3)
            @test_throws DomainError BinaryTreeNode(p,p,-1)
    	end
    end

    @testsetwithinfo "BinaryTree" begin
        @testsetwithinfo "SequentialBinaryTree" begin
            @testset "pid and parent" begin
                for imax = 1:100
                    procs = 1:imax
                    tree = SequentialBinaryTree(procs)
                    @test length(tree) == length(procs) 
                    topnoderank = ParallelUtilities.topnoderank(tree)
                    @test topnoderank == 1
                    @test tree[topnoderank] == ParallelUtilities.topnode(tree)
                    @test tree[1].parent == 1
                    for rank in 1:length(tree)
                        node = tree[rank]
                        @test node.p == procs[rank]
                        @test node.parent == procs[parentnoderank(tree,rank)]
                    end

                    for ind in [0,imax+1]
                        @test_throws BoundsError(tree,ind) parentnoderank(tree,ind)
                        @test_throws BoundsError(tree,ind) tree[ind]
                    end
                end
            end
                
            @testset "nchildren" begin
                tree = SequentialBinaryTree(1:1)
                @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 0
                @test_throws BoundsError(tree,0) nchildren(tree,0)
                @test_throws BoundsError(tree,2) nchildren(tree,2)

                tree = SequentialBinaryTree(1:2)
                @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 1
                @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 0
                @test_throws BoundsError(tree,0) nchildren(tree,0)
                @test_throws BoundsError(tree,3) nchildren(tree,3)

                tree = SequentialBinaryTree(1:8)
                @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 2
                @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 2
                @test nchildren(tree,3) == nchildren(tree[3]) == tree[3].nchildren == 2
                @test nchildren(tree,4) == nchildren(tree[4]) == tree[4].nchildren == 1
                @test nchildren(tree,5) == nchildren(tree[5]) == tree[5].nchildren == 0
                @test nchildren(tree,6) == nchildren(tree[6]) == tree[6].nchildren == 0
                @test nchildren(tree,7) == nchildren(tree[7]) == tree[7].nchildren == 0
                @test nchildren(tree,8) == nchildren(tree[8]) == tree[8].nchildren == 0
                @test_throws BoundsError(tree,0) nchildren(tree,0)
                @test_throws BoundsError(tree,9) nchildren(tree,9)
            end

            @testset "level" begin
                tree = SequentialBinaryTree(1:15)
                @test ParallelUtilities.levels(tree) == 4

                @test ParallelUtilities.levelfromtop(tree,1) == 1
                @test ParallelUtilities.levelfromtop.((tree,),2:3) == ones(Int,2)*2
                @test ParallelUtilities.levelfromtop.((tree,),4:7) == ones(Int,4)*3
                @test ParallelUtilities.levelfromtop.((tree,),8:15) == ones(Int,8)*4

                for p in [0,length(tree)+1]
                    @test_throws BoundsError(tree,p) ParallelUtilities.levelfromtop(tree,p)
                end
            end

            @testset "summary" begin
                tree = SequentialBinaryTree(1:4)
                io = IOBuffer()
                summary(io,tree)
                strexp = "$(length(tree))-node $(typeof(tree))"
                @test String(take!(io)) == strexp
                @test summary(tree) == strexp
            end
        end

        @testsetwithinfo "OrderedBinaryTree" begin
            @testset "pid and parent" begin
                for imax = 1:100
                    procs = 1:imax
                    tree = OrderedBinaryTree(procs)
                    @test length(tree) == length(procs)

                    topnoderank = ParallelUtilities.topnoderank(tree)
                    @test tree[topnoderank].parent == topnoderank
                    for rank in 1:length(tree)
                        node = tree[rank]
                        @test node.p == procs[rank]
                        @test node.parent == procs[parentnoderank(tree,rank)]
                    end
                    @test_throws BoundsError(tree,0) parentnoderank(tree,0)
                    @test_throws BoundsError(tree,imax+1) parentnoderank(tree,imax+1)
                end
            end

            @testset "nchildren" begin
                tree = OrderedBinaryTree(1:1)
                @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 0
                @test_throws BoundsError(tree,0) nchildren(tree,0)
                @test_throws BoundsError(tree,2) nchildren(tree,2)
                @test ParallelUtilities.topnoderank(tree) == 1

                tree = OrderedBinaryTree(1:2)
                @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 0
                @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 1
                @test_throws BoundsError(tree,0) nchildren(tree,0)
                @test_throws BoundsError(tree,3) nchildren(tree,3)
                @test ParallelUtilities.topnoderank(tree) == 2

                tree = OrderedBinaryTree(1:8)
                @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 0
                @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 2
                @test nchildren(tree,3) == nchildren(tree[3]) == tree[3].nchildren == 0
                @test nchildren(tree,4) == nchildren(tree[4]) == tree[4].nchildren == 2
                @test nchildren(tree,5) == nchildren(tree[5]) == tree[5].nchildren == 0
                @test nchildren(tree,6) == nchildren(tree[6]) == tree[6].nchildren == 2
                @test nchildren(tree,7) == nchildren(tree[7]) == tree[7].nchildren == 0
                @test nchildren(tree,8) == nchildren(tree[8]) == tree[8].nchildren == 1
                @test_throws BoundsError(tree,0) nchildren(tree,0)
                @test_throws BoundsError(tree,9) nchildren(tree,9)
                @test ParallelUtilities.topnoderank(tree) == 8

                tree = OrderedBinaryTree(1:11)
                @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 0
                @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 2
                @test nchildren(tree,3) == nchildren(tree[3]) == tree[3].nchildren == 0
                @test nchildren(tree,4) == nchildren(tree[4]) == tree[4].nchildren == 2
                @test nchildren(tree,5) == nchildren(tree[5]) == tree[5].nchildren == 0
                @test nchildren(tree,6) == nchildren(tree[6]) == tree[6].nchildren == 2
                @test nchildren(tree,7) == nchildren(tree[7]) == tree[7].nchildren == 0
                @test nchildren(tree,8) == nchildren(tree[8]) == tree[8].nchildren == 2
                @test nchildren(tree,9) == nchildren(tree[9]) == tree[9].nchildren == 0
                @test nchildren(tree,10) == nchildren(tree[10]) == tree[10].nchildren == 2
                @test nchildren(tree,11) == nchildren(tree[11]) == tree[11].nchildren == 0
                @test_throws BoundsError(tree,0) nchildren(tree,0)
                @test_throws BoundsError(tree,12) nchildren(tree,12)
                @test ParallelUtilities.topnoderank(tree) == 8

                tree = OrderedBinaryTree(1:13)
                @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 0
                @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 2
                @test nchildren(tree,3) == nchildren(tree[3]) == tree[3].nchildren == 0
                @test nchildren(tree,4) == nchildren(tree[4]) == tree[4].nchildren == 2
                @test nchildren(tree,5) == nchildren(tree[5]) == tree[5].nchildren == 0
                @test nchildren(tree,6) == nchildren(tree[6]) == tree[6].nchildren == 2
                @test nchildren(tree,7) == nchildren(tree[7]) == tree[7].nchildren == 0
                @test nchildren(tree,8) == nchildren(tree[8]) == tree[8].nchildren == 2
                @test nchildren(tree,9) == nchildren(tree[9]) == tree[9].nchildren == 0
                @test nchildren(tree,10) == nchildren(tree[10]) == tree[10].nchildren == 2
                @test nchildren(tree,11) == nchildren(tree[11]) == tree[11].nchildren == 0
                @test nchildren(tree,12) == nchildren(tree[12]) == tree[12].nchildren == 2
                @test nchildren(tree,13) == nchildren(tree[13]) == tree[13].nchildren == 0
                @test_throws BoundsError(tree,0) nchildren(tree,0)
                @test_throws BoundsError(tree,14) nchildren(tree,14)
                @test ParallelUtilities.topnoderank(tree) == 8
            end

            @testset "level" begin
                tree = OrderedBinaryTree(1:15)
                @test ParallelUtilities.levels(tree) == 4

                @test ParallelUtilities.levelfromtop.((tree,),1:2:15) == ones(Int,8).*4
                @test ParallelUtilities.levelfromtop.((tree,),(2,6,10,14)) == (3,3,3,3)
                @test ParallelUtilities.levelfromtop.((tree,),(4,12)) == (2,2)
                @test ParallelUtilities.levelfromtop(tree,8) == 1
                for p in [0,length(tree)+1]
                    @test_throws BoundsError(tree,p) ParallelUtilities.levelfromtop(tree,p)
                end

                tree = OrderedBinaryTree(1:13)
                @test ParallelUtilities.levels(tree) == 4
                @test ParallelUtilities.levelfromtop.((tree,),1:2:11) == ones(Int,6).*4
                @test ParallelUtilities.levelfromtop.((tree,),(2,6,10,13)) == (3,3,3,3)
                @test ParallelUtilities.levelfromtop.((tree,),(4,12)) == (2,2)
                @test ParallelUtilities.levelfromtop(tree,8) == 1
                for p in [0,length(tree)+1]
                    @test_throws BoundsError(tree,p) ParallelUtilities.levelfromtop(tree,p)
                end
            end
        end

        @testsetwithinfo "SegmentedSequentialBinaryTree" begin
            @testsetwithinfo "single host" begin
                @testset "pid and parent" begin
                    for imax = 1:100
                        procs = 1:imax
                        workersonnodes = Dict("host"=>procs)
                        tree = SegmentedSequentialBinaryTree(procs,workersonnodes)
                        SBT = SequentialBinaryTree(procs)
                        @test length(tree) == length(procs) == length(SBT)

                        topnoderank = ParallelUtilities.topnoderank(tree)
                        @test topnoderank == 1
                        @test tree[topnoderank] == ParallelUtilities.topnode(tree)
                        @test tree[1].parent == 1
                        for rank in 1:length(tree)
                            node = tree[rank]
                            parentnode = tree[parentnoderank(tree,rank)]
                            @test length(procs) > 1 ? nchildren(parentnode) > 0 : nchildren(parentnode) == 0
                            @test node.p == procs[rank]
                            @test node.parent == procs[parentnoderank(SBT,rank)]
                            @test parentnode.p == node.parent
                        end
                    end
                end;

                @testset "nchildren" begin
                    procs = 1:1
                    tree = SegmentedSequentialBinaryTree(procs,Dict("host"=>procs))
                    @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 0
                    @test_throws BoundsError(tree,0) nchildren(tree,0)
                    @test_throws BoundsError(tree,2) nchildren(tree,2)

                    procs = 1:2
                    tree = SegmentedSequentialBinaryTree(procs,Dict("host"=>procs))
                    @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 1
                    @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 0
                    @test_throws BoundsError(tree,0) nchildren(tree,0)
                    @test_throws BoundsError(tree,3) nchildren(tree,3)

                    procs = 1:8
                    tree = SegmentedSequentialBinaryTree(procs,Dict("host"=>procs))
                    @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 2
                    @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 2
                    @test nchildren(tree,3) == nchildren(tree[3]) == tree[3].nchildren == 2
                    @test nchildren(tree,4) == nchildren(tree[4]) == tree[4].nchildren == 1
                    @test nchildren(tree,5) == nchildren(tree[5]) == tree[5].nchildren == 0
                    @test nchildren(tree,6) == nchildren(tree[6]) == tree[6].nchildren == 0
                    @test nchildren(tree,7) == nchildren(tree[7]) == tree[7].nchildren == 0
                    @test nchildren(tree,8) == nchildren(tree[8]) == tree[8].nchildren == 0
                    @test_throws BoundsError(tree,0) nchildren(tree,0)
                    @test_throws BoundsError(tree,9) nchildren(tree,9)
                end;
            end;

            @testsetwithinfo "multiple hosts" begin
                @testset "length" begin
                    procs = 1:2
                    tree = SegmentedSequentialBinaryTree(procs,
                        OrderedDict("host1"=>1:1,"host2"=>2:2))
                    @test length(tree) == 2 + 1

                    procs = 1:4
                    tree = SegmentedSequentialBinaryTree(procs,
                        OrderedDict("host1"=>1:2,"host2"=>3:4))

                    @test length(tree) == 4 + 1

                    procs = 1:12
                    tree = SegmentedSequentialBinaryTree(procs,
                        OrderedDict(
                            "host1"=>1:3,"host2"=>4:6,
                            "host3"=>7:9,"host4"=>10:12))

                    @test length(tree) == 12 + 3 
                end;

                @testset "leafrankfoldedtree" begin
                    treeflag = SequentialBinaryTree(1:1)
                    @test leafrankfoldedtree(treeflag,5,1) == 8
                    @test leafrankfoldedtree(treeflag,5,2) == 9
                    @test leafrankfoldedtree(treeflag,5,3) == 5
                    @test leafrankfoldedtree(treeflag,5,4) == 6
                    @test leafrankfoldedtree(treeflag,5,5) == 7
                end;

                @testset "pid and parent" begin
                    for imax = 2:100
                        procs = 1:imax
                        mid = div(imax,2)
                        workersonnodes = OrderedDict{String,Vector{Int}}()
                        workersonnodes["host1"] = procs[1:mid]
                        workersonnodes["host2"] = procs[mid+1:end]
                        tree = SegmentedSequentialBinaryTree(procs,workersonnodes)

                        topnoderank = ParallelUtilities.topnoderank(tree)
                        @test topnoderank == 1
                        @test tree[topnoderank] == ParallelUtilities.topnode(tree)
                        @test tree[1].parent == 1
                        @test parentnoderank(tree,1) == 1
                        for (ind,rank) in enumerate(1:mid)
                            node = tree[rank+1]
                            parentnode = tree[parentnoderank(tree,rank+1)]
                            @test nchildren(parentnode) > 0
                            @test parentnode.p == node.parent
                            pnodes = workersonnodes["host1"]
                            @test node.p == pnodes[ind]
                            SBT = SequentialBinaryTree(pnodes)
                            if ind == 1
                                @test node.parent == 1
                            else
                                @test node.parent == pnodes[parentnoderank(SBT,ind)]
                            end
                        end
                        for (ind,rank) in enumerate(mid+1:imax)
                            node = tree[rank+1]
                            parentnode = tree[parentnoderank(tree,rank+1)]
                            @test nchildren(parentnode) > 0
                            @test parentnode.p == node.parent
                            pnodes = workersonnodes["host2"]
                            @test node.p == pnodes[ind]
                            SBT = SequentialBinaryTree(pnodes)
                            if ind == 1
                                @test node.parent == 1
                            else
                                @test node.parent == pnodes[parentnoderank(SBT,ind)]
                            end
                        end
                    end
                end;

                @testset "nchildren" begin
                    procs = 1:2
                    tree = SegmentedSequentialBinaryTree(procs,
                        OrderedDict("host1"=>1:1,"host2"=>2:2))
                    @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 2
                    @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 0
                    @test nchildren(tree,3) == nchildren(tree[3]) == tree[3].nchildren == 0
                    @test_throws BoundsError(tree,0) nchildren(tree,0)
                    @test_throws BoundsError(tree,4) nchildren(tree,4)

                    procs = 1:12
                    tree = SegmentedSequentialBinaryTree(procs,
                        OrderedDict(
                            "host1"=>1:3,"host2"=>4:6,
                            "host3"=>7:9,"host4"=>10:12))
                    @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 2
                    @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 2
                    @test nchildren(tree,3) == nchildren(tree[3]) == tree[3].nchildren == 2
                    @test nchildren(tree,4) == nchildren(tree[4]) == tree[4].nchildren == 2
                    @test nchildren(tree,5) == nchildren(tree[5]) == tree[5].nchildren == 0
                    @test nchildren(tree,6) == nchildren(tree[6]) == tree[6].nchildren == 0
                    @test nchildren(tree,7) == nchildren(tree[7]) == tree[7].nchildren == 2
                    @test nchildren(tree,8) == nchildren(tree[8]) == tree[8].nchildren == 0
                    @test nchildren(tree,9) == nchildren(tree[9]) == tree[9].nchildren == 0
                    @test nchildren(tree,10) == nchildren(tree[10]) == tree[10].nchildren == 2
                    @test nchildren(tree,11) == nchildren(tree[11]) == tree[11].nchildren == 0
                    @test nchildren(tree,12) == nchildren(tree[12]) == tree[12].nchildren == 0
                    @test nchildren(tree,13) == nchildren(tree[13]) == tree[13].nchildren == 2
                    @test nchildren(tree,14) == nchildren(tree[14]) == tree[14].nchildren == 0
                    @test nchildren(tree,15) == nchildren(tree[15]) == tree[15].nchildren == 0
                    @test_throws BoundsError(tree,0) nchildren(tree,0)
                    @test_throws BoundsError(tree,16) nchildren(tree,16)
                end;
            end;
        end

        @testsetwithinfo "SegmentedOrderedBinaryTree" begin
            @testsetwithinfo "single host" begin
                @testset "pid and parent" begin
                    for imax = 1:100
                        procs = 1:imax
                        workersonnodes = Dict("host"=>procs)
                        tree = SegmentedOrderedBinaryTree(procs,workersonnodes)
                        treeOBT = OrderedBinaryTree(procs)
                        @test length(tree) == length(procs) == length(treeOBT)

                        topnoderank = ParallelUtilities.topnoderank(tree)
                        # The top node is its own parent
                        @test tree[topnoderank].parent == topnoderank
                        @test tree[topnoderank] == ParallelUtilities.topnode(tree)
                        for rank in 1:length(tree)
                            node = tree[rank]
                            parentnode = tree[parentnoderank(tree,rank)]
                            @test length(procs) > 1 ? nchildren(parentnode) > 0 : nchildren(parentnode) == 0
                            @test node.p == procs[rank]
                            @test node.parent == procs[parentnoderank(treeOBT,rank)]
                            @test parentnode.p == node.parent
                        end
                    end
                end;

                @testset "nchildren" begin
                    procs = 1:1
                    tree = SegmentedOrderedBinaryTree(procs,Dict("host"=>procs))
                    @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 0
                    @test_throws BoundsError(tree,0) nchildren(tree,0)
                    @test_throws BoundsError(tree,2) nchildren(tree,2)
                    @test ParallelUtilities.topnoderank(tree) == 1

                    procs = 1:2
                    tree = SegmentedOrderedBinaryTree(procs,Dict("host"=>procs))
                    @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 0
                    @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 1
                    @test_throws BoundsError(tree,0) nchildren(tree,0)
                    @test_throws BoundsError(tree,3) nchildren(tree,3)
                    @test ParallelUtilities.topnoderank(tree) == 2

                    procs = 1:8
                    tree = SegmentedOrderedBinaryTree(procs,Dict("host"=>procs))
                    @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 0
                    @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 2
                    @test nchildren(tree,3) == nchildren(tree[3]) == tree[3].nchildren == 0
                    @test nchildren(tree,4) == nchildren(tree[4]) == tree[4].nchildren == 2
                    @test nchildren(tree,5) == nchildren(tree[5]) == tree[5].nchildren == 0
                    @test nchildren(tree,6) == nchildren(tree[6]) == tree[6].nchildren == 2
                    @test nchildren(tree,7) == nchildren(tree[7]) == tree[7].nchildren == 0
                    @test nchildren(tree,8) == nchildren(tree[8]) == tree[8].nchildren == 1
                    @test_throws BoundsError(tree,0) nchildren(tree,0)
                    @test_throws BoundsError(tree,9) nchildren(tree,9)
                    @test ParallelUtilities.topnoderank(tree) == 8

                    procs = 1:11
                    tree = SegmentedOrderedBinaryTree(procs,Dict("host"=>procs))
                    @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 0
                    @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 2
                    @test nchildren(tree,3) == nchildren(tree[3]) == tree[3].nchildren == 0
                    @test nchildren(tree,4) == nchildren(tree[4]) == tree[4].nchildren == 2
                    @test nchildren(tree,5) == nchildren(tree[5]) == tree[5].nchildren == 0
                    @test nchildren(tree,6) == nchildren(tree[6]) == tree[6].nchildren == 2
                    @test nchildren(tree,7) == nchildren(tree[7]) == tree[7].nchildren == 0
                    @test nchildren(tree,8) == nchildren(tree[8]) == tree[8].nchildren == 2
                    @test nchildren(tree,9) == nchildren(tree[9]) == tree[9].nchildren == 0
                    @test nchildren(tree,10) == nchildren(tree[10]) == tree[10].nchildren == 2
                    @test nchildren(tree,11) == nchildren(tree[11]) == tree[11].nchildren == 0
                    @test_throws BoundsError(tree,0) nchildren(tree,0)
                    @test_throws BoundsError(tree,12) nchildren(tree,12)
                    @test ParallelUtilities.topnoderank(tree) == 8

                    procs = 1:13
                    tree = SegmentedOrderedBinaryTree(procs,Dict("host"=>procs))
                    @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 0
                    @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 2
                    @test nchildren(tree,3) == nchildren(tree[3]) == tree[3].nchildren == 0
                    @test nchildren(tree,4) == nchildren(tree[4]) == tree[4].nchildren == 2
                    @test nchildren(tree,5) == nchildren(tree[5]) == tree[5].nchildren == 0
                    @test nchildren(tree,6) == nchildren(tree[6]) == tree[6].nchildren == 2
                    @test nchildren(tree,7) == nchildren(tree[7]) == tree[7].nchildren == 0
                    @test nchildren(tree,8) == nchildren(tree[8]) == tree[8].nchildren == 2
                    @test nchildren(tree,9) == nchildren(tree[9]) == tree[9].nchildren == 0
                    @test nchildren(tree,10) == nchildren(tree[10]) == tree[10].nchildren == 2
                    @test nchildren(tree,11) == nchildren(tree[11]) == tree[11].nchildren == 0
                    @test nchildren(tree,12) == nchildren(tree[12]) == tree[12].nchildren == 2
                    @test nchildren(tree,13) == nchildren(tree[13]) == tree[13].nchildren == 0
                    @test_throws BoundsError(tree,0) nchildren(tree,0)
                    @test_throws BoundsError(tree,14) nchildren(tree,14)
                    @test ParallelUtilities.topnoderank(tree) == 8
                end;
            end;

            @testsetwithinfo "multiple hosts" begin
                @testset "length" begin
                    procs = 1:2
                    tree = SegmentedOrderedBinaryTree(procs,
                        OrderedDict("host1"=>1:1,"host2"=>2:2))
                    @test length(tree) == 2 + 1

                    procs = 1:4
                    tree = SegmentedOrderedBinaryTree(procs,
                        OrderedDict("host1"=>1:2,"host2"=>3:4))

                    @test length(tree) == 4 + 1

                    procs = 1:12
                    tree = SegmentedOrderedBinaryTree(procs,
                        OrderedDict(
                            "host1"=>1:3,"host2"=>4:6,
                            "host3"=>7:9,"host4"=>10:12))

                    @test length(tree) == 12 + 3 
                end;

                @testset "leafrankfoldedtree" begin
                    treeflag = OrderedBinaryTree(1:1)
                    @test leafrankfoldedtree(treeflag,5,1) == 1
                    @test leafrankfoldedtree(treeflag,5,2) == 3
                    @test leafrankfoldedtree(treeflag,5,3) == 5
                    @test leafrankfoldedtree(treeflag,5,4) == 7
                    @test leafrankfoldedtree(treeflag,5,5) == 9
                end;

                @testset "pid and parent" begin
                    for imax = 2:100
                        procs = 1:imax
                        mid = div(imax,2)
                        workersonnodes = OrderedDict{String,Vector{Int}}()
                        workersonnodes["host1"] = procs[1:mid]
                        workersonnodes["host2"] = procs[mid+1:end]
                        tree = SegmentedOrderedBinaryTree(procs,workersonnodes)

                        top = ParallelUtilities.topnoderank(tree)
                        @test tree[top] == ParallelUtilities.topnode(tree)
                        for (ind,rank) in enumerate(1:mid)
                            node = tree[rank+1]
                            parentnode = tree[parentnoderank(tree,rank+1)]
                            @test parentnode.p == node.parent
                            pnodes = workersonnodes["host1"]
                            @test node.p == pnodes[ind]
                            OBT = OrderedBinaryTree(pnodes)
                            if ind == ParallelUtilities.topnoderank(OBT)
                                # Special check for 2 hosts as 
                                # there's only one node in the top tree
                                @test node.parent == ParallelUtilities.topnode(tree.toptree).p
                            else
                                @test node.parent == pnodes[parentnoderank(OBT,ind)]
                            end
                        end
                        for (ind,rank) in enumerate(mid+1:imax)
                            node = tree[rank+1]
                            parentnode = tree[parentnoderank(tree,rank+1)]
                            @test parentnode.p == node.parent
                            pnodes = workersonnodes["host2"]
                            @test node.p == pnodes[ind]
                            OBT = OrderedBinaryTree(pnodes)
                            if ind == ParallelUtilities.topnoderank(OBT)
                                # Special check for 2 hosts as 
                                # there's only one node in the top tree
                                @test node.parent == ParallelUtilities.topnode(tree.toptree).p
                            else
                                @test node.parent == pnodes[parentnoderank(OBT,ind)]
                            end
                        end
                    end
                end;

                @testset "nchildren" begin
                    procs = 1:2
                    tree = SegmentedOrderedBinaryTree(procs,
                        OrderedDict("host1"=>1:1,"host2"=>2:2))
                    @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 2
                    @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 0
                    @test nchildren(tree,3) == nchildren(tree[3]) == tree[3].nchildren == 0
                    @test_throws BoundsError(tree,0) nchildren(tree,0)
                    @test_throws BoundsError(tree,4) nchildren(tree,4)

                    procs = 1:12
                    tree = SegmentedOrderedBinaryTree(procs,
                        OrderedDict(
                            "host1"=>1:3,"host2"=>4:6,
                            "host3"=>7:9,"host4"=>10:12))
                    @test nchildren(tree,1) == nchildren(tree[1]) == tree[1].nchildren == 2
                    @test nchildren(tree,2) == nchildren(tree[2]) == tree[2].nchildren == 2
                    @test nchildren(tree,3) == nchildren(tree[3]) == tree[3].nchildren == 2
                    @test nchildren(tree,4) == nchildren(tree[4]) == tree[4].nchildren == 0
                    @test nchildren(tree,5) == nchildren(tree[5]) == tree[5].nchildren == 2
                    @test nchildren(tree,6) == nchildren(tree[6]) == tree[6].nchildren == 0
                    @test nchildren(tree,7) == nchildren(tree[7]) == tree[7].nchildren == 0
                    @test nchildren(tree,8) == nchildren(tree[8]) == tree[8].nchildren == 2
                    @test nchildren(tree,9) == nchildren(tree[9]) == tree[9].nchildren == 0
                    @test nchildren(tree,10) == nchildren(tree[10]) == tree[10].nchildren == 0
                    @test nchildren(tree,11) == nchildren(tree[11]) == tree[11].nchildren == 2
                    @test nchildren(tree,12) == nchildren(tree[12]) == tree[12].nchildren == 0
                    @test nchildren(tree,13) == nchildren(tree[13]) == tree[13].nchildren == 0
                    @test nchildren(tree,14) == nchildren(tree[14]) == tree[14].nchildren == 2
                    @test nchildren(tree,15) == nchildren(tree[15]) == tree[15].nchildren == 0
                    @test_throws BoundsError(tree,0) nchildren(tree,0)
                    @test_throws BoundsError(tree,16) nchildren(tree,16)
                end;
            end;
        end
    end
    
    @testsetwithinfo "RemoteChannelContainer" begin
    	@testsetwithinfo "Constructor" begin
    	    rc = ParallelUtilities.RemoteChannelContainer{Int}(1,myid())
	        @test rc.out.where == myid()
	        @test rc.err.where == myid()
	        @test eltype(rc) == Int
	        for p in workers()
    	        rc = ParallelUtilities.RemoteChannelContainer{Int}(1,p)
    	        @test rc.out.where == p
    	        @test rc.err.where == p
    	        @test eltype(rc) == Int
    	    end

	        rc = ParallelUtilities.RemoteChannelContainer{Int}(1)
	        @test rc.out.where == myid()
	        @test rc.err.where == myid()
	        @test eltype(rc) == Int

	        rc = ParallelUtilities.RemoteChannelContainer(1,myid())
	        @test rc.out.where == myid()
	        @test rc.err.where == myid()
	        @test eltype(rc) == Any

	        for p in workers()
    	        rc = ParallelUtilities.RemoteChannelContainer(1,p)
    	        @test rc.out.where == p
    	        @test rc.err.where == p
    	        @test eltype(rc) == Any
    	    end

	        rc = ParallelUtilities.RemoteChannelContainer(1)
	        @test rc.out.where == myid()
	        @test rc.err.where == myid()
	        @test eltype(rc) == Any
    	end

        @testsetwithinfo "finalize" begin
            rc = ParallelUtilities.RemoteChannelContainer{Int}(1)
            finalize(rc)
            @test rc.out.where == 0
            @test rc.err.where == 0
        end

        @testsetwithinfo "finalize_except_wherewhence" begin
            rc = ParallelUtilities.RemoteChannelContainer{Int}(1)
            ParallelUtilities.finalize_except_wherewhence(rc)
            @test rc.out.where == myid()
            @test rc.err.where == myid()

            @testset "rc on where" begin
                # Create on this processor
                rc = ParallelUtilities.RemoteChannelContainer{Int}(1)
                for (ind,p) in enumerate(workers())
                    showworkernumber(ind,nworkers())
                    rcoutw,rcerrw = @fetchfrom p begin 
                        ParallelUtilities.finalize_except_wherewhence(rc)
                        rc.out.where,rc.err.where
                    end
                    @test rc.out.where == myid()
                    @test rc.err.where == myid()
                    @test (rcoutw,rcerrw) == (0,0)
                end
            end

            @testset "rc on remote" begin
                # Create elsewhere
                p_rc = workers()[1]
                rc = ParallelUtilities.RemoteChannelContainer{Int}(1,p_rc)
                for (ind,p) in enumerate(procs())
                    showworkernumber(ind,nprocs())
                    rcw = @fetchfrom p begin 
                        ParallelUtilities.finalize_except_wherewhence(rc)
                        (rc.out.where,rc.err.where)
                    end
                    if p != myid() && p != p_rc
                        @test rcw == (0,0)
                    else
                        @test rcw == (p_rc,p_rc)
                    end
                end
            end
        end
    end

	@testsetwithinfo "BranchChannel" begin
	    @testset "Constructor" begin
	    	@testset "all channels supplied" begin
    	        rc_self = RemoteChannelContainer{Int}(1)
    	        rc_parent = RemoteChannelContainer{Int}(1)
    	        rc_children = RemoteChannelContainer{Int}(1)
    	        for n=0:2
    	        	b = BranchChannel(1,rc_self,rc_parent,rc_children,n)
    	        	@test b isa BranchChannel{Int,Int}
    	        	@test b.p == 1
    	        	@test b.selfchannels == rc_self
    	        	@test b.parentchannels == rc_parent
    	        	@test b.childrenchannels == rc_children
                    @test nchildren(b) == b.nchildren == n
    	        end
    	        @test_throws ParallelUtilities.DomainError BranchChannel(1,rc_self,rc_parent,rc_children,3)
	    	end

	    	@testset "only parent channels supplied" begin
	    		rc_parent = RemoteChannelContainer{Int}(1)
	    		for n=0:2
    	        	b = BranchChannel(1,Int,rc_parent,n)
    	        	@test b isa BranchChannel{Int,Int}
    	        	@test b.p == 1
    	        	@test b.parentchannels == rc_parent
    	        	@test b.selfchannels isa RemoteChannelContainer{Int}
    	        	@test b.childrenchannels isa RemoteChannelContainer{Int}
    	        	@test b.selfchannels.out.where == b.p
    	        	@test b.selfchannels.err.where == b.p
    	        	@test b.childrenchannels.out.where == b.p
    	        	@test b.childrenchannels.err.where == b.p
                    @test nchildren(b) == b.nchildren == n
    	        end
	        	@test_throws ParallelUtilities.DomainError BranchChannel(1,Int,rc_parent,3)
	    	end

	    	@testset "no channels supplied" begin
	    		function testbranchchannel(b::BranchChannel{T,T},p,n) where {T}
    	        	@test b.p == p
    	        	@test b.parentchannels isa RemoteChannelContainer{T}
    	        	@test b.selfchannels isa RemoteChannelContainer{T}
    	        	@test b.childrenchannels isa RemoteChannelContainer{T}
    	        	@test b.parentchannels.out.where == b.p
    	        	@test b.parentchannels.err.where == b.p
    	        	@test b.selfchannels.out.where == b.p
    	        	@test b.selfchannels.err.where == b.p
    	        	@test b.childrenchannels.out.where == b.p
    	        	@test b.childrenchannels.err.where == b.p
                    @test nchildren(b) == b.nchildren == n
	    		end

	    		p = workers()[1]
	    	    for n=0:2
    	        	b = BranchChannel{Int,Int}(p,n)
    	        	testbranchchannel(b,p,n)
    	        end
	        	@test_throws ParallelUtilities.DomainError BranchChannel{Int,Int}(1,3)
                @test_throws ParallelUtilities.DomainError BranchChannel{Int,Int}(1,-1)
	    	end
	    end

	    @testset "finalize" begin
	    	@testset "sameprocessor" begin
    	        parentchannels = RemoteChannelContainer{Int}(1)
    	        b = BranchChannel(1,Int,parentchannels,1)
    	        finalize(b)
    	        @test b.selfchannels.out.where == 0
    	        @test b.selfchannels.err.where == 0
	        	@test b.childrenchannels.out.where == 0
	        	@test b.childrenchannels.err.where == 0
	        	@test b.parentchannels.out.where == myid()
	        	@test b.parentchannels.err.where == myid()
	    	end
	    	@testset "elsewhere" begin
	    		p = workers()[1]
    	        selfchannels = RemoteChannelContainer{Int}(1,p)
    	        childrenchannels = RemoteChannelContainer{Int}(1,p)
	    		
	    		@testset "parent == whence == where == myid()" begin
	    	        parentchannels = RemoteChannelContainer{Int}(1)
	    	        b = BranchChannel(1,selfchannels,parentchannels,childrenchannels,1)
	    	        self_w,parent_w,child_w = @fetchfrom p begin
	    	        	finalize(b)
	    	        	(b.selfchannels.out.where,b.selfchannels.err.where),
	    	        	(b.parentchannels.out.where,b.parentchannels.err.where),
	    	        	(b.childrenchannels.out.where,b.childrenchannels.err.where)
	    	    	end
	    	    	@test self_w == (0,0)
		        	@test child_w == (0,0)
		        	@test parent_w == (0,0)
	    		end

	    		@testset "(parent == where) != (whence == myid())" begin
		        	parentchannels = RemoteChannelContainer{Int}(1,p)
	    	        b = BranchChannel(1,selfchannels,parentchannels,childrenchannels,1)
	    	        self_w,parent_w,child_w = @fetchfrom p begin
	    	        	finalize(b)
	    	        	(b.selfchannels.out.where,b.selfchannels.err.where),
	    	        	(b.parentchannels.out.where,b.parentchannels.err.where),
	    	        	(b.childrenchannels.out.where,b.childrenchannels.err.where)
	    	    	end
	    	    	@test self_w == (0,0)
		        	@test child_w == (0,0)
		        	@test parent_w == (p,p)
		        end
	    	end
	    end

	    @testset "createbranchchannels" begin
	        function testbranches(T,tree)
    	        branches = createbranchchannels(T,T,tree)
    	        @test length(branches) == length(tree)
    	        for (rank,branch) in enumerate(branches)
    	        	parentrank = parentnoderank(tree,rank)
    	        	p = branch.p
    	        	p_parent = branches[parentrank].p
    	        	@test branch.selfchannels.out.where == p
    	        	@test branch.selfchannels.err.where == p
    	        	@test branch.childrenchannels.out.where == p
    	        	@test branch.childrenchannels.err.where == p
    	        	@test branch.parentchannels.out.where == p_parent
    	        	@test branch.parentchannels.err.where == p_parent
    	        end
    	    end
	        
	        for TreeType in [SequentialBinaryTree,
                OrderedBinaryTree,
                SegmentedSequentialBinaryTree]

                tree = TreeType(workers())
                for T in [Int,Any,Bool,Vector{Float64},Array{ComplexF64,2}]
                    testbranches(T,tree)
                end
	        end

	        iterators = (1:nworkers()+1,)
	        tree,branches = createbranchchannels(iterators,SequentialBinaryTree)
	        @test eltype(first(branches).parentchannels) == Any
            tree,branches = createbranchchannels(iterators,SegmentedSequentialBinaryTree)
            @test eltype(first(branches).parentchannels) == Any
            tree,branches = createbranchchannels(iterators,OrderedBinaryTree)
            @test eltype(first(branches).parentchannels) == Any
            tree,branches = createbranchchannels(Int,Int,iterators,SequentialBinaryTree)
            @test eltype(first(branches).parentchannels) == Int
            tree,branches = createbranchchannels(Int,Int,iterators,SegmentedSequentialBinaryTree)
            @test eltype(first(branches).parentchannels) == Int
	        tree,branches = createbranchchannels(Int,Int,iterators,OrderedBinaryTree)
	        @test eltype(first(branches).parentchannels) == Int

            # Make sure that all branches are defined
            for T in [SequentialBinaryTree,
                OrderedBinaryTree,
                SegmentedSequentialBinaryTree]

                for nmax = 1:nworkers()
                    iterators = (1:nmax,)
                    tree,branches = createbranchchannels(iterators,T)
                    for i in eachindex(branches)
                        @test isassigned(branches,i)
                    end
                end
            end
	    end
	end
end;

@testset "pmap and reduce" begin

	exceptiontype = RemoteException
	if VERSION >= v"1.3"
		exceptiontype = CompositeException
	end

	@testset "Sorted and Unsorted" begin
	    @test Sorted() isa Ordering
	    @test Unsorted() isa Ordering
	end;

	@testset "pval" begin
		p = pval(2,3)
	    @test value(p) == 3
	    @test value(3) == 3
	    @test value(p) == value(value(p))

        @test convert(pval{Any},p) == pval{Any}(2,3)
        @test convert(pval{Float64},p) == pval{Any}(2,3.0)
	end;

	@testset "mapTreeNode" begin

		@testset "maybepvalput!" begin
		    pipe = BranchChannel{Int,Int}(myid(),0)
            rank = 1
		    maybepvalput!(pipe,rank,0)
		    @test isready(pipe.selfchannels.out)
		    @test take!(pipe.selfchannels.out) == 0

		    pipe = BranchChannel{pval,pval}(myid(),0)
		    maybepvalput!(pipe,rank,0)
		    @test isready(pipe.selfchannels.out)
		    @test take!(pipe.selfchannels.out) == pval(rank,0)

            pipe = BranchChannel{pval{Int},pval{Int}}(myid(),0)
            maybepvalput!(pipe,rank,0)
            @test isready(pipe.selfchannels.out)
            @test take!(pipe.selfchannels.out) == pval(rank,0)

            T = Vector{ComplexF64}
            pipe = BranchChannel{pval{T},pval{T}}(myid(),1)

            val = ones(1).*im
            maybepvalput!(pipe,rank,val)
            @test isready(pipe.selfchannels.out)
            @test take!(pipe.selfchannels.out) == pval(rank,ComplexF64[im])

            val = ones(1)
            maybepvalput!(pipe,rank,val)
            @test isready(pipe.selfchannels.out)
            @test take!(pipe.selfchannels.out) == pval(rank,ComplexF64[1])

            T = Vector{Float64}
            pipe = BranchChannel{pval{T},pval{T}}(myid(),1)

            val = ones(1)
            maybepvalput!(pipe,rank,val)
            @test isready(pipe.selfchannels.out)
            @test take!(pipe.selfchannels.out) == pval(rank,Float64[1])

            val = ones(Int,1)
            maybepvalput!(pipe,rank,val)
            @test isready(pipe.selfchannels.out)
            @test take!(pipe.selfchannels.out) == pval(rank,Float64[1])

            pipe = BranchChannel{pval,pval}(myid(),1)

            val = ones(1)
            maybepvalput!(pipe,rank,val)
            @test isready(pipe.selfchannels.out)
            @test take!(pipe.selfchannels.out) == pval(rank,Float64[1])

            val = ones(Int,1)
            maybepvalput!(pipe,rank,val)
            @test isready(pipe.selfchannels.out)
            @test take!(pipe.selfchannels.out) == pval(rank,Int[1])
		end

		function test_on_pipe(fn,iterator,pipe,result_expected)
            progressrc = nothing
            rank = 1
			@test_throws ErrorException mapTreeNode(x->error(""),iterator,rank,pipe,progressrc)
			@test !isready(pipe.selfchannels.out) # should not have any result as there was an error
			@test isready(pipe.selfchannels.err)
			@test take!(pipe.selfchannels.err) # error flag should be true
			@test !isready(pipe.selfchannels.err) # should not hold anything now
			@test !isready(pipe.parentchannels.out)
			@test !isready(pipe.parentchannels.err)
			@test !isready(pipe.childrenchannels.out)
			@test !isready(pipe.childrenchannels.err)

			mapTreeNode(fn,iterator,rank,pipe,progressrc)
			@test isready(pipe.selfchannels.err)
			@test !take!(pipe.selfchannels.err) # error flag should be false
			@test !isready(pipe.selfchannels.err)
			@test isready(pipe.selfchannels.out)
			@test take!(pipe.selfchannels.out) == result_expected
			@test !isready(pipe.selfchannels.out)
			@test !isready(pipe.parentchannels.out)
			@test !isready(pipe.parentchannels.err)
			@test !isready(pipe.childrenchannels.out)
			@test !isready(pipe.childrenchannels.err)
		end

		@testset "range" begin
	 		iterator = 1:10
			
			pipe = BranchChannel{Int,Int}(myid(),0)
			test_on_pipe(sum,iterator,pipe,sum(iterator))
		end
		
		@testset "ProductSplit" begin
			iterators = (1:10,)
			ps = ProductSplit(iterators,1,1)

			pipe = BranchChannel{Int,Int}(myid(),0)
			test_on_pipe(x->sum(y[1] for y in x),ps,pipe,sum(iterators[1]))

			pipe = BranchChannel{Int,Int}(myid(),1)
			test_on_pipe(x->sum(y[1] for y in x),ps,pipe,sum(iterators[1]))

			pipe = BranchChannel{Int,Int}(myid(),2)
			test_on_pipe(x->sum(y[1] for y in x),ps,pipe,sum(iterators[1]))
		end

        @testset "progress" begin
            @test isnothing(ParallelUtilities.indicatemapprogress!(nothing,1))
            rettype = Tuple{Bool,Bool,Int}
            progress = RemoteChannel(()->Channel{rettype}(1))
            ParallelUtilities.indicatemapprogress!(progress,10)
            @test take!(progress) == (true,false,10)
        end
	end;

	@testset "reduce" begin

		# Leaves just push results to the parent
		# reduced value at a leaf is simply whatever is stored in the local output channel
		@testset "at a leaf" begin
			# These do not check for errors
            result = 1
            rank = 1
            val = pval(rank,result)

            pipe = BranchChannel{typeof(val),typeof(val)}(myid(),0)
            put!(pipe.selfchannels.out,val)
            @test ParallelUtilities.reducedvalue(sum,rank,pipe,Sorted()) == val

            pipe = BranchChannel{typeof(result),typeof(result)}(myid(),0)
            put!(pipe.selfchannels.out,result)
            @test ParallelUtilities.reducedvalue(sum,rank,pipe,Unsorted()) == result
		end;

		# # Values are collected at the intermediate nodes
		@testset "at parent nodes" begin

			# Put some known values on the self and children channels
			function putselfchildren!(pipe::BranchChannel,::Unsorted,rank=1)
                if rank >= 1
    		    	put!(pipe.selfchannels.out,0)
    		    	put!(pipe.selfchannels.err,false)
                end
		    	for i=1:nchildren(pipe)
		    		put!(pipe.childrenchannels.out,i)
		    		put!(pipe.childrenchannels.err,false)
		    	end
		    end
		    function putselfchildren!(pipe::BranchChannel{<:pval},::Sorted,rank=1)
		    	put!(pipe.selfchannels.out,pval(2,2))
		    	put!(pipe.selfchannels.err,false)
                N = nchildren(pipe)
		    	
                if N > 0
                    # left child
    		    	put!(pipe.childrenchannels.out,pval(1,1))
    		    	put!(pipe.childrenchannels.err,false)
                end

                if N > 1
                    # right child
                    put!(pipe.childrenchannels.out,pval(3,3))
                    put!(pipe.childrenchannels.err,false)
                end
		    end

		    function clearerrors!(pipe::BranchChannel,rank=1)
                if rank >= 1
                    take!(pipe.selfchannels.err)
                end
		    	for i=1:nchildren(pipe)
		    		take!(pipe.childrenchannels.err)
		    	end
		    end

			@testset "reducedvalue" begin

				function testreduction(freduce::Function,pipe::BranchChannel,
		    		ifsorted::Ordering,res_exp,rank=2)

			    	p = pipe.p

			    	try
				    	putselfchildren!(pipe,ifsorted,rank)
						@test value(reducedvalue(freduce,rank,pipe,ifsorted)) == res_exp
						clearerrors!(pipe,rank)
				    	
				  		@fetchfrom p putselfchildren!(pipe,ifsorted,rank)
						@test value(@fetchfrom p reducedvalue(freduce,rank,pipe,ifsorted)) == res_exp
						clearerrors!(pipe,rank)
						
						@fetchfrom p putselfchildren!(pipe,ifsorted,rank)
						@test value(reducedvalue(freduce,rank,pipe,ifsorted)) == res_exp
						clearerrors!(pipe,rank)
						
						putselfchildren!(pipe,ifsorted,rank)
						@test value(@fetchfrom p reducedvalue(freduce,rank,pipe,ifsorted)) == res_exp
						clearerrors!(pipe,rank)
					catch
						rethrow()
					end
				end

			    for nchildren = 1:2
			    	@testset "Unsorted" begin
			            pipe = BranchChannel{Int,Int}(myid(),nchildren)
			            res_exp = sum(0:nchildren)
			            testreduction(sum,pipe,Unsorted(),res_exp,2)
                        testreduction(sum,pipe,Unsorted(),res_exp,0)
			        end
			    	@testset "Sorted" begin
			            pipe = BranchChannel{pval,pval}(myid(),nchildren)
			            res_exp = collect(1:nchildren+1)
			            testreduction(x->vcat(x...),pipe,Sorted(),res_exp)
			    
			            pipe = BranchChannel{pval,pval}(myid(),nchildren)
			            res_exp = sum(1:nchildren+1)
			            testreduction(sum,pipe,Sorted(),res_exp)
			    	end
			    end

                # The top tree must have children by definition
                pipe = BranchChannel{Int,Int}(myid(),0)
                putselfchildren!(pipe,Unsorted(),0)
                err = ErrorException("nodes with rank <=0 must have children")
                @test_throws err reducedvalue(sum,0,pipe,Unsorted())
                clearerrors!(pipe,0)
			end

			@testset "reduceTreeNode" begin

		    	function testreduction(freduce::Function,pipe::BranchChannel,
		    		ifsorted::Ordering,res_exp)

		    		@test !isready(pipe.parentchannels.out)
		    		@test !isready(pipe.parentchannels.err)

                    progressrc = nothing
                    rank = 2

		    		try
			    		wait(@spawnat pipe.p putselfchildren!(pipe,ifsorted))
			    		reduceTreeNode(freduce,rank,pipe,ifsorted,progressrc)
			    	catch
			    		rethrow()
			    	end
					@test isready(pipe.parentchannels.out)
					@test isready(pipe.parentchannels.err)
					@test !take!(pipe.parentchannels.err) # there should be no error
					@test value(take!(pipe.parentchannels.out)) == res_exp

					# The pipe should be finalized at this point
					@test pipe.selfchannels.out.where == 0
					@test pipe.selfchannels.err.where == 0
					@test pipe.childrenchannels.out.where == 0
					@test pipe.childrenchannels.err.where == 0
		    	end

		    	for nchildren = 1:2
			    	@testset "Unsorted" begin
			            pipe = BranchChannel{Int,Int}(myid(),nchildren)
			            res_exp = sum(0:nchildren)
			            testreduction(sum,pipe,Unsorted(),res_exp)

			            rc_parent = RemoteChannelContainer{Int}(1)
			            p = workers()[1]
			            pipe = BranchChannel(p,Int,rc_parent,nchildren)
			            testreduction(sum,pipe,Unsorted(),res_exp)
			        end
			    	@testset "Sorted" begin
			            pipe = BranchChannel{pval,pval}(myid(),nchildren)
			            res_exp = collect(1:nchildren+1)
			            testreduction(x->vcat(x...),pipe,Sorted(),res_exp)

			            rc_parent = RemoteChannelContainer{pval}(myid(),1)
			            p = workers()[1]
			            pipe = BranchChannel(p,pval,rc_parent,nchildren)
			            testreduction(x->vcat(x...),pipe,Sorted(),res_exp)
			    
			            pipe = BranchChannel{pval,pval}(myid(),nchildren)
			            res_exp = sum(1:nchildren+1)
			            testreduction(sum,pipe,Sorted(),res_exp)

			            rc_parent = RemoteChannelContainer{pval}(1)
			            p = workers()[1]
			            pipe = BranchChannel(p,pval,rc_parent,nchildren)
			            testreduction(sum,pipe,Sorted(),res_exp)
			    	end
			    end
		    end
		end;

        @testset "progress" begin
            @test isnothing(ParallelUtilities.indicatereduceprogress!(nothing,1))
            rettype = Tuple{Bool,Bool,Int}
            progress = RemoteChannel(()->Channel{rettype}(1))
            ParallelUtilities.indicatereduceprogress!(progress,10)
            @test take!(progress) == (false,true,10)

            @test isnothing(ParallelUtilities.indicatefailure!(nothing,1))
            ParallelUtilities.indicatefailure!(progress,10)
            @test take!(progress) == (false,false,10)
        end
	end;

	@testsetwithinfo "pmapbatch" begin
		@testsetwithinfo "batch" begin
			@testset "comparison with map" begin
				iterable = 1:nworkers()
			    res = pmapbatch(x->myid(),iterable)
			    @test res == workers()
			    res = pmapbatch(x->myid(),(iterable,))
			    @test res == workers()
			    res = pmapbatch(x->myid(),(iterable,1:1))
			    @test res == workers()
			    res = pmapbatch(x->myid(),iterable,num_workers=1)
			    @test res == workers()[1:1]

			    iterable = 1:nworkers()-1
			    res = pmapbatch(x->myid(),iterable)
			    @test res == workersactive(iterable)

			    iterable = 1:nworkers()
			    res = pmapbatch(identity,iterable)
			    resexp = [ProductSplit((iterable,),nworkersactive(iterable),p) for p=1:nworkersactive(iterable)]
				@test res == resexp
			    
			    iterable = 1:nworkers()
			    res = pmapbatch(identity,iterable)
			    resexp = [ProductSplit((iterable,),nworkers(),p) for p=1:nworkers()]
				@test res == resexp
			    
			    iterable = 1:2nworkers()
			    res = pmapbatch(identity,iterable)
			    resexp = [ProductSplit((iterable,),nworkersactive(iterable),p) for p=1:nworkersactive(iterable)]
				@test res == resexp			    
			end

			@testset "errors" begin
			    @test_throws exceptiontype pmapbatch(x->throw(BoundsError()),1:10)
			end
		end
		
		@testsetwithinfo "elementwise" begin
			@testset "comparison with map" begin
			    iterable = 1:nworkers()
			    res = pmapbatch_elementwise(identity,iterable)
			    @test res == iterable

                res = pmapbatch_elementwise(identity,iterable,num_workers=1)
                @test res == iterable

				iterable = 1:20
			    res = pmapbatch_elementwise(x->x^2,iterable)
			    @test res == iterable.^2
			end

		    @testset "errors" begin
			    @test_throws exceptiontype pmapbatch_elementwise(x->throw(BoundsError()),1:10)
			end
		end
	end;

	@testsetwithinfo "pmapsum" begin
		@testsetwithinfo "batch" begin
		    @testset "rank" begin
                res_exp = sum(1:nworkers())
                @testset "without progress" begin
                    res = pmapsum(x->x[1][1],Int,1:nworkers())
    			    @test res == res_exp
                    res = pmapsum(x->x[1][1],1:nworkers())
                    @test res == res_exp
                end
                @testset "with progress" begin
                    res = pmapsum(x->x[1][1],Int,1:nworkers(),showprogress=true)
                    @test res == res_exp
                    res = pmapsum(x->x[1][1],1:nworkers(),showprogress=true)
                    @test res == res_exp
                end
			    @test pmapsum(x->x[1][1],Int,(1:nworkers(),)) == res_exp
                @test pmapsum(x->x[1][1],(1:nworkers(),)) == res_exp
                @test pmapsum(x->x[1][1],Int,(1:nworkers(),1:1)) == res_exp
			    @test pmapsum(x->x[1][1],(1:nworkers(),1:1)) == res_exp
			    @test pmapsum(x->myid(),1:nworkers()) == sum(workers())
		    end
		    
		    @testset "one iterator" begin
			    rng = 1:100
			    @test pmapsum(x->sum(y[1] for y in x),rng) == sum(rng)
			    @test pmapsum(x->sum(y[1] for y in x),(rng,)) == sum(rng)
		    end

		    @testset "array" begin
		    	@test pmapsum(x->ones(2),1:nworkers()) == ones(2).*nworkers()
		    end

		    @testset "stepped iterator" begin
			    rng = 1:5:100
			    @test pmapsum(x->sum(y[1] for y in x),rng) == sum(rng)
		    end

		    @testset "two iterators" begin
			    iters = (1:100,1:2)
			    @test pmapsum(x->sum(y[1] for y in x),iters) == sum(iters[1])*length(iters[2])
		    end
		    
		    @testsetwithinfo "run elsewhere" begin
		    	res_exp = sum(workers())
		    	for (ind,p) in enumerate(workers())
                    showworkernumber(ind,nworkers())
		    		res = @fetchfrom p pmapsum(x->myid(),1:nworkers())
		        	@test res == res_exp
		        end
		    end

		    @testset "errors" begin
		        @test_throws exceptiontype pmapsum(x->error("map"),1:10)
                @test_throws exceptiontype pmapsum(x->fmap(x),1:10)
		    end
		end

		@testsetwithinfo "elementwise" begin
			@testset "comparison with map" begin
			    iterable = 1:100
                @testset "without progress" begin
                    res = pmapsum_elementwise(identity,iterable)
                    @test res == sum(iterable)
                end
                @testset "with progress" begin
                    res = pmapsum_elementwise(identity,iterable,showprogress=true)
                    @test res == sum(iterable) 
                end
                res = pmapsum_elementwise(identity,(iterable,))
                @test res == sum(iterable)
			    res = pmapsum_elementwise(identity,Int,iterable)
			    @test res == sum(iterable)
			    res = pmapsum_elementwise(identity,Int,(iterable,))
			    @test res == sum(iterable)

			    iterable = 1:100
			    res = pmapsum_elementwise(x->x^2,iterable)
			    @test res == sum(x->x^2,iterable)
			    @test res == pmapsum(plist->sum(x[1]^2 for x in plist),iterable)
			end

			@testset "run elsewhere" begin
				iterable = 1:100
				res_exp = sum(iterable)
		    	for (ind,p) in enumerate(workers())
                    showworkernumber(ind,nworkers())
		    		res = @fetchfrom p pmapsum_elementwise(identity,iterable)
		        	@test res == res_exp
		        end
		    end

		    @testset "errors" begin
		        @test_throws exceptiontype pmapsum_elementwise(x->error("hi"),1:10)
		    end
		end

        @testset "type coercion" begin
            @test_throws exceptiontype pmapsum(x->[1.1],Vector{Int},1:nworkers())
            @test pmapsum(x->ones(2).*myid(),Vector{Int},1:nworkers()) isa Vector{Int}
        end
	end;

	@testsetwithinfo "pmapreduce_commutative" begin
	    @testsetwithinfo "batch" begin
			@testset "sum" begin
                res_exp = sum(workers())
                @testset "without progress" begin
                    res = pmapreduce_commutative(x->myid(),Int,sum,Int,1:nworkers())
                    @test res == res_exp
                    res = pmapreduce_commutative(x->myid(),sum,1:nworkers())
                    @test res == res_exp
                end
                @testset "with progress" begin
                    res = pmapreduce_commutative(x->myid(),Int,sum,Int,1:nworkers(),showprogress=true)
    			    @test res == res_exp
                    res = pmapreduce_commutative(x->myid(),sum,1:nworkers(),showprogress=true)
                    @test res == res_exp
                end
                @test pmapreduce_commutative(x->myid(),Int,sum,Int,(1:nworkers(),)) == res_exp
			    @test pmapreduce_commutative(x->myid(),sum,(1:nworkers(),)) == res_exp
			    @test pmapreduce_commutative(x->myid(),Int,sum,Int,(1:nworkers(),1:1)) == res_exp
                @test pmapreduce_commutative(x->myid(),sum,(1:nworkers(),1:1)) == res_exp
			    @test pmapreduce_commutative(x->myid(),sum,1:nworkers()) == pmapsum(x->myid(),1:nworkers())
		    end
		    @testset "prod" begin
			    @test pmapreduce_commutative(x->myid(),prod,1:nworkers()) == prod(workers())
			    @test pmapreduce_commutative(x->myid(),prod,(1:nworkers(),)) == prod(workers())
			    @test pmapreduce_commutative(x->myid(),prod,(1:nworkers(),1:1)) == prod(workers())
		    end

		    @testsetwithinfo "run elsewhere" begin
		    	res_exp = prod(workers())
		    	for (ind,p) in enumerate(workers())
                    showworkernumber(ind,nworkers())
		    		res = @fetchfrom p pmapreduce_commutative(x->myid(),prod,1:nworkers())
		        	@test res == res_exp
		        end
		    end

		    @testset "errors" begin
		        @test_throws exceptiontype pmapreduce_commutative(
												x->error("map"),sum,1:10)
		        @test_throws exceptiontype pmapreduce_commutative(
												identity,x->error("reduce"),1:10)
				@test_throws exceptiontype pmapreduce_commutative(
												x->error("map"),x->error("reduce"),1:10)

                @test_throws exceptiontype pmapreduce_commutative(
                                                x->fmap("map"),sum,1:10)
                @test_throws exceptiontype pmapreduce_commutative(
                                                x->1,x->fred(x),1:10)
                @test_throws exceptiontype pmapreduce_commutative(
                                                x->fmap(x),x->fred(x),1:10)
		    end

            @testset "type coercion" begin
                @test_throws exceptiontype pmapreduce_commutative(x->[1.1],Vector{Int},
                                                sum,Vector{Int},1:nworkers())
                res = pmapreduce_commutative(x->ones(2).*myid(),Vector{Int},sum,Vector{Int},1:nworkers())
                @test res isa Vector{Int}
            end
		end;

		@testsetwithinfo "elementwise" begin
			@testset "comparison with map" begin
			    iter = 1:1000
                res_exp = sum(x->x^2,iter)
                @testset "without progress" begin
                    res = pmapreduce_commutative_elementwise(x->x^2,sum,iter)
                    @test res == res_exp
                end
                @testset "with progress" begin
    			    res = pmapreduce_commutative_elementwise(x->x^2,sum,iter,showprogress=true)
    			    @test res == res_exp
                end
			    @test res == pmapsum_elementwise(x->x^2,iter)
			    @test res == pmapsum(plist->sum(x[1]^2 for x in plist),iter)
			    res = pmapreduce_commutative_elementwise(x->x^2,sum,(iter,))
			    @test res == res_exp
                res = pmapreduce_commutative_elementwise(x->x^2,Int,sum,Int,(iter,))
                @test res == res_exp
                res = pmapreduce_commutative_elementwise(x->x^2,Int,sum,Int,iter)
                @test res == res_exp
                res = pmapreduce_commutative_elementwise(x->x^2,Int,x->float(sum(x)),Float64,iter)
                @test res == float(res_exp)
			end

			@testsetwithinfo "run elsewhere" begin
				iter = 1:1000
				res_exp = sum(x->x^2,iter)
		    	for (ind,p) in enumerate(workers())
                    showworkernumber(ind,nworkers())
		    		res = @fetchfrom p pmapreduce_commutative_elementwise(x->x^2,sum,iter)
		        	@test res == res_exp
		        end
		    end

			@testsetwithinfo "errors" begin
				@test_throws exceptiontype pmapreduce_commutative_elementwise(
												x->error("map"),sum,1:10)
				@test_throws exceptiontype pmapreduce_commutative_elementwise(
												identity,x->error("reduce"),1:10)
				@test_throws exceptiontype pmapreduce_commutative_elementwise(
												x->error("map"),
												x->error("reduce"),1:10)
			end
		end;
	end;
    
	@testsetwithinfo "pmapreduce" begin
		@testsetwithinfo "batch" begin
		    @testset "sum" begin
                res_exp = sum(workers())
                @testset "without progress" begin
                    @test pmapreduce(x->myid(),Int,sum,Int,1:nworkers()) == res_exp
                    @test pmapreduce(x->myid(),sum,1:nworkers()) == res_exp
                end
                @testset "without progress" begin
                    res = pmapreduce(x->myid(),Int,sum,Int,1:nworkers(),showprogress=true)
                    @test res == res_exp
                    res = pmapreduce(x->myid(),sum,1:nworkers(),showprogress=true)
                    @test res == res_exp
                end
			    @test pmapreduce(x->myid(),Int,sum,Int,(1:nworkers(),)) == res_exp
			    @test pmapreduce(x->myid(),sum,(1:nworkers(),)) == res_exp
                @test pmapreduce(x->myid(),Int,sum,Int,(1:nworkers(),1:1)) == res_exp
			    @test pmapreduce(x->myid(),sum,(1:nworkers(),1:1)) == res_exp

                @testset "comparison with pmapsum" begin
                    res_exp = pmapsum(x->myid(),1:nworkers())
                    @test pmapreduce(x->myid(),Int,sum,Int,1:nworkers()) == res_exp
                    @test pmapreduce(x->myid(),sum,1:nworkers()) == res_exp
                end
		    end;

		    @testset "concatenation" begin
                res_vcat = ones(2*nworkers())
                res_hcat = ones(2,nworkers())
			    @test pmapreduce(x->ones(2),Vector{Float64},
                    x->vcat(x...),Vector{Float64},1:nworkers()) == res_vcat
                @test pmapreduce(x->ones(2),x->vcat(x...),1:nworkers()) == res_vcat
			    @test pmapreduce(x->ones(2),Vector{Float64},
                    x->hcat(x...),Matrix{Float64},1:nworkers()) == res_hcat
                @test pmapreduce(x->ones(2),x->hcat(x...),1:nworkers()) == res_hcat

                @testset "sorting" begin
                    @test pmapreduce(x->ones(2)*x[1][1],x->vcat(x...),1:nworkers()) == 
                            vcat((ones(2).*i for i=1:nworkers())...)

                    @test pmapreduce(x->x[1][1],x->vcat(x...),1:nworkers()) == collect(1:nworkers())
                    @test pmapreduce(x->myid(),Int,x->vcat(x...),Vector{Int},(1:nworkers(),)) == workers()
                    @test pmapreduce(x->myid(),x->vcat(x...),1:nworkers()) == workers()
                end
		    end;

			@testsetwithinfo "run elsewhere" begin
                @testsetwithinfo "sum" begin
    				res_exp = sum(workers())
    		    	for (ind,p) in enumerate(workers())
                        showworkernumber(ind,nworkers())
    		    		res = @fetchfrom p pmapreduce(x->myid(),sum,1:nworkers())
    		        	@test res == res_exp
    		        end
                end
                # concatenation where the rank is used in the mapping function
                # Preserves order of the iterators
                @testsetwithinfo "concatenation using rank" begin
                    res_exp = collect(1:nworkers())
                    for (ind,p) in enumerate(workers())
                        showworkernumber(ind,nworkers())
                        res = @fetchfrom p pmapreduce(x->x[1][1],x->vcat(x...),1:nworkers())
                        @test res == res_exp
                    end
                end
		    end;

			@testset "errors" begin
			    @test_throws exceptiontype pmapreduce(x->error("map"),sum,1:10)
				@test_throws exceptiontype pmapreduce(identity,x->error("reduce"),1:10)
				@test_throws exceptiontype pmapreduce(x->error("map"),x->error("reduce"),1:10)
                @test_throws exceptiontype pmapreduce(x->fmap(x),sum,1:10)
                @test_throws exceptiontype pmapreduce(x->1,x->fred(x),1:10)
                @test_throws exceptiontype pmapreduce(x->fmap(x),x->fred(x),1:10)
			end;

            @testset "type coercion" begin
                @test_throws exceptiontype pmapreduce(x->[1.1],Vector{Int},sum,Vector{Int},1:nworkers())
                @test pmapreduce(x->ones(2).*myid(),Vector{Int},sum,Vector{Int},1:nworkers()) isa Vector{Int}
            end;
		end;
	end;
end;

@testset "show" begin

    @testset "error" begin
        io = IOBuffer()
        
        showerror(io,ParallelUtilities.ProcessorNumberError(5,2))
        strexp = "processor id 5 does not lie in the range 1:2"
        @test String(take!(io)) == strexp

        showerror(io,ParallelUtilities.DecreasingIteratorError())
        strexp = "all the iterators need to be strictly increasing"
        @test String(take!(io)) == strexp

        showerror(io,ParallelUtilities.TaskNotPresentError((1:4,),(5,)))
        strexp = "could not find the task $((5,)) in the list $((1:4,))"
        @test String(take!(io)) == strexp
    end;

    @testset "BranchChannel" begin
        io = IOBuffer()

        b = BranchChannel{Any,Any}(1,0)
        show(io,b)
        strexp = "Leaf  : 1 ← 1"
        @test String(take!(io)) == strexp

        b = BranchChannel{Any,Any}(1,1)
        show(io,b)
        strexp = "Branch: 1 ← 1 ← 1 child"
        @test String(take!(io)) == strexp

        b = BranchChannel{Any,Any}(1,2)
        show(io,b)
        strexp = "Branch: 1 ← 1 ⇇ 2 children"
        @test String(take!(io)) == strexp
    end;

    @testset "BinaryTreeNode" begin
        io = IOBuffer()
        b = BinaryTreeNode(2,3,1)
        show(io,b)
        strexp = "BinaryTreeNode(p = 2, parent = 3, nchildren = 1)"
        @test String(take!(io)) == strexp
    end
end;