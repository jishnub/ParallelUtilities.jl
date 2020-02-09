using ParallelUtilities
using Test
using Distributed

addprocs(2)
@everywhere begin
	using Pkg
    Pkg.activate(".")
    using ParallelUtilities
end

@testset "ProductSplit" begin

	function split_across_processors_iterators(arr::Base.Iterators.ProductIterator,num_procs,proc_id)

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
	    	ntasks_total = prod(map(length,iters))
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

    	@testset "empty" begin
    	    iters = (1:1,)
    	    ps = ProductSplit(iters,10,2)
    	    @test isempty(ps)
    	    @test length(ps) == 0
    	end

    	@testset "first and last ind" begin
    	    iters = (1:10,)
    	    ps = ProductSplit(iters,2,1)
    	    @test ps.firstind == 1
    	    @test ps.lastind == div(length(iters[1]),2)
    	    ps = ProductSplit(iters,2,2)
    	    @test ps.firstind == div(length(iters[1]),2) + 1
    	    @test ps.lastind == length(iters[1])

    	    for np in length(iters[1])+1:length(iters[1])+10,
    	    	p in length(iters[1])+1:np

	    	    ps = ProductSplit(iters,np,p)
	    	    @test ps.firstind == length(iters[1]) + 1
	    	    @test ps.lastind == length(iters[1])
	    	end
    	end
    end

    @testset "firstlast" begin
        @testset "first" begin

        	@test ParallelUtilities._first(()) == ()

            for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)],
            	np=1:5ntasks(iters)

	            ps = ProductSplit(iters,np,1)
	            @test first(ps) == ( isempty(ps) ? nothing : map(first,iters) )
	        end

	        iters = (1:1,)
	        ps = ProductSplit(iters,2length(iters[1]),length(iters[1])+1) # must be empty
	        @test first(ps) === nothing
        end
        @testset "last" begin

        	@test ParallelUtilities._last(()) == ()

            for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)],
            	np=1:5ntasks(iters)

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
			        		res = fn(ps,dim) == fn(x[dim] for x in pcol)
			        		if !res
			        			println("ProductSplit(",iters,",",np,",",p,")")
			        		end
			        		res
			        	end
			        end
			    end
			end

		    for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)],
		    	fn in [maximum,minimum,extrema]

		        checkPSextrema(iters,fn)
		    end
    	end

    	@testset "extremadims" begin
    		ps = ProductSplit((1:10,),2,1)
    		@test ParallelUtilities._extremadims(ps,1,()) == ()
    		for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)]
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

        for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)]
	        checkifpresent(iters)
	    end

	    @test ParallelUtilities._infullrange((),())
    end

    @testset "evenlyscatterproduct" begin
        n = 10
        np,proc_id = 5,3
        @test evenlyscatterproduct(n,np,proc_id) == ProductSplit((1:n,),np,proc_id)
        for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)]
        	@test evenlyscatterproduct(iters,np,proc_id) == ProductSplit(iters,np,proc_id)
        	itp = Iterators.product(iters...)
        	@test evenlyscatterproduct(itp,np,proc_id) == ProductSplit(iters,np,proc_id)
        end
    end

    @testset "whichproc + procrange_recast" begin
        np,proc_id = 5,3
        iters = (1:10,4:6,1:4)
        ps = ProductSplit(iters,np,proc_id)
        @test whichproc(iters,first(ps),1) == 1
        @test whichproc(iters,(100,100,100),1) === nothing
        @test procrange_recast(iters,ps,1) == 1:1
        @test procrange_recast(ps,1) == 1:1

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
        
        for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)]
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

    @testset "procid_and_localindex" begin
        for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)]
	        for np=1:ntasks(iters),proc_id=1:np
	        	ps_col = collect(ProductSplit(iters,np,proc_id))
	        	ps_col_rev = [reverse(t) for t in ps_col] 
	        	for val in ps_col
	        		p,ind = procid_and_localindex(iters,val,np)
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

        for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)]
            for np=1:ntasks(iters),p=1:np
            	ps = ProductSplit(iters,np,p)
            	ps_col = collect(ps)
            	for i in 1:length(ps)
            		@test ps[i] == ps_col[i]
            	end
            	@test_throws ParallelUtilities.BoundsErrorPS ps[0]
            	@test_throws ParallelUtilities.BoundsErrorPS ps[length(ps)+1]
            end
        end
    end
end

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
end

@testset "utilities" begin
    @testset "workerrank" begin
    	for (rank,workerid) in enumerate(workers())
	        @test @fetchfrom workerid myid() == workerid
	        @test @fetchfrom workerid workerrank() == rank
	    end
    end

    @testset "workers active" begin
        @test nworkersactive((1:1,)) == 1
        @test nworkersactive((1:2,)) == min(2,nworkers())
        @test nworkersactive((1:1,1:2)) == min(2,nworkers())
        @test nworkersactive(1:2) == min(2,nworkers())
        @test nworkersactive(1:1,1:2) == min(2,nworkers())
        @test nworkersactive((1:nworkers()+1,)) == 2
        @test nworkersactive(1:nworkers()+1) == 2
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
        itp = Iterators.product(iters...)
        @test nworkersactive(itp) == nworkersactive(iters)
        @test workersactive(itp) == workersactive(iters)

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
        npnodes = Dict(hostnames[1]=>nworkers())
        @test nprocs_node(hostnames,nodes) == npnodes
        @test nprocs_node(hostnames) == npnodes
        @test nprocs_node() == npnodes
    end
end

@testset "pmap and reduce" begin

	exceptiontype = RemoteException
	if VERSION >= v"1.3"
		exceptiontype = CompositeException
	end

	@testset "pmapbatch" begin
		@testset "batch" begin
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
			    
			    iterable = 1:2nworkers()
			    res = pmapbatch(identity,Iterators.product(iterable))
			    resexp = [ProductSplit((iterable,),nworkersactive(iterable),p) for p=1:nworkersactive(iterable)]
				@test res == resexp
			end

			@testset "errors" begin
			    @test_throws exceptiontype pmapbatch(x->throw(BoundsError()),1:10)
			end
		end
		
		@testset "elementwise" begin
			@testset "comparison with map" begin
			    iterable = 1:nworkers()
			    res = pmapbatch_elementwise(identity,iterable)
			    @test res == iterable

				iterable = 1:20
			    res = pmapbatch_elementwise(x->x^2,iterable)
			    @test res == iterable.^2
			end

		    @testset "errors" begin
			    @test_throws exceptiontype pmapbatch_elementwise(x->throw(BoundsError()),1:10)
			end
		end
	end

	@testset "pmapsum" begin
		@testset "batch" begin
		    @testset "worker id" begin
			    @test pmapsum(x->workerrank(),1:nworkers()) == sum(1:nworkers())
			    @test pmapsum(x->workerrank(),(1:nworkers(),)) == sum(1:nworkers())
			    @test pmapsum(x->workerrank(),Iterators.product(1:nworkers())) == sum(1:nworkers())
			    @test pmapsum(x->workerrank(),(1:nworkers(),1:1)) == sum(1:nworkers())
			    @test pmapsum(x->workerrank(),Iterators.product(1:nworkers(),1:1)) == sum(1:nworkers())
			    @test pmapsum(x->myid(),1:nworkers()) == sum(workers())
		    end
		    
		    @testset "one iterator" begin
			    rng = 1:100
			    @test pmapsum(x->sum(y[1] for y in x),rng) == sum(rng)
			    @test pmapsum(x->sum(y[1] for y in x),Iterators.product(rng)) == sum(rng)
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
		    
		    @testset "run elsewhere" begin
		    	res_exp = sum(workers())
		    	for p in workers()
		    		res = @fetchfrom p pmapsum(x->myid(),1:nworkers())
		        	@test res == res_exp
		        end
		    end

		    @testset "errors" begin
		        @test_throws exceptiontype pmapsum(x->throws(BoundsError()),1:10)
		    end
		end

		@testset "elementwise" begin
			@testset "comparison with map" begin
			    iterable = 1:100
			    res = pmapsum_elementwise(identity,iterable)
			    @test res == sum(iterable)
			    res = pmapsum_elementwise(identity,Iterators.product(iterable))
			    @test res == sum(iterable)
			    res = pmapsum_elementwise(identity,(iterable,))
			    @test res == sum(iterable)

			    iterable = 1:100
			    res = pmapsum_elementwise(x->x^2,iterable)
			    @test res == sum(x->x^2,iterable)
			    @test res == pmapsum(plist->sum(x[1]^2 for x in plist),iterable)
			end

			@testset "run elsewhere" begin
				iterable = 1:100
				res_exp = sum(iterable)
		    	for p in workers()
		    		res = @fetchfrom p pmapsum_elementwise(identity,iterable)
		        	@test res == res_exp
		        end
		    end

		    @testset "errors" begin
		        @test_throws exceptiontype pmapsum_elementwise(x->throws(BoundsError()),1:10)
		    end
		end
	end

	@testset "pmapreduce_commutative" begin
	    @testset "batch" begin
			@testset "sum" begin
			    @test pmapreduce_commutative(x->myid(),sum,1:nworkers()) == sum(workers())
			    @test pmapreduce_commutative(x->myid(),sum,(1:nworkers(),)) == sum(workers())
			    @test pmapreduce_commutative(x->myid(),sum,Iterators.product(1:nworkers())) == sum(workers())
			    @test pmapreduce_commutative(x->myid(),sum,(1:nworkers(),1:1)) == sum(workers())
			    @test pmapreduce_commutative(x->myid(),sum,Iterators.product(1:nworkers(),1:1)) == sum(workers())
			    @test pmapreduce_commutative(x->myid(),sum,1:nworkers()) == pmapsum(x->myid(),1:nworkers())
		    end
		    @testset "prod" begin
			    @test pmapreduce_commutative(x->myid(),prod,1:nworkers()) == prod(workers())
			    @test pmapreduce_commutative(x->myid(),prod,(1:nworkers(),)) == prod(workers())
			    @test pmapreduce_commutative(x->myid(),prod,Iterators.product(1:nworkers())) == prod(workers())
			    @test pmapreduce_commutative(x->myid(),prod,(1:nworkers(),1:1)) == prod(workers())
			    @test pmapreduce_commutative(x->myid(),prod,Iterators.product(1:nworkers(),1:1)) == prod(workers())
		    end

		    @testset "run elsewhere" begin
		    	res_exp = prod(workers())
		    	for p in workers()
		    		res = @fetchfrom p pmapreduce_commutative(x->myid(),prod,1:nworkers())
		        	@test res == res_exp
		        end
		    end

		    @testset "errors" begin
		        @test_throws exceptiontype pmapreduce_commutative(
												x->throws(BoundsError()),sum,1:10)
		        @test_throws exceptiontype pmapreduce_commutative(
												identity,x->throws(BoundsError()),1:10)
				@test_throws exceptiontype pmapreduce_commutative(
												x->throw(ErrorException("eh")),
												x->throws(BoundsError()),1:10)
		    end
		end
		@testset "elementwise" begin
			@testset "comparison with map" begin
			    iter = 1:1000
			    res = pmapreduce_commutative_elementwise(x->x^2,sum,iter)
			    @test res == sum(x->x^2,iter)
			    @test res == pmapsum_elementwise(x->x^2,iter)
			    @test res == pmapsum(plist->sum(x[1]^2 for x in plist),iter)
			    res = pmapreduce_commutative_elementwise(x->x^2,sum,(iter,))
			    @test res == sum(x->x^2,iter)
			    res = pmapreduce_commutative_elementwise(x->x^2,sum,Iterators.product(iter))
			    @test res == sum(x->x^2,iter)
			end

			@testset "run elsewhere" begin
				iter = 1:1000
				res_exp = sum(x->x^2,iter)
		    	for p in workers()
		    		res = @fetchfrom p pmapreduce_commutative_elementwise(x->x^2,sum,iter)
		        	@test res == res_exp
		        end
		    end

			@testset "errors" begin
				@test_throws exceptiontype pmapreduce_commutative_elementwise(
												x->throws(BoundsError()),sum,1:10)
				@test_throws exceptiontype pmapreduce_commutative_elementwise(
												identity,x->throws(BoundsError()),1:10)
				@test_throws exceptiontype pmapreduce_commutative_elementwise(
												x->throw(ErrorException("eh")),
												x->throws(BoundsError()),1:10)
			end
		end
	end
    
	@testset "pmapreduce" begin
		@testset "batch" begin
		    @testset "sum" begin
			    @test pmapreduce(x->myid(),sum,1:nworkers()) == sum(workers())
			    @test pmapreduce(x->myid(),sum,(1:nworkers(),)) == sum(workers())
			    @test pmapreduce(x->myid(),sum,Iterators.product(1:nworkers())) == sum(workers())
			    @test pmapreduce(x->myid(),sum,(1:nworkers(),1:1)) == sum(workers())
			    @test pmapreduce(x->myid(),sum,Iterators.product(1:nworkers(),1:1)) == sum(workers())
			    @test pmapreduce(x->myid(),sum,1:nworkers()) == pmapsum(x->myid(),1:nworkers())
		    end

		    @testset "concatenation" begin
			    @test pmapreduce(x->ones(2),x->vcat(x...),1:nworkers()) == ones(2*nworkers())
			    @test pmapreduce(x->ones(2),x->hcat(x...),1:nworkers()) == ones(2,nworkers())
		    end

		    @testset "sorting" begin
			    @test pmapreduce(x->ones(2)*workerrank(),x->vcat(x...),1:nworkers()) == 
			    		vcat((ones(2).*i for i=1:nworkers())...)
		    end

		    @testset "worker id" begin
		    	@test pmapreduce(x->workerrank(),x->vcat(x...),1:nworkers()) == collect(1:nworkers())
		    	@test pmapreduce(x->myid(),x->vcat(x...),1:nworkers()) == workers()
			end

			@testset "run elsewhere" begin
				res_exp = sum(workers())
		    	for p in workers()
		    		res = @fetchfrom p pmapreduce(x->myid(),sum,1:nworkers())
		        	@test res == res_exp
		        end
		    end

			@testset "errors" begin
			    @test_throws exceptiontype pmapreduce(x->throws(BoundsError()),sum,1:10)
				@test_throws exceptiontype pmapreduce(identity,x->throws(BoundsError()),1:10)
				@test_throws exceptiontype pmapreduce(x->throw(ErrorException("eh")),
												x->throws(BoundsError()),1:10)
			end
		end
	end
end

rmprocs(workers())