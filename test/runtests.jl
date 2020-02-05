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
		        @test ParallelUtilities.ntasks(ps) == ntasks_total
		        @test ParallelUtilities.ntasks(ps.iterators) == ntasks_total
		    end

		    @test_throws ParallelUtilities.ProcessorNumberError ProductSplit(iters,npmax,npmax+1)
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
    	end
    end

    @testset "firstlast" begin
        @testset "first" begin
            for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)],np=1:10
	            ps = ProductSplit(iters,np,1)
	            @test first(ps) == map(first,iters)
	        end

	        iters = (1:1,)
	        ps = ProductSplit(iters,2length(iters[1]),length(iters[1])+1) # must be empty
	        @test first(ps) === nothing
        end
        @testset "last" begin
            for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)],np=1:10
	            ps = ProductSplit(iters,np,np)
	            @test last(ps) == map(last,iters)
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
    		for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)]
	    		for np = 1:10, proc_id = 1:np
	    	    	ps = ProductSplit(iters,np,proc_id)
	    	    	ext = Tuple(map(extrema,zip(collect(ps)...)))
	    	    	@test extremadims(ps) == ext
	    	    end
	    	end
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
    end

    @testset "evenlyscatterproduct" begin
        n = 10
        np,proc_id = 5,3
        @test evenlyscatterproduct(n,np,proc_id) == ProductSplit((1:n,),np,proc_id)
        for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)]
        	@test evenlyscatterproduct(iters,np,proc_id) == ProductSplit(iters,np,proc_id)
        end
    end

    @testset "whichproc + newprocrange" begin
        np,proc_id = 5,3
        iters = (1:10,4:6,1:4)
        ps = ProductSplit(iters,np,proc_id)
        @test whichproc(iters,first(ps),1) == 1
        @test whichproc(iters,(100,100,100),1) === nothing
        @test newprocrange(ps,1) == 1:1

        for np_new in 1:np
        	for proc_id_new=1:np_new
	        	ps_new = ProductSplit(iters,np_new,proc_id_new)
	        	for val in ps_new
	        		@test whichproc(iters,val,np_new) == proc_id_new
	        	end
	        end
	        procid_new_first = whichproc(iters,first(ps),np_new)
	        proc_new_last = whichproc(iters,last(ps),np_new)
        	@test newprocrange(ps,np_new) == procid_new_first:proc_new_last
        end
    end

    @testset "localindex" begin
        
        for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)]
	        for np=1:10,proc_id=1:np
	        	ps = ProductSplit(iters,np,proc_id)
	        	for (ind,val) in enumerate(ps)
	        		@test localindex(ps,val) == ind
	        	end
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
        @test a <= a
        b = ParallelUtilities.ReverseLexicographicTuple(a.t)
        @test a == b
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

	@testset "pmapbatch" begin
		iterable = 1:nworkers()
	    res = pmapbatch(x->myid(),iterable)
	    @test res == workers()
	    res = pmapbatch(x->myid(),(iterable,))
	    @test res == workers()
	    res = pmapbatch(x->myid(),(iterable,1:1))
	    @test res == workers()

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

	@testset "pmapbatch_elementwise" begin
	    iterable = 1:nworkers()
	    res = pmapbatch_elementwise(identity,iterable)
	    @test res == iterable

		iterable = 1:20
	    res = pmapbatch_elementwise(x->x^2,iterable)
	    @test res == iterable.^2
	end

	@testset "pmapsum" begin

	    @testset "worker id" begin
		    @test pmapsum(x->workerrank(),1:nworkers()) == sum(1:nworkers())
		    @test pmapsum(x->workerrank(),(1:nworkers(),)) == sum(1:nworkers())
		    @test pmapsum(x->workerrank(),(1:nworkers(),1:1)) == sum(1:nworkers())
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
	end

	@testset "pmapsum_elementwise" begin
	    iterable = 1:100
	    res = pmapsum_elementwise(identity,iterable)
	    @test res == sum(iterable)

	    iterable = 1:100
	    res = pmapsum_elementwise(x->x^2,iterable)
	    @test res == sum(x->x^2,iterable)
	    @test res == pmapsum(plist->sum(x[1]^2 for x in plist),iterable)
	end
    
	@testset "pmapreduce" begin
		
		@testset "unsorted" begin
			@testset "sum" begin
			    @test pmapreduce_commutative(x->myid(),sum,1:nworkers()) == sum(workers())
			    @test pmapreduce_commutative(x->myid(),sum,(1:nworkers(),)) == sum(workers())
			    @test pmapreduce_commutative(x->myid(),sum,(1:nworkers(),1:1)) == sum(workers())
			    @test pmapreduce_commutative(x->myid(),sum,1:nworkers()) == pmapsum(x->myid(),1:nworkers())
		    end
		    @testset "prod" begin
			    @test pmapreduce_commutative(x->myid(),prod,1:nworkers()) == prod(workers())
			    @test pmapreduce_commutative(x->myid(),prod,(1:nworkers(),)) == prod(workers())
			    @test pmapreduce_commutative(x->myid(),prod,(1:nworkers(),1:1)) == prod(workers())
		    end
		end

		@testset "unsorted elementwise" begin
		    iter = 1:1000
		    res = pmapreduce_commutative_elementwise(x->x^2,sum,iter)
		    @test res == sum(x->x^2,iter)
		    @test res == pmapsum_elementwise(x->x^2,iter)
		    @test res == pmapsum(plist->sum(x[1]^2 for x in plist),iter)
		end
		
		@testset "sorted" begin
		    @testset "sum" begin
			    @test pmapreduce(x->myid(),sum,1:nworkers()) == sum(workers())
			    @test pmapreduce(x->myid(),sum,(1:nworkers(),)) == sum(workers())
			    @test pmapreduce(x->myid(),sum,(1:nworkers(),1:1)) == sum(workers())
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
		end
		    
	end
end

rmprocs(workers())