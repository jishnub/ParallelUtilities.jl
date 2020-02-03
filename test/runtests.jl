using ParallelUtilities,Test,Distributed

@testset "ProductSplit" begin

	function split_across_processors_iterators(arr₁::Base.Iterators.ProductIterator,num_procs,proc_id)
	    @assert(proc_id<=num_procs,"processor rank has to be less than number of workers engaged")

	    num_tasks = length(arr₁);

	    num_tasks_per_process,num_tasks_leftover = div(num_tasks,num_procs),mod(num_tasks,num_procs)

	    num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
	    task_start = num_tasks_per_process*(proc_id-1) + min(num_tasks_leftover,proc_id-1) + 1;

	    Iterators.take(Iterators.drop(arr₁,task_start-1),num_tasks_on_proc)
	end

	function split_product_across_processors_iterators(arrs_tuple,num_procs,proc_id)
		split_across_processors_iterators(Iterators.product(arrs_tuple...),num_procs,proc_id)
	end

    @testset "Constructor" begin

	    function checkPSconstructor(iters,npmax=10)
			for np = 1:npmax, p = 1:np
		        ps = ProductSplit(iters,np,p)
		        @test collect(ps) == collect(split_product_across_processors_iterators(iters,np,p))
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
    end

    @testset "extrema" begin

    	@testset "min max extrema" begin
	    	function checkPSextrema(iters,fn::Function,npmax=10)
				for np = 1:npmax, p = 1:np
			        ps = ProductSplit(iters,np,p)
			        pcol = collect(ps)
			        for dim in 1:length(iters)
			        	@test begin 
			        		res = fn(ps,dim) == fn(x->x[dim],pcol)
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
        @test isnothing(whichproc(iters,(100,100,100),1))
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

    @testset "indexinsplitproduct" begin
        
        for iters in [(1:10,),(1:10,4:6),(1:10,4:6,1:4),(1:2:10,4:1:6)]
	        for np=1:10,proc_id=1:np
	        	ps = ProductSplit(iters,np,proc_id)
	        	for (ind,val) in enumerate(ps)
	        		@test indexinsplitproduct(ps,val) == ind
	        	end
	        end
	    end
    end
end

@testset "LittleEndianTuple" begin
    @testset "isless" begin
    	a = ParallelUtilities.LittleEndianTuple((1,2,3))
        b = ParallelUtilities.LittleEndianTuple((2,2,3))
        @test a < b
        @test a <= b
        b = ParallelUtilities.LittleEndianTuple((1,1,3))
        @test b < a
        @test b <= a
        b = ParallelUtilities.LittleEndianTuple((2,1,3))
        @test b < a
        @test b <= a
        b = ParallelUtilities.LittleEndianTuple((2,1,4))
        @test a < b
        @test a <= b
    end
    @testset "equal" begin
        a = ParallelUtilities.LittleEndianTuple((1,2,3))
        @test a == a
        @test a <= a
        b = ParallelUtilities.LittleEndianTuple(a.t)
        @test a == b
        @test a <= b
    end
end