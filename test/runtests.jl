using ParallelUtilities,Test

function split_across_processors(num_tasks::Integer,num_procs,proc_id)
    if num_procs == 1
        return num_tasks
    end

    num_tasks_per_process,num_tasks_leftover = div(num_tasks,num_procs),mod(num_tasks,num_procs)

    num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
    task_start = num_tasks_per_process*(proc_id-1) + min(num_tasks_leftover+1,proc_id);

    return task_start:(task_start+num_tasks_on_proc-1)
end

function split_across_processors(arr₁::Base.Iterators.ProductIterator,num_procs,proc_id)

    @assert(proc_id<=num_procs,"processor rank has to be less than number of workers engaged")

    num_tasks = length(arr₁);

    num_tasks_per_process,num_tasks_leftover = div(num_tasks,num_procs),mod(num_tasks,num_procs)

    num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
    task_start = num_tasks_per_process*(proc_id-1) + min(num_tasks_leftover,proc_id-1) + 1;

    Iterators.take(Iterators.drop(arr₁,task_start-1),num_tasks_on_proc)
end

function split_product_across_processors(arrs_tuple,num_procs,proc_id)
	split_across_processors(Iterators.product(arrs_tuple...),num_procs,proc_id)
end



@testset "ProductSplit" begin
    @testset "Constructor" begin

	    function checkPSconstructor(iters,npmax=10)
			for np = 1:npmax, p = 1:np
		        ps = ProductSplit(iters,np,p)
		        @test collect(ps) == collect(split_product_across_processors(iters,np,p))
		    end
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
    	end
    end

    @testset "extrema" begin

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
end