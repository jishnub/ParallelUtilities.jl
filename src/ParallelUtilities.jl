module ParallelUtilities

using Reexport
@reexport using Distributed

worker_rank() = myid()-minimum(workers())+1

function split_across_processors(num_tasks::Integer,num_procs=nworkers(),proc_id=worker_rank())
	if num_procs == 1
		return num_tasks
	end

	num_tasks_per_process,num_tasks_leftover = div(num_tasks,num_procs),mod(num_tasks,num_procs)

	num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
	task_start = num_tasks_per_process*(proc_id-1) + min(num_tasks_leftover+1,proc_id);

	return task_start:(task_start+num_tasks_on_proc-1)
end

function split_across_processors(arr₁,num_procs=nworkers(),proc_id=worker_rank())

	@assert(proc_id<=num_procs,"processor rank has to be less than number of workers engaged")

	num_tasks = length(arr₁);

	num_tasks_per_process,num_tasks_leftover = div(num_tasks,num_procs),mod(num_tasks,num_procs)

	num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
	task_start = num_tasks_per_process*(proc_id-1) + min(num_tasks_leftover+1,proc_id);

	Iterators.take(Iterators.drop(arr₁,task_start-1),num_tasks_on_proc)
end

function split_product_across_processors(arr₁,arr₂,num_procs::Integer=nworkers(),proc_id::Integer=worker_rank())
	# arr₁ will change faster
	split_across_processors(Iterators.product(arr₁,arr₂),num_procs,proc_id)
end

function split_product_across_processors(arrs_tuple,num_procs::Integer=nworkers(),proc_id::Integer=worker_rank())
	return split_across_processors(Iterators.product(arrs_tuple...),num_procs,proc_id)
end

function get_processor_id_from_split_array(arr₁,arr₂,(arr₁_value,arr₂_value)::Tuple,num_procs)
	# Find the closest match in arrays

	if (arr₁_value ∉ arr₁) || (arr₂_value ∉ arr₂)
		return nothing # invalid
	end
	
	num_tasks = length(arr₁)*length(arr₂);

	a1_match_index = searchsortedfirst(arr₁,arr₁_value)
	a2_match_index = searchsortedfirst(arr₂,arr₂_value)

	num_tasks_per_process,num_tasks_leftover = div(num_tasks,num_procs),mod(num_tasks,num_procs)

	proc_id = 1
	num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
	total_tasks_till_proc_id = num_tasks_on_proc

	task_no = 0

	for (ind2,a2) in enumerate(arr₂), (ind1,a1) in enumerate(arr₁)
		
		task_no +=1
		if task_no > total_tasks_till_proc_id
			proc_id += 1
			num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks,num_procs) ? 1 : 0 );
			total_tasks_till_proc_id += num_tasks_on_proc
		end

		if ind2< a2_match_index
			continue
		end

		if (ind2 == a2_match_index) && (ind1 == a1_match_index)
			break
		end
	end

	return proc_id
end

function get_processor_range_from_split_array(arr₁,arr₂,modes_on_proc,num_procs)
	
	if isempty(modes_on_proc)
		return 0:-1 # empty range
	end

	tasks_arr = collect(modes_on_proc)
	proc_id_start = get_processor_id_from_split_array(arr₁,arr₂,first(tasks_arr),num_procs)
	proc_id_end = get_processor_id_from_split_array(arr₁,arr₂,last(tasks_arr),num_procs)
	return proc_id_start:proc_id_end
end

function get_index_in_split_array(modes_on_proc,(arr₁_value,arr₂_value))
	if isnothing(modes_on_proc)
		return nothing
	end
	for (ind,(t1,t2)) in enumerate(modes_on_proc)
		if (t1==arr₁_value) && (t2 == arr₂_value)
			return ind
		end
	end
	nothing
end

function procid_and_mode_index(arr₁,arr₂,(arr₁_value,arr₂_value),num_procs)
	proc_id_mode = get_processor_id_from_split_array(arr₁,arr₂,(arr₁_value,arr₂_value),num_procs)
	modes_in_procid_file = split_product_across_processors(arr₁,arr₂,num_procs,proc_id_mode)
	mode_index = get_index_in_split_array(modes_in_procid_file,(arr₁_value,arr₂_value))
	return proc_id_mode,mode_index
end

function mode_index_in_file(arr₁,arr₂,(arr₁_value,arr₂_value),num_procs,proc_id_mode)
	modes_in_procid_file = split_product_across_processors(arr₁,arr₂,num_procs,proc_id_mode)
	mode_index = get_index_in_split_array(modes_in_procid_file,(arr₁_value,arr₂_value))
end

function procid_allmodes(arr₁,arr₂,iter,num_procs=nworkers_active(arr₁,arr₂))
	procid = zeros(Int64,length(iter))
	for (ind,mode) in enumerate(iter)
		procid[ind] = get_processor_id_from_split_array(arr₁,arr₂,mode,num_procs)
	end
	return procid
end

workers_active(arr) = workers()[1:min(length(arr),nworkers())]

workers_active(arr₁,arr₂) = workers_active(Iterators.product(arr₁,arr₂))

nworkers_active(args...) = length(workers_active(args...))

function minmax_from_split_array(iterable)
	arr₁_min,arr₂_min = first(iterable)
	arr₁_max,arr₂_max = arr₁_min,arr₂_min
	for (arr₁_value,arr₂_value) in iterable
		arr₁_min = min(arr₁_min,arr₁_value)
		arr₁_max = max(arr₁_max,arr₁_value)
		arr₂_min = min(arr₂_min,arr₂_value)
		arr₂_max = max(arr₂_max,arr₂_value)
	end
	return (arr₁_min=arr₁_min,arr₁_max=arr₁_max,arr₂_min=arr₂_min,arr₂_max=arr₂_max)
end

function get_hostnames(procs_used=workers())
	hostnames = Vector{String}(undef,length(procs_used))
	@sync for (ind,p) in enumerate(procs_used)
		@async hostnames[ind] = @fetchfrom p Libc.gethostname()
	end
	return hostnames
end

get_nodes(hostnames::Vector{String}) = unique(hostnames)
get_nodes(procs_used::Vector{<:Integer}=workers()) = get_nodes(get_hostnames(procs_used))

function get_nprocs_node(hostnames::Vector{String})
	nodes = get_nodes(hostnames)
	get_nprocs_node(hostnames,nodes)	
end

function get_nprocs_node(hostnames::Vector{String},nodes::Vector{String})
	Dict(node=>count(isequal(node),hostnames) for node in nodes)
end

get_nprocs_node(procs_used::Vector{<:Integer}=workers()) = get_nprocs_node(get_hostnames(procs_used))

function pmapsum(f::Function,iterable,args...;kwargs...)

	procs_used = workers_active(iterable)
	num_workers = length(procs_used)
	hostnames = get_hostnames(procs_used)
	nodes = get_nodes(hostnames)
	np_nodes = get_nprocs_node(hostnames)
	pid_rank0_on_node = [procs_used[findfirst(isequal(node),hostnames)] for node in nodes]

	function apply_and_stash(f,iterable,args...;channel,kwargs...)
		result = f(iterable,args...;kwargs...)
		put!(channel,result)
	end

	# Worker at which final reduction takes place
	p_final = first(pid_rank0_on_node)
	sum_channel = RemoteChannel(()->Channel{Any}(10),p_final)
	master_channel_nodes = Dict(node=>RemoteChannel(()->Channel{Any}(10),p)
							for (node,p) in zip(nodes,pid_rank0_on_node))
	worker_channel_nodes = Dict(node=>RemoteChannel(()->Channel{Any}(10),p)
							for (node,p) in zip(nodes,pid_rank0_on_node))

	# Run the function on each processor
	@sync for (rank,p) in enumerate(procs_used)
		@async begin
			iterable_on_proc = split_across_processors(iterable,num_workers,rank)
			node = hostnames[rank]
			master_channel_node = master_channel_nodes[node]
			worker_channel_node = worker_channel_nodes[node]
			np_node = np_nodes[node]
			if p in pid_rank0_on_node
				@spawnat p apply_and_stash(f,iterable_on_proc,args...;kwargs...,
					channel=master_channel_node)
				@spawnat p sum_channel(worker_channel_node,master_channel_node,np_node)
			else
				@spawnat p apply_and_stash(f,iterable_on_proc,args...;kwargs...,
					channel=worker_channel_node)
			end
		end
		@async begin
		    @spawnat p_final sum_channel(master_channel_nodes,sum_channel,length(nodes))
		end
	end

	finalize.(values(worker_channel_nodes))
	finalize.(values(master_channel_nodes))

	take!(sum_channel)
end

function pmap_onebatch_per_worker(f::Function,iterable,args...;num_workers=nothing,kwargs...)

	procs_used = workers_active(iterable)
	if !isnothing(num_workers) && num_workers<=length(procs_used)
		procs_used = procs_used[1:num_workers]
	end
	num_workers = length(procs_used)

	futures = Vector{Future}(undef,num_workers)
	@sync for (rank,p) in enumerate(procs_used)
		@async begin
			iterable_on_proc = split_across_processors(iterable,num_workers,rank)
			futures[rank] = @spawnat p f(iterable_on_proc,args...;kwargs...)
		end
	end
	return futures
end

function spawnf(f::Function,iterable)

	futures = Vector{Future}(undef,nworkers())
	@sync for (rank,p) in enumerate(workers())
		futures[rank] = @spawnat p f(iterable)
	end
	return futures
end

function sum_channel(worker_channel,master_channel,np_node)
	@sync for i in 1:np_node-1
		@async begin
			s = take!(worker_channel) + take!(master_channel)
			put!(master_channel,s)
		end
	end
end

#############################################################################

export split_across_processors,split_product_across_processors,
get_processor_id_from_split_array,
procid_allmodes,mode_index_in_file,
get_processor_range_from_split_array,workers_active,worker_rank,
get_index_in_split_array,procid_and_mode_index,minmax_from_split_array,
node_remotechannels,pmapsum,sum_at_node,pmap_onebatch_per_worker,
get_nodes,get_hostnames,get_nprocs_node

end # module
