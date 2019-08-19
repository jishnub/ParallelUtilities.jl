module ParallelUtilities

using Reexport
@reexport using Distributed

export split_across_processors,split_product_across_processors,
get_processor_id_from_split_array,procid_allmodes,mode_index_in_file,
get_processor_range_from_split_array,workers_active,nworkers_active,worker_rank,
get_index_in_split_array,procid_and_mode_index,extrema_from_split_array,
pmapsum,sum_at_node,pmap_onebatch_per_worker,moderanges_common_lastarray,
get_nodes,get_hostnames,get_nprocs_node

function worker_rank()
	if nworkers()==1
		return 1
	end
	myid()-minimum(workers())+1
end

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

function split_product_across_processors(arr₁::AbstractVector,arr₂::AbstractVector,
	num_procs::Integer=nworkers(),proc_id::Integer=worker_rank())
	# arr₁ will change faster
	split_across_processors(Iterators.product(arr₁,arr₂),num_procs,proc_id)
end

function split_product_across_processors(arrs_tuple,num_procs::Integer=nworkers(),proc_id::Integer=worker_rank())
	return split_across_processors(Iterators.product(arrs_tuple...),num_procs,proc_id)
end

function get_processor_id_from_split_array(arr₁::AbstractVector,arr₂::AbstractVector,
	(arr₁_value,arr₂_value)::Tuple,num_procs)
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

function get_processor_id_from_split_array(iter,val::Tuple,num_procs)
	for proc_id in 1:num_procs
		tasks_on_proc = split_across_processors(iter,num_procs,proc_id)
		if val ∈ tasks_on_proc
			return proc_id
		end
	end
	return 0
end

function get_processor_range_from_split_array(arr₁::AbstractVector,arr₂::AbstractVector,
	iter_section,num_procs::Integer)
	
	if isempty(iter_section)
		return 0:-1 # empty range
	end

	tasks_arr = collect(iter_section)
	proc_id_start = get_processor_id_from_split_array(arr₁,arr₂,first(tasks_arr),num_procs)
	proc_id_end = get_processor_id_from_split_array(arr₁,arr₂,last(tasks_arr),num_procs)
	return proc_id_start:proc_id_end
end

function get_processor_range_from_split_array(iter,iter_section,num_procs::Integer)
	
	if isempty(iter_section)
		return 0:-1 # empty range
	end

	tasks_arr = collect(iter_section)
	proc_id_start = get_processor_id_from_split_array(iter,first(tasks_arr),num_procs)
	proc_id_end = get_processor_id_from_split_array(iter,last(tasks_arr),num_procs)
	return proc_id_start:proc_id_end
end

function get_index_in_split_array(iter_section,val::Tuple)
	if isnothing(iter_section)
		return nothing
	end
	for (ind,val_ind) in enumerate(iter_section)
		if val_ind == val
			return ind
		end
	end
	nothing
end

function procid_and_mode_index(arr₁::AbstractVector,arr₂::AbstractVector,
	(arr₁_value,arr₂_value)::Tuple,num_procs::Integer)
	proc_id_mode = get_processor_id_from_split_array(arr₁,arr₂,(arr₁_value,arr₂_value),num_procs)
	modes_in_procid_file = split_product_across_processors(arr₁,arr₂,num_procs,proc_id_mode)
	mode_index = get_index_in_split_array(modes_in_procid_file,(arr₁_value,arr₂_value))
	return proc_id_mode,mode_index
end

function procid_and_mode_index(iter,val::Tuple,num_procs::Integer)
	proc_id_mode = get_processor_id_from_split_array(iter,val,num_procs)
	modes_in_procid_file = split_across_processors(iter,num_procs,proc_id_mode)
	mode_index = get_index_in_split_array(modes_in_procid_file,val)
	return proc_id_mode,mode_index
end

function mode_index_in_file(arr₁::AbstractVector,arr₂::AbstractVector,
	(arr₁_value,arr₂_value)::Tuple,num_procs::Integer,proc_id_mode::Integer)
	modes_in_procid_file = split_product_across_processors(arr₁,arr₂,num_procs,proc_id_mode)
	mode_index = get_index_in_split_array(modes_in_procid_file,(arr₁_value,arr₂_value))
end

function procid_allmodes(arr₁::AbstractVector,arr₂::AbstractVector,
	iter,num_procs=nworkers_active(arr₁,arr₂))
	procid = zeros(Int64,length(iter))
	for (ind,mode) in enumerate(iter)
		procid[ind] = get_processor_id_from_split_array(arr₁,arr₂,mode,num_procs)
	end
	return procid
end

workers_active(arr) = workers()[1:min(length(arr),nworkers())]

workers_active(arrs...) = workers_active(Iterators.product(arrs...))

nworkers_active(args...) = length(workers_active(args...))

function extrema_from_split_array(iterable)
	val_first = first(iterable)
	min_vals = collect(val_first)
	max_vals = copy(min_vals)

	for val in iterable
		for (ind,vi) in enumerate(val)
			min_vals[ind] = min(min_vals[ind],vi)
			max_vals[ind] = max(max_vals[ind],vi)
		end
	end
	collect(zip(min_vals,max_vals))
end

function moderanges_common_lastarray(iterable)
	m = extrema_from_split_array(iterable)
	lastvar_min = last(m)[1]
	lastvar_max = last(m)[2]

	val_first = first(iterable)
	min_vals = collect(val_first[1:end-1])
	max_vals = copy(min_vals)

	for val in iterable
		for (ind,vi) in enumerate(val[1:end-1])
			if val[end]==lastvar_min
				min_vals[ind] = min(min_vals[ind],vi)
			end
			if val[end]==lastvar_max
				max_vals[ind] = max(max_vals[ind],vi)
			end
		end
	end

	[(m,lastvar_min) for m in min_vals],[(m,lastvar_max) for m in max_vals]
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
	num_procs_node = Dict(node=>count(x->x==node,hostnames) for node in nodes)
end

get_nprocs_node(procs_used::Vector{<:Integer}=workers()) = get_nprocs_node(get_hostnames(procs_used))

function pmapsum(f::Function,iterable,args...;kwargs...)

	procs_used = workers_active(iterable)

	futures = pmap_onebatch_per_worker(f,iterable,args...;kwargs...)

	# Final sum across all nodes 
	# sum(fetch(f) for f in futures)
	@fetchfrom first(procs_used) @distributed (+) for f in futures
		fetch(f)
	end

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

function sum_at_node(futures::Vector{Future},hostnames)
	myhost = hostnames[worker_rank()]
	futures_on_myhost = futures[hostnames .== myhost]
	sum(fetch(f) for f in futures_on_myhost)
end

end # module
