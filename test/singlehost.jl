using Distributed

include("misctests_singleprocess.jl")
include("productsplit.jl")
include("paralleltests.jl")

for workersused in [1, 2, 4, 8]
	addprocs(workersused)

	try
		include("paralleltests.jl")
	finally
		rmprocs(workers())
	end
end

