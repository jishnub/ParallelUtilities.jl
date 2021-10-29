using ClusterManagers
job_file_loc = mktempdir(@__DIR__)
addprocs_slurm(78, exeflags=["--startup=no"], job_file_loc = job_file_loc)
using Distributed
@everywhere include(joinpath(@__DIR__, "pmapreduce.jl"))
PMapReduceTiming.compare_with_serial()
