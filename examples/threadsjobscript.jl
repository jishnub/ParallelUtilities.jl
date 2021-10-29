using ClusterManagers
job_file_loc = mktempdir(@__DIR__)
addprocs_slurm(2, exeflags=["-t 28", "--startup=no"], job_file_loc = job_file_loc)
using Distributed
@everywhere include(joinpath(@__DIR__, "threads.jl"))
ThreadsTiming.compare_with_serial()
