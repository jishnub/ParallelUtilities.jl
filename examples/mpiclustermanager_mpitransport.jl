using MPIClusterManagers
import MPI
using Distributed

# This uses MPI to communicate with the workers
mgr = MPIClusterManagers.start_main_loop(MPI_TRANSPORT_ALL)

@everywhere include(joinpath(@__DIR__, "pmapreduce.jl"))
println("Using MPI_TRANSPORT_ALL")
PMapReduceTiming.compare_with_serial()

MPIClusterManagers.stop_main_loop(mgr)
