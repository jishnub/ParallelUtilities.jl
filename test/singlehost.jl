using Distributed

const workersused = 8
addprocs(workersused)

include("tests.jl")

rmprocs(workers())
