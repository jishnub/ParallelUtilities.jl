#!/bin/bash

#SBATCH -n 2
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 28
#SBATCH --job-name=threadstest
#SBATCH --time=00:05:00
#SBATCH -e threadstest.err
#SBATCH -o threadstest.out

juliaexe=$SCRATCH/julia/julia-1.7.0-rc2/bin/julia
$juliaexe -t 28 --startup=no threadsjobscript.jl
