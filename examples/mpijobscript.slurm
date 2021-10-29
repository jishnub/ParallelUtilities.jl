#!/bin/bash

#SBATCH -n 78
#SBATCH --job-name=mpitest
#SBATCH --time=00:05:00
#SBATCH -e mpitest.err
#SBATCH -o mpitest.out

juliaexe=$SCRATCH/julia/julia-1.7.0-rc2/bin/julia
mpirun $juliaexe --startup=no mpiclustermanager_mpitransport.jl
