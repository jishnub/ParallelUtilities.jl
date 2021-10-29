#!/bin/bash

#SBATCH -n 78
#SBATCH --job-name=threadstest
#SBATCH --time=00:10:00
#SBATCH -e pmapreducetest.err
#SBATCH -o pmapreducetest.out

juliaexe=$SCRATCH/julia/julia-1.7.0-rc2/bin/julia
$juliaexe --startup=no pmapreducejobscript.jl
