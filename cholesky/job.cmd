#!/bin/bash

# Configuration for 1 node, 4 cores and 5 minutes of execution time
#SBATCH --job-name=ex1
#SBATCH -p ippd-cpu
#SBATCH --output=out_cholesky_%j.out
#SBATCH --error=out_cholesky_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=00:05:00

module load GCC/10.2.0

make >> make.out || exit 1      # Exit if make fails

./cholesky 3000

