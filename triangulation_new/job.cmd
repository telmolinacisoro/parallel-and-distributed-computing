#!/bin/bash
#SBATCH --job-name=ex1
#SBATCH -p ippd-gpu
#SBATCH --output=out_triangulation_%j.out
#SBATCH --error=out_triangulation_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00

module load NVHPC/21.9-CUDA-11.4.3

make cpu >> make.out || exit 1      # Exit if make fails
srun delaunay 1000 200 150

make gpu >> make.out || exit 1      # Exit if make fails
srun delaunay 1000 200 150
