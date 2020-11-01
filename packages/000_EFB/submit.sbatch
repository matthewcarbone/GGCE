#!/bin/bash
#SBATCH -N 1
#SBATCH -p New
#SBATCH -c 48
#SBATCH -J GMA
#SBATCH --exclusive
#SBATCH --output=job_data/GMA_%A.out
#SBATCH --error=job_data/GMA_%A.err
#SBATCH --mem=62000M

module load compilers/gcc-10.1.0

export OMP_NUM_THREADS=1
nprocs=12

#run the application:
python3 ._submit.py "$nprocs" "$@"
