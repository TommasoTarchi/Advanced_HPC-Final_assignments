#!/bin/bash
#SBATCH --job-name=first_assignment-1
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


# set matrix size
mat_size=27


module load openmpi/4.1.6--gcc--12.2.0

cd ../

srun -n 1 mpicc -lm src/functions.c src/matmul_simple.c -DMAT_SIZE=$mat_size -o matmul.x

mpirun ./matmul.x

rm matmul.x

cd batch_scripts/
