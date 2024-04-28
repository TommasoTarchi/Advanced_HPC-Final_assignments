#!/bin/bash
#SBATCH --job-name=matmul_cublas
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


# set matrix size
mat_size=27


module load cuda/12.1
module load nvhpc/23.11
module load openmpi/4.1.6--gcc--12.2.0
module load openblas/0.3.24--nvhpc--23.11

cd ../

srun -n 1 nvcc -lm src/functions.c -lmpi -lcublas -lcudart -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib/ -L/leonardo/prod/opt/compilers/cuda/12.1/none src/matmul_cublas.c -DMAT_SIZE=$mat_size -o matmul.x

mpirun ./matmul.x

rm matmul.x

cd batch_scripts/
