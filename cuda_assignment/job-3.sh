#!/bin/bash
#SBATCH --job-name=first_assignment-3
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


#module load cuda/12.1
module load nvhpc/23.11
module load openmpi/4.1.6--gcc--12.2.0
#module load openmpi/4.1.6--nvhpc--23.11
module load openblas/0.3.24--nvhpc--23.11

srun -n 1 nvcc -o first_assignment-3.x first_assignment-3.cu -lmpi -lcublas -lcudart -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib/ -L/leonardo/prod/opt/compilers/cuda/12.1/none
#srun -n 1 mpicc -o first_assignment-3.x first_assignment-3.cu -lcublas -lcudart -L/leonardo/prod/opt/compilers/cuda/12.1/none -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib/

mpirun ./first_assignment-3.x

rm first_assignment-3.x
