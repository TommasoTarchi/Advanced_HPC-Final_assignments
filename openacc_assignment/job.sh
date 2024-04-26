#!/bin/bash
#SBATCH --job-name=Jacobi
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


module load cuda/
module load openmpi/4.1.6--nvhpc--23.11
#module load openmpi/

srun -n 1 mpicc -acc=noautopar -Minfo=all -DOPENACC -o jacobi_gpu.x jacobi_gpu.c functions.c
#srun -n 1 mpicc -o jacobi_gpu.x jacobi_gpu.c functions.c

mpirun ./jacobi_gpu.x 10 100 7 4

rm jacobi_gpu.x
