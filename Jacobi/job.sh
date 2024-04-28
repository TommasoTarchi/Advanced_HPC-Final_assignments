#!/bin/bash
#SBATCH --job-name=Jacobi
#SBATCH --ntasks=3
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


# set matrix size
mat_size=27


module load cuda/
module load openmpi/4.1.6--nvhpc--23.11
#module load openmpi/4.1.6--gcc--12.2.0

srun -n 1 mpicc -acc=noautopar -Minfo=all -DOPENACC src/functions.c src/jacobi.c -o jacobi.x
#srun -n 1 mpicc src/functions.c src/jacobi.c -o jacobi.x

mpirun ./jacobi.x $mat_size 100 7 4

rm jacobi.x
