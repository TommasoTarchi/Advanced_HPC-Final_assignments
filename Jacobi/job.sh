#!/bin/bash
#SBATCH --job-name=Jacobi
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


# set matrix size
mat_size=20


module load cuda/
module load openmpi/4.1.6--nvhpc--23.11
#module load openmpi/4.1.6--gcc--12.2.0


echo "#init,communication,computation" > profiling/times.csv

srun -n 1 mpicc -acc=noautopar -Minfo=all -DOPENACC -DTIME src/functions.c src/jacobi.c -o jacobi.x
#srun -n 1 mpicc src/functions.c src/jacobi.c -o jacobi.x

for ((nprocs = 1; nprocs <= 8; nprocs *= 2))
do
	mpirun -np "$nprocs" ./jacobi.x $mat_size 100 12 4 
done

rm jacobi.x
