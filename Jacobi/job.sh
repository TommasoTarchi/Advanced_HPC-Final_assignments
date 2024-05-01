#!/bin/bash
#SBATCH --job-name=matmul_simple
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


# set matrix size
mat_size=1000

# set number of openMP threads per process
export OMP_NUM_THREADS=20

# set openMP binding policy (each thread on a different core)
export OMP_PROC_BIND=true
export OMP_PLACES=cores


# load modules
module load cuda/
module load openmpi/4.1.6--nvhpc--23.11
#module load openmpi/4.1.6--gcc--12.2.0


# create datafile
echo "#init,communication,computation" > profiling/times.csv

# compile program
srun -n 1 -N 1 mpicc -acc=noautopar -Minfo=all -fopenmp -DOPENMP -DOPENACC -DTIME src/functions.c src/jacobi.c -o jacobi.x
#srun -n 1 -N 1 mpicc -fopenmp -DOPENMP -DTIME src/functions.c src/jacobi.c -o jacobi.x  # compile without openACC

# run program
for ((nprocs = 1; nprocs <= 8; nprocs *= 2))
do
	mpirun -np "$nprocs" --map-by node:PE=1 ./jacobi.x $mat_size 100 12 4 
done

# remove executable
rm jacobi.x
