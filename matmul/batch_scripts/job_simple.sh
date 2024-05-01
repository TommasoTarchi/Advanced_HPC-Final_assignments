#!/bin/bash
#SBATCH --job-name=matmul_simple
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
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
module load openmpi/4.1.6--gcc--12.2.0


cd ../

# create datafile
echo "#init,communication,computation" > profiling/times_simple.csv

# compile program
srun -n 1 -N 1 mpicc -fopenmp -lm src/functions.c src/matmul_simple.c -DOPENMP -DTIME -DTEST -DMAT_SIZE=$mat_size -o matmul_simple.x

# run program (each node will host 1 MPI processe)
for ((nprocs = 1; nprocs <= 8; nprocs *= 2))
do
    mpirun -np "$nprocs" --map-by node:PE=1 ./matmul_simple.x
done

# remove executable
rm matmul_simple.x

cd batch_scripts/ || exit
