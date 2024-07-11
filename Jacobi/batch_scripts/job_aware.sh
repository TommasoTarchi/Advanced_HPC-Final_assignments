#!/bin/bash
#SBATCH --job-name=Jacobi
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=11
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


# set matrix size
mat_size=12000

# set number of openMP threads per process
export OMP_NUM_THREADS=10

# set openMP binding policy (each thread on a different core)
export OMP_PROC_BIND=close
export OMP_PLACES=cores


# load modules
module load cuda/
module load openmpi/4.1.6--nvhpc--23.11


cd ../

# create datafile
echo "#n_procs,init,communication,computation" > profiling/times_aware.csv

# compile program
srun -n 1 -N 1 mpicc -acc=noautopar -Minfo=all -fopenmp -DOPENMP -DOPENACC -DTIME src/functions.c src/jacobi_aware.c -o jacobi_aware.x

# run program
for ((nprocs = 1; nprocs <= 32; nprocs *= 2))
do
	echo -n "$nprocs," >> profiling/times_aware.csv
	mpirun -np "$nprocs" --map-by node:PE=10 --report-bindings ./jacobi_aware.x $mat_size 10 11 4 
done

# remove executable
srun -n 1 -N 1 rm jacobi_aware.x

cd batch_scripts/ || exit