#!/bin/bash
#SBATCH --job-name=Jacobi
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=11
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


# choose matrix size and number of threads
mat_size=111
num_threads=5


# set number of openMP threads per process
export OMP_NUM_THREADS=$num_threads

# set openMP binding policy (each thread on a different core)
export OMP_PROC_BIND=close
export OMP_PLACES=cores


# load modules
module load cuda/
module load openmpi/4.1.6--nvhpc--23.11


cd ../

# create datafile
echo "#n_procs,init,communication,computation" > profiling/times.csv

# compile program
srun -n 1 -N 1 mpicc -acc=noautopar -Minfo=all -fopenmp -DOPENMP -DOPENACC -DTIME src/functions.c src/jacobi.c -o jacobi.x

# run program
for ((nprocs = 2; nprocs <= 4; nprocs *= 2))
do
	echo -n "$nprocs," >> profiling/times.csv
	mpirun -np "$nprocs" --map-by node:PE=$num_threads --report-bindings ./jacobi.x $mat_size 10 11 4 
done

# remove executable
srun -n 1 -N 1 rm jacobi.x

cd batch_scripts/ || exit
