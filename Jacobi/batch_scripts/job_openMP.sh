#!/bin/bash
#SBATCH --job-name=Jacobi
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


# set matrix size
mat_size=111

# set number of openMP threads per process
export OMP_NUM_THREADS=4

# set openMP binding policy (each thread on a different core)
export OMP_PROC_BIND=close
export OMP_PLACES=cores


# load modules
module load openmpi/4.1.6--gcc--12.2.0


cd ../

# create datafile
echo "#n_procs,init,communication,computation" > profiling/times.csv

# compile program
srun -n 1 -N 1 mpicc -fopenmp -DOPENMP -DTIME src/functions.c src/jacobi.c -o jacobi.x

# run program
for ((nprocs = 2; nprocs <= 2; nprocs *= 2))
do
	echo -n "$nprocs," >> profiling/times.csv
	mpirun -np "$nprocs" --map-by node:PE=4444 --report-bindings ./jacobi.x $mat_size 10 11 4 
done

# remove executable
srun -n 1 -N 1 rm jacobi.x

cd batch_scripts/ || exit
