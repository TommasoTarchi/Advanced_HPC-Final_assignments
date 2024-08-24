#!/bin/bash
#SBATCH --job-name=jacobi_MPI-RMA
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --partition=dcgp_usr_prod
#SBATCH -A ict24_dssc_cpu
#SBATCH --output=report_MPI-RMA.out


# choose matrix size and number of threads
mat_size=12000
num_threads=20


# set number of openMP threads per process
export OMP_NUM_THREADS=$num_threads

# set openMP binding policy (each thread on a different core)
export OMP_PROC_BIND=close
export OMP_PLACES=cores


# load modules
module load openmpi/4.1.6--gcc--12.2.0


cd ../

# create datafile
echo "#n_procs,init,communication,computation" > profiling/times_MPI-RMA.csv

# compile program
srun -n 1 -N 1 mpicc -fopenmp -DOPENMP -DTIME src/functions.c src/jacobi_MPI-RMA.c -o jacobi_MPI-RMA.x

# run program
for ((nprocs = 1; nprocs <= 32; nprocs *= 2))
do
	echo -n "$nprocs," >> profiling/times_MPI-RMA.csv
	mpirun -np "$nprocs" --map-by socket:PE=$num_threads --report-bindings ./jacobi_MPI-RMA.x $mat_size 10 11 4 
done

# remove executable
srun -n 1 -N 1 rm jacobi_MPI-RMA.x

cd batch_scripts/ || exit
