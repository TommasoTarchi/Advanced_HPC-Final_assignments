#!/bin/bash
#SBATCH --job-name=matmul_simple
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --partition=dcgp_usr_prod
#SBATCH -A ict24_dssc_cpu
#SBATCH --output=report_simple.out


# choose matrix size and number of threads
mat_size=10000
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
echo "#n_procs,init,communication,computation" > profiling/times_simple.csv

# compile program
srun -n 1 -N 1 mpicc -fopenmp -lm src/functions.c src/matmul_simple.c -DOPENMP -DTIME -DTEST -DMAT_SIZE=$mat_size -o matmul_simple.x

# run program (each process will be placed on a different socket)
for ((nprocs = 1; nprocs <= 32; nprocs *= 2))
do
    echo -n "$nprocs," >> profiling/times_simple.csv
    mpirun -np "$nprocs" --map-by socket:PE=$num_threads --report-bindings ./matmul_simple.x
    echo
done

# remove executable
srun -n 1 -N 1 rm matmul_simple.x

cd batch_scripts/ || exit
