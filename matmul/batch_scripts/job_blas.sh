#!/bin/bash
#SBATCH --job-name=matmul_blas
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=11
#SBATCH --partition=dcgp_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


# set matrix size
mat_size=1200

# set number of openMP threads per process
export OMP_NUM_THREADS=10

# set number of BLAS threads
export OPENBLAS_NUM_THREADS=10

# set openMP binding policy (each thread on a different core)
export OMP_PROC_BIND=close
export OMP_PLACES=cores


# load modules
module load openmpi/4.1.6--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0


cd ../

# create datafile
echo "#n_procs,init,communication,computation" > profiling/times_blas.csv

# compile program
srun -n 1 -N 1 mpicc -fopenmp -lm src/functions.c -lopenblas src/matmul_blas.c -DMAT_SIZE=$mat_size -DTIME -DTEST -DOPENMP -o matmul_blas.x

# run program (each process will be placed on a different socket)
for ((nprocs = 1; nprocs <= 32; nprocs *= 2))
do
    echo -n "$nprocs," >> profiling/times_blas.csv
    mpirun -np "$nprocs" --map-by socket:PE=10 --report-bindings ./matmul_blas.x
done

# remove executable
srun -n 1 -N 1 rm matmul_blas.x

cd batch_scripts/ || exit
