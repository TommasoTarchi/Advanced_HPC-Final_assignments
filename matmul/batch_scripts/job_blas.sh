#!/bin/bash
#SBATCH --job-name=matmul_blas
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --partition=dcgp_usr_prod
#SBATCH -A ict24_dssc_cpu
#SBATCH --output=report_blas.out


# choose matrix size and number of threads
mat_size=10000
num_threads=20


# set number of openMP threads per process
export OMP_NUM_THREADS=$num_threads

# set number of BLAS threads
#export OPENBLAS_NUM_THREADS=$num_threads

# set openMP binding policy (each thread on a different core)
export OMP_PROC_BIND=close
export OMP_PLACES=cores


# load modules
module load openmpi/4.1.6--gcc--12.2.0
#module load openblas/0.3.24--gcc--12.2.0
module load intel-oneapi-mkl


cd ../

# create datafile
echo "#n_procs,init,communication,computation,host_device" > profiling/times_blas.csv

# compile program
#srun -n 1 -N 1 mpicc -fopenmp -lpthread -lm src/functions.c -lopenblas src/matmul_blas.c -DMAT_SIZE=$mat_size -DTIME -DTEST -DOPENMP -o matmul_blas.x
srun -n 1 -N 1 mpicc -fopenmp -L$MKLROOT/lib/intel64 -lmkl_rt -lpthread -lm -ldl src/functions.c src/matmul_blas.c -DMAT_SIZE=$mat_size -DTIME -DTEST -DOPENMP -o matmul_blas.x

# run program (each process will be placed on a different socket)
for ((nprocs = 1; nprocs <= 32; nprocs *= 2))
do
    echo -n "$nprocs," >> profiling/times_blas.csv
    mpirun -np "$nprocs" --map-by socket:PE=$num_threads --report-bindings ./matmul_blas.x
    echo
done

# remove executable
srun -n 1 -N 1 rm matmul_blas.x

cd batch_scripts/ || exit
