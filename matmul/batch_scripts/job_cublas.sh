#!/bin/bash
#SBATCH --job-name=matmul_cublas
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report_cublas.out


# choose matrix size and number of threads
mat_size=1200
num_threads=7


# set number of openMP threads per process
export OMP_NUM_THREADS=$num_threads

# set openMP binding policy (each thread on a different core)
export OMP_PROC_BIND=close
export OMP_PLACES=cores


# load modules
module load cuda/12.1
module load gcc/12.2.0
module load nvhpc/23.11
module load openmpi/4.1.6--gcc--12.2.0
module load openblas/0.3.24--nvhpc--23.11


cd ../

# create datafile
echo "#n_procs,init,communication,computation" > profiling/times_cublas.csv

# compile program
srun -n 1 -N 1 gcc -fopenmp -lm -c src/functions.c -DOPENMP -o src/functions.o
srun -n 1 -N 1 nvcc -lgomp -lmpi -lcublas -lcudart src/functions.o -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib/ -L/leonardo/prod/opt/compilers/cuda/12.1/none src/matmul_cublas.c -DMAT_SIZE=$mat_size -DTIME -DTEST -o matmul_cublas.x

# run program
for ((nprocs = 1; nprocs <= 8; nprocs *= 2))
do
    echo -n "$nprocs," >> profiling/times_cublas.csv
    mpirun -np "$nprocs" --map-by ppr:4:node:PE=$num_threads --report-bindings ./matmul_cublas.x
    echo
done

# remove executable
srun -n 1 -N 1 rm matmul_cublas.x
srun -n 1 -N 1 rm src/functions.o

cd batch_scripts/ || exit
