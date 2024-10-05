#!/bin/bash
#SBATCH --job-name=jacobi_unaware
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --time=2:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report_unaware.out


# choose matrix size and number of threads
mat_size=12000
num_threads=7


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
echo "#n_procs,init,communication,computation,host_dev_once,host_dev_iter" > profiling/times_unaware.csv

# compile program
srun -n 1 -N 1 mpicc -acc=noautopar -Minfo=all -fopenmp -DOPENMP -DOPENACC -DTIME src/functions.c src/jacobi.c -o jacobi.x

# run program
for ((nprocs = 1; nprocs <= 32; nprocs *= 2))
do
	echo -n "$nprocs," >> profiling/times_unaware.csv
	mpirun -np "$nprocs" --map-by ppr:4:node:PE=$num_threads --report-bindings ./jacobi.x $mat_size 10 11 4 
done

# remove executable
srun -n 1 -N 1 rm jacobi.x

cd batch_scripts/ || exit
