#!/bin/bash
#SBATCH --job-name=matmul_cublas
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


# set matrix size
mat_size=20


module load cuda/12.1
module load nvhpc/23.11
module load openmpi/4.1.6--gcc--12.2.0
module load openblas/0.3.24--nvhpc--23.11


cd ../

echo "#init,communication,computation" > profiling/times_cublas.csv
echo "#,," >> profiling/times_cublas.csv

srun -n 1 nvcc -lm src/functions.c -lmpi -lcublas -lcudart -L/leonardo/prod/opt/libraries/openmpi/4.1.6/nvhpc--23.11/lib/ -L/leonardo/prod/opt/compilers/cuda/12.1/none src/matmul_cublas.c -DMAT_SIZE=$mat_size -DTIME -DTEST -o matmul.x

for ((nprocs = 1; nprocs <= 8; nprocs *= 2))
do
    mpirun -np "$nprocs" ./matmul.x
done

rm matmul.x

cd batch_scripts/ || exit
