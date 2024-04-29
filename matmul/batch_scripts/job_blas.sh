#!/bin/bash
#SBATCH --job-name=matmul_blas
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


# set matrix size
mat_size=20


module load openmpi/4.1.6--gcc--12.2.0
module load openblas/0.3.24--gcc--12.2.0


cd ../

echo "#init,communication,computation" > profiling/times_blas.csv
echo "#" >> profiling/times_blas.csv

srun -n 1 mpicc -lm src/functions.c -lopenblas src/matmul_blas.c -DMAT_SIZE=$mat_size -DTIME -DTEST -o matmul.x

for ((nprocs = 1; nprocs <= 8; nprocs *= 2))
do
    mpirun -np "$nprocs" ./matmul.x
done

rm matmul.x

cd batch_scripts/ || exit
