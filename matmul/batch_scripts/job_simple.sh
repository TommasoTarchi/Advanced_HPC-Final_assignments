#!/bin/bash
#SBATCH --job-name=first_assignment-1
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


# set matrix size
mat_size=20


module load openmpi/4.1.6--gcc--12.2.0


cd ../

echo "#init,communication,computation" > profiling/times_simple.csv
echo "#,," >> profiling/times_simple.csv

srun -n 1 mpicc -lm src/functions.c src/matmul_simple.c -DTIME -DTEST -DMAT_SIZE=$mat_size -o matmul_simple.x

for ((nprocs = 1; nprocs <= 8; nprocs *= 2))
do
    mpirun -np "$nprocs" ./matmul_simple.x
done

rm matmul_simple.x

cd batch_scripts/ || exit
