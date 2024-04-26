#!/bin/bash
#SBATCH --job-name=first_assignment-2
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


module load openmpi/4.1.6--gcc--12.2.0
module load openblas/

srun -n 1 mpicc -lm -lopenblas functions.c first_assignment-2.c -o first_assignment-2.x

mpirun ./first_assignment-2.x

rm first_assignment-2.x
