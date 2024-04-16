#!/bin/bash
#SBATCH --job-name=first_assignment-1
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out


module load openmpi/4.1.6--gcc--12.2.0

srun -n 1 mpicc -lm first_assignment-1.c -o first_assignment-1.x

mpirun ./first_assignment-1.x

rm first_assignment-1.x
