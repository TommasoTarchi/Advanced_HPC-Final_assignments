#!/bin/bash
#SBATCH --job-name=first_assignment-2
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --partition=boost_usr_prod
#SBATCH -A ict24_dssc_gpu
#SBATCH --output=report.out

module load openmpi/

mpirun ./first_assignment-2.x
