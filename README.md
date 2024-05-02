# Final assignments for advanced HPC class

This is the final assignment for the exam of advanced high performance computing at
University of Trieste.

It consists of three HPC exercises on three different topics: **CUDA** (matrix-matrix
multiplication), **OpenACC** (parallel Jacobi) and **OpenMP tasks** (*to be defined*). Details can be found at 
[this repository](https://github.com/Foundations-of-HPC/Advanced-High-Performance-Computing-2023/tree/main).

All codes are paralelized using MPI.

All codes were run on *Leonardo*, cluster hosted at *CINECA* (Bologna, Italy).


## What you will find in this repository

This repo contains:
- this README file
- `Jacobi/`: directory containing all codes and results related to the CUDA assignment:
  - `profiling/`: directory containing profiling data and python script to plot them
  - `plot/`: directory containing final state of the system and scripts to plot it using gnuplot
  - `src/`: directory containing source code for parallel Jacobi, in particular:
    - `functions.c`: functions for parallel jacobi
    - `functions.h`: header file
    - `jacobi.c`: parallel MPI C code for Jacobi accelerated with OpenACC (OpenMP is used to
      further parallelize system initialization on host)
  - `job.sh`: batch file to run the scaling on Leonardo
- `matmul/`:
  - `batch_scripts/`: directory containing batch files to run the scaling on Leonardo
  - `profiling/`: directory containing profiling data and python script to plot them
  - `src/`: directory containing source code for parallel matmul, in particular:
    - `functions.c`: functions for parallel matmul (function for initialization of matrices is
      further parallelized using OpenMP)
    - `functions.h`: header file
    - `matmul_blas.c`: parallel MPI C code for matmul using multiplication implemented "by hand"
    - `matmul_cublas.c`: parallel MPI C code for matmul using **BLAS**
    - `matmul_simple.c`: parallel MPI CUDA code for matmul using **cuBLAS**
  - `test_matmul/`: directory containing dumped matrices and code to test matmul correctness
- `Report.pdf`: brief report about the assignment with more detailed description of algorithm and
  its implementation


## How to reproduce data on Leonardo
