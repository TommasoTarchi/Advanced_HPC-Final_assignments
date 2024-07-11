# Final assignments for advanced HPC class

This is the final assignment for the exam of advanced high performance computing at
University of Trieste.

It consists of three HPC exercises on three different topics: **CUDA** (matrix-matrix
multiplication), **OpenACC** (parallel Jacobi) and **MPI Remote Memory Access** (parallel Jacobi). Details
on the assignment can be found at [this repository](https://github.com/Foundations-of-HPC/Advanced-High-Performance-Computing-2023/tree/main).

All codes are parallelized using MPI.

All scripts were run on *Leonardo*, cluster hosted at *CINECA* (Bologna, Italy).

Details on my solution (both algorithm and implmentation) can be found in the [report](Report.pdf).


## What you will find in this repository

This repo contains:
- this README file
- `Jacobi/`: directory containing all codes and results related to the CUDA assignment:
  - `batch_scripts/`: directory containing batch files to run the scaling on Leonardo
  - `profiling/`: directory containing profiling data and python script to plot them
  - `plot/`: directory containing final state of the system and scripts to plot it using gnuplot
  - `src/`: directory containing source code for parallel Jacobi, in particular:
    - `functions.c`: functions for parallel jacobi
    - `functions.h`: header file
    - `jacobi.c`: parallel MPI C code for Jacobi accelerated with OpenACC (OpenMP is used to
      further parallelize system initialization on host)
    - `jacobi_aware.c`: same as above but with CUDA-aware MPI communications
    - `jacobi_MPI-RMA.c`: parallel C code for Jacobi with openMP and *Remote Memory Access* MPI
- `matmul/`:
  - `batch_scripts/`: directory containing batch files to run the scaling on Leonardo
  - `profiling/`: directory containing profiling data and python script to plot them
  - `src/`: directory containing source code for parallel matmul, in particular:
    - `functions.c`: functions for parallel matmul (function for initialization of matrices is
      further parallelized using OpenMP)
    - `functions.h`: header file
    - `matmul_simple.c`: parallel MPI C code for matmul using multiplication implemented "by hand"
    - `matmul_blas.c`: parallel MPI C code for matmul using **BLAS**
    - `matmul_cublas.c`: parallel MPI CUDA code for matmul using **cuBLAS**
  - `test_matmul/`: directory containing dumped matrices and code to test matmul correctness
- `Report.pdf`: brief report about the assignment with more detailed description of algorithm and
  its implementation


## How to reproduce data on Leonardo

To reproduce the scaling studies you can follow these simple steps:

1. Clone this repository on *Lonardo*:
   ````
   $ git clone git@github.com:TommasoTarchi/Advanced_HPC-Final_assignments.git
   ````

2. Go to `Jacobi/` or `matmul/` directory depending on which results you are interested in reproducing:
   ````
   $ cd <Jacobi/matmul>
   ````

4. Go to `batch_scripts/` directory:
   ````
   $ cd batch_scripts
   ````

5. Run the scaling you are interested in:
   ````
   $ sbatch <batch_file_name>
   ````

6. Go to profiling directory:
   ````
   $ cd ../profiling
   ````

7. Build plots using `make_plot.py`, setting `--mode` option to the scaling you ran:
   ````
   $ python3 make_plot.py --mode <mode_option>
   ````
   Available mode options are:
   - for Jacobi: `openMP`, `openACC`, `aware`, `MPI-RMA`;
   - for matmul: `simple`, `blas`, `cublas`.
