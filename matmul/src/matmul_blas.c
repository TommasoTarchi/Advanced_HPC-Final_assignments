/*
 * matrix-matrix multiplication using dgemm() function of 
 * openBLAS/MKL library
 *
 * N is the side of the matrices and can be passed during
 * compilation using -DMAT_SIZE=<desired_value>
 *
 * to test correctness of matmul compile with -DTEST: 
 * matrices A, B and C will be dumped to binary files in 
 * the test/ folder
 *
 * to profile the code compile with -DTIME: times for 
 * matrix initialization, communications and computations
 * will be printed to CSV file called times_blas.csv in 
 * profiling/ folder
 *
 * to speed up matrices initialization using openMP 
 * compile function.c with -fopenmp and -DOPENMP
 *
 * */


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <mkl.h>
#include "functions.h"


#ifndef MAT_SIZE
    #define MAT_SIZE 100 // Default value if not defined during compilation
#endif


int main(int argc, char** argv) {

    int N = MAT_SIZE;

    int my_rank, n_procs;

    // variables for timing
    //
    // (for clarity, we use t1, t2 to time initialization, t3, t4
    // to time communications, t5, t6 to time actual computation
    // and t7, t8 to time host-device communications)
#ifdef TIME
    double t1, t2, t3, t4, t5, t6, t7, t8, t_comm = 0, t_comp = 0, t_host_dev = 0;
#endif

    // init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    // compute local matrices size
    int N_loc_short = N / n_procs;
    int N_loc_long = N_loc_short + 1;
    int N_rest = N % n_procs;
    int N_loc;
    if (my_rank < N_rest)
        N_loc = N_loc_long;
    else
        N_loc = N_loc_short;

    // define array to store sizes pf blocks to be received
    int* counts_recv = (int*) malloc(n_procs * sizeof(int));
    for (int count=0; count<N_rest; count++)
        counts_recv[count] = N_loc_long*N_loc_long;
    for (int count=N_rest; count<n_procs; count++) {
        if (N_rest)
            counts_recv[count] = N_loc_short*N_loc_long;
        else
            counts_recv[count] = N_loc_short*N_loc_short;
    }

    // define array with positions of blocks to be received
    int* displacements = (int*) malloc(n_procs * sizeof(int));
    displacements[0] = 0;
    int while_count = 1;
    while (while_count < n_procs) {
        displacements[while_count] = displacements[while_count-1] + counts_recv[while_count-1];
        while_count++;
    }
    
    // allocate local matrices
    double* A = (double*) malloc(N_loc * N * sizeof(double));
    double* B = (double*) malloc(N_loc * N * sizeof(double));
    double* C = (double*) malloc(N_loc * N * sizeof(double));

    // compute global seed and broadcast to all processes
    unsigned int my_seed;
    if (my_rank == 0) {
        double current_time = MPI_Wtime();
        my_seed = (unsigned int) (current_time + 1);  // '+1' needed because seeds 0 and 1 give same random seq.
    }
    MPI_Bcast(&my_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

#ifdef TIME
    t1 = MPI_Wtime();
#endif

    // initialize A and B using different seed for each thread
    // (and different for A and B)
    int n_threads = random_mat(A, N_loc*N, my_seed, my_rank);
    my_seed += (unsigned int) n_procs * (unsigned int) n_threads;
    random_mat(B, N_loc*N, my_seed, my_rank);

#ifdef TIME
    t2 = MPI_Wtime();
#endif

    // for testing correctness of matmul
#ifdef TEST
    if (my_rank == 0) {
        FILE* file = fopen("test_matmul/A_blas.bin", "wb");
        fwrite(A, sizeof(double), N_loc*N, file);
        fclose(file);
        file = fopen("test_matmul/B_blas.bin", "wb");
        fwrite(B, sizeof(double), N_loc*N, file);
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int count=1; count<n_procs; count++) {
        if (my_rank == count) {
            FILE* file = fopen("test_matmul/A_blas.bin", "ab");
            fwrite(A, sizeof(double), N_loc*N, file);
            fclose(file);
            file = fopen("test_matmul/B_blas.bin", "ab");
            fwrite(B, sizeof(double), N_loc*N, file);
            fclose(file);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    // define quantities for blocks computation
    int offset = 0;  // offset of C blocks
    int N_rows = N_loc;
    int N_cols = N_loc_long;

    // allocate auxiliary matrices
    double* B_block = (double*) malloc(N_rows * N_cols * sizeof(double));  // matrix to store process's block
    double* B_col = (double*) malloc(N * N_cols * sizeof(double));  // matrix to store received blocks
    
    // set number of openBLAS/MKL threads
    //openblas_set_num_threads(n_threads);
    mkl_set_num_threads(n_threads);

    for (int count=0; count<n_procs; count++) {

        if (count == N_rest) {
            // update number of columns and reallocate auxiliary matrices
            N_cols = N_loc_short;
            B_block = (double*) realloc(B_block, N_rows * N_cols * sizeof(double));
            B_col = (double*) realloc(B_col, N * N_cols * sizeof(double));

            // update count_recv and displacements arrays
            for (int count2=0; count2<N_rest; count2++)
                counts_recv[count2] = N_loc_long*N_loc_short;
            for (int count2=N_rest; count2<n_procs; count2++)
                counts_recv[count2] = N_loc_short*N_loc_short;  // not changed in case of zero rest
            while_count = 1;
            while (while_count < n_procs) {
                displacements[while_count] = displacements[while_count-1] + counts_recv[while_count-1];
                while_count++;
            }
        }

#ifdef TIME
        t3 = MPI_Wtime();
#endif

        // create block to send to other processes
        create_block(B, B_block, N_rows, N_cols, offset, N);

        // send and receive blocks
        MPI_Allgatherv(B_block, N_rows*N_cols, MPI_DOUBLE, B_col, counts_recv, displacements, MPI_DOUBLE, MPI_COMM_WORLD);

#ifdef TIME
        t4 = MPI_Wtime();
        t_comm += t4 - t3;
#endif

#ifdef TIME
        t5 = MPI_Wtime();
#endif

        // matmul
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_rows, N_cols, N, 1.0, A, N, B_col, N_cols, 0.0, &C[offset], N);
        
#ifdef TIME
        t6 = MPI_Wtime();
        t_comp += t6 - t5;
#endif

        // update offset of C blocks
        offset += N_cols;
    }

    // for testing correctness of matmul
#ifdef TEST
    if (my_rank == 0) {
        FILE* file = fopen("test_matmul/C_blas.bin", "wb");
        fwrite(C, sizeof(double), N_loc*N, file);
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int count=1; count<n_procs; count++) {
        if (my_rank == count) {
            FILE* file = fopen("test_matmul/C_blas.bin", "ab");
            fwrite(C, sizeof(double), N_loc*N, file);
            fclose(file);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    free(counts_recv);
    free(displacements);
    free(A);
    free(B);
    free(C);
    free(B_block);
    free(B_col);
    
    // gather measured times and print them
#ifdef TIME
    double* times;

    if (my_rank == 0)
        times = (double*) malloc(n_procs * 4 * sizeof(double));    
    else
        times = (double*) malloc(4 * sizeof(double));

    times[0] = t2 - t1;  // time for initialization
    times[1] = t_comm;  // time for communications
    times[2] = t_comp;  // time for computation
    times[3] = t_host_dev;  // time for host-device communications
    
    MPI_Gather(times, 4, MPI_DOUBLE, times, 4, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        char csv_name[] = "profiling/times_blas.csv";
        save_time(times, csv_name, n_procs);
    }

    free(times);
#endif

    MPI_Finalize();

    return 0;
}
