/*
 * matrix-matrix multiplication implemented "by hand"
 *
 * N is the side of the matrices and can be passed during
 * compilation using -DMAT_SIZE=<desired_value>
 *
 * to test correctness of matmul compile with -DTEST: 
 * matrices A, B and C will be dumped to binary files in 
 * the test/ folder
 *
 * to time the code compile with -DTIME: times for 
 * matrix initialization, communications and computations
 * will be printed ...
 *
 * */


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "functions.h"


#ifndef MAT_SIZE
    #define MAT_SIZE 100 // Default value if not defined during compilation
#endif


int main(int argc, char** argv) {

    int N = MAT_SIZE;

    int my_rank, n_procs;

    // variables for timing
    //
    // (for clarity, we use t1 and t2 to time initialization,
    // t3 and t4 to time communications, and t5 and t6 to time
    // actual computation)
#ifdef TIME
    double t1, t2, t3, t4, t_comm = 0, t5, t6, t_comp = 0;
#endif

    // init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    //////////////////////////////////////////////////////////////
    printf("I'm %d of %d\n", my_rank, n_procs);
    //////////////////////////////////////////////////////////////

    // compute local matrices size
    int N_loc_short = N / n_procs;
    int N_loc_long = N_loc_short + 1;
    int N_rest = N % n_procs;
    int N_loc;
    if (my_rank < N_rest)
        N_loc = N_loc_long;
    else
        N_loc = N_loc_short;
    
#ifdef TIME
    t3 = MPI_Wtime();
#endif

    // define array to store sizes of blocks to be received
    //
    // (actually part of parallel communication process)
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
    //
    // (still part of parallel communication process)
    int* displacements = (int*) malloc(n_procs * sizeof(int));
    displacements[0] = 0;
    int while_count = 1;
    while (while_count < n_procs) {
        displacements[while_count] = displacements[while_count-1] + counts_recv[while_count-1];
        while_count++;
    }

#ifdef TIME
        t4 = MPI_Wtime();
        t_comm += t4 - t3;
#endif

#ifdef TIME
    t1 = MPI_Wtime();
#endif

    // allocate local matrices
    double* A = (double*) malloc(N_loc * N * sizeof(double));
    double* B = (double*) malloc(N_loc * N * sizeof(double));
    double* C = (double*) malloc(N_loc * N * sizeof(double));

    // initialize A and B with personal seeds
    double current_time = MPI_Wtime();
    unsigned int my_seed = (unsigned int) (current_time + my_rank + 1);  // '+1' needed because seeds 0 and 1 give same results
    random_mat(A, N_loc*N, my_seed);
    my_seed += n_procs;
    random_mat(B, N_loc*N, my_seed);

#ifdef TIME
    t2 = MPI_Wtime();
#endif

    // for testing correctness of matmul
#ifdef TEST
    if (my_rank == 0) {
        FILE* file = fopen("test_matmul/A_simple.bin", "wb");
        fwrite(A, sizeof(double), N_loc*N, file);
        fclose(file);
        file = fopen("test_matmul/B_simple.bin", "wb");
        fwrite(B, sizeof(double), N_loc*N, file);
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int count=1; count<n_procs; count++) {
        if (my_rank == count) {
            FILE* file = fopen("test_matmul/A_simple.bin", "ab");
            fwrite(A, sizeof(double), N_loc*N, file);
            fclose(file);
            file = fopen("test_matmul/B_simple.bin", "ab");
            fwrite(B, sizeof(double), N_loc*N, file);
            fclose(file);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif

    // define quantities for blocks computation
    int offset = 0;
    int N_rows = N_loc;
    int N_cols = N_loc_long;

    // allocate auxiliary matrices
    double* B_block = (double*) malloc(N_rows * N_cols * sizeof(double));  // matrix to store process's block
    double* B_col = (double*) malloc(N * N_cols * sizeof(double));  // matrix to store received blocks

    for (int count=0; count<n_procs; count++) {
	
#ifdef TIME
        t3 = MPI_Wtime();
#endif

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

        // matmul ('row' and 'col' count rows and columns of the block of C)
        for (int row=0; row<N_rows; row++) {
            for (int col=0; col<N_cols; col++) {
                double acc=0;
                for (int k=0; k<N; k++)
                    acc += A[row*N + k] * B_col[col + k*N_cols];
                C[offset + row*N + col] = acc;
            }
        }

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
        FILE* file = fopen("test_matmul/C_simple.bin", "wb");
        fwrite(C, sizeof(double), N_loc*N, file);
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int count=1; count<n_procs; count++) {
        if (my_rank == count) {
            FILE* file = fopen("test_matmul/C_simple.bin", "ab");
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
        times = (double*) malloc(n_procs * 3 * sizeof(double));    
    else
        times = (double*) malloc(3 * sizeof(double));

    times[0] = t2 - t1;  // time for initialization
    times[1] = t_comm;  // time for communications
    times[2] = t_comp;  // time for computation
    
    MPI_Gather(times, 3, MPI_DOUBLE, times, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {

        // compute average times
        double* total_times;
        total_times = (double*) malloc(3 * sizeof(double));
        total_times[0] = 0;
        total_times[1] = 0;
        total_times[2] = 0;
        for (int count=0; count<n_procs; count++) {
            total_times[0] += times[3 * count] / (double) n_procs;
            total_times[1] += times[1 + 3 * count] / (double) n_procs;
            total_times[2] += times[2 + 3 * count] / (double) n_procs;
        }

        // print times
        FILE* file = fopen("profiling/times_simple.csv", "a");
        fprintf(file, "%f,%f,%f\n", total_times[0], total_times[1], total_times[2]);
        fclose(file);

        free(total_times);
    }

    free(times);
#endif

    MPI_Finalize();

    return 0;
}
