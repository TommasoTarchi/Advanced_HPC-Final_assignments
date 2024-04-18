#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cblas.h>
#include "functions.h"


#define N 10


int main(int argc, char** argv) {

    int my_rank, n_procs;

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

    // initialize A and B with personal seeds
    double current_time = MPI_Wtime();
    unsigned int my_seed = (unsigned int) (current_time + my_rank + 1);  // '+1' needed because seeds 0 and 1 give same results
    random_mat(A, N_loc*N, my_seed);
    my_seed += n_procs;
    random_mat(B, N_loc*N, my_seed);

    //////////////////////////////////////////////////////////////
    if (my_rank == 0) {
        FILE* file = fopen("A2.bin", "wb");
        fwrite(A, sizeof(double), N_loc*N, file);
        fclose(file);
        file = fopen("B2.bin", "wb");
        fwrite(B, sizeof(double), N_loc*N, file);
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int count=1; count<n_procs; count++) {
        if (my_rank == count) {
            FILE* file = fopen("A2.bin", "ab");
            fwrite(A, sizeof(double), N_loc*N, file);
            fclose(file);
            file = fopen("B2.bin", "ab");
            fwrite(B, sizeof(double), N_loc*N, file);
            fclose(file);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //////////////////////////////////////////////////////////////

    // define quantities for blocks computation
    int offset = 0;  // offset of C blocks
    int N_rows = N_loc;
    int N_cols = N_loc_long;

    // allocate auxiliary matrices
    double* B_block = (double*) malloc(N_rows * N_cols * sizeof(double));  // matrix to store process's block
    double* B_col = (double*) malloc(N * N_cols * sizeof(double));  // matrix to store received blocks
    //double* C_block = (double*) malloc(N_rows * N_cols * sizeof(double));  // matrix to store computed data

    for (int count=0; count<n_procs; count++) {
	
	if (count == N_rest) {
            // update number of columns and reallocate auxiliary matrices
	    N_cols = N_loc_short;
            B_block = (double*) realloc(B_block, N_rows * N_cols * sizeof(double));
            B_col = (double*) realloc(B_col, N * N_cols * sizeof(double));
	    //C_block = (double*) realloc(C_block, N_rows * N_cols * sizeof(double));

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

	///////////////////////////////////////////////////////////////////////////
        //printf("I'm %d of %d\n", my_rank, n_procs);
        //MPI_Barrier(MPI_COMM_WORLD);
	//////////////////////////////////////////////////////////////////////////

	// matmul
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N_rows, N_cols, N, 1.0, A, N, B_col, N_cols, 0.0, &C[offset], N);  // C, N_cols
        
        // copy result of computation to C
        //for (int row=0; row<N_rows; row++)
        //    for (int col=0; col<N_cols; col++)
        //        C[offset + row*N + col] = C_block[row*N_cols + col];

	// update offset of C blocks
	offset += N_cols;
    }

    //////////////////////////////////////////////////////////////
    if (my_rank == 0) {
        FILE* file = fopen("C2.bin", "wb");
        fwrite(C, sizeof(double), N_loc*N, file);
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int count=1; count<n_procs; count++) {
        if (my_rank == count) {
            FILE* file = fopen("C2.bin", "ab");
            fwrite(C, sizeof(double), N_loc*N, file);
            fclose(file);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //////////////////////////////////////////////////////////////

    free(counts_recv);
    free(displacements);
    free(A);
    free(B);
    free(C);
    free(B_block);
    free(B_col);
    //free(C_block);

    MPI_Finalize();

    return 0;
}
