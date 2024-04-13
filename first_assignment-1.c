#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


#define N 102


// function to randomly initialize matrices
void random_mat(double* mat, int mat_size) {

    for (int i=0; i<mat_size; i++)
        mat[i] = (double) rand() / 100.;
}

// function to create blocks to send to other processes
// 
// 'offset' is the first element of block and 'jump' is the distance between
// first elements of each row of the block
void create_block(double* mat, double* block, int block_y, int block_x, int offset, int jump) {

    // copy data in block
    for (int row=0; row<block_y; row++) {
        int row_offset = offset + row*jump;
        
        for (int i=0; i<block_x; i++)
            block[i] = mat[row_offset+i];
    }
}


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

    // initialize A and B
    random_mat(A, N_loc*N);
    random_mat(B, N_loc*N);

    //////////////////////////////////////////////////////////////
    if (my_rank == 0) {
        FILE* file = fopen("A.bin", "wb");
        fwrite(A, sizeof(double), N*N, file);
        fclose(file);
        file = fopen("B.bin", "wb");
        fwrite(B, sizeof(double), N*N, file);
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int count=1; count<n_procs; count++) {
        if (my_rank == count) {
            FILE* file = fopen("A.bin", "ab");
            fwrite(A, sizeof(double), N*N, file);
            fclose(file);
            file = fopen("B.bin", "ab");
            fwrite(B, sizeof(double), N*N, file);
            fclose(file);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    //////////////////////////////////////////////////////////////

    // define quantities for blocks computation
    int offset = 0;
    int N_rows = N_loc;
    int N_cols = N_loc_long;

    // allocate auxiliary matrices
    double* B_block = (double*) malloc(N_rows * N_cols * sizeof(double));  // matrix to store process's block
    double* B_col = (double*) malloc(N * N_cols * sizeof(double));  // matrix to store received blocks

    for (int count=0; count<n_procs; count++) {
	
	if (count == N_rest) {
            // update number of columns and reallocate auxiliary matrices
	    N_cols = N_loc_short;
            B_block = (double*) realloc(B_block, N_rows * N_cols * sizeof(double));
            B_col = (double*) realloc(B_col, N * N_cols * sizeof(double));

	    // update count_recv and displacements arrays
	    for (int count=0; count<N_rest; count++)
		counts_recv[count] = N_loc_long*N_loc_short;
	    for (int count=N_rest; count<n_procs; count++) {
		if (N_rest)
		    counts_recv[count] = N_loc_short*N_loc_short;
		else
		    counts_recv[count] = N_loc_short*N_loc_short;  // not changed in case of zero rest
	    }
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

    printf("I'm %d of %d\n", my_rank, n_procs);
    MPI_Barrier(MPI_COMM_WORLD);

        // matmul ('row' and 'col' count rows and columns of the block of C)
        for (int row=0; row<N_rows; row++) {
            for (int col=0; col<N_cols; col++) {
                int acc=0;
                for (int k=0; k<N; k++)
                    acc += A[row*N + k] * B_col[k*N+col];
                C[offset + row*N + col] = acc;
            }
        }
    }

    //////////////////////////////////////////////////////////////
    if (my_rank == 0) {
        FILE* file = fopen("C.bin", "wb");
        fwrite(C, sizeof(double), N*N, file);
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int count=1; count<n_procs; count++) {
        if (my_rank == count) {
            FILE* file = fopen("C.bin", "ab");
            fwrite(C, sizeof(double), N*N, file);
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

    MPI_Finalize();

    return 0;
}
