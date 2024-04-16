#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


#define N 10 


// function to randomly initialize matrices
void random_mat(double* mat, int mat_size, unsigned int seed) {

    // set seed
    srand(seed);

    // set factor to obtain elements with at most an order of 
    // magnitude ~10^6 (to avoid overflow)
    double exp = (6. - log10((double) mat_size)) / 2.;
    double factor = pow(10., exp);

    for (int i=0; i<mat_size; i++)
        mat[i] = (double) rand() / (double) RAND_MAX * factor;
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
            block[row*block_x + i] = mat[row_offset + i];
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

    // initialize A and B with personal seeds
    double current_time = MPI_Wtime();
    unsigned int my_seed = (unsigned int) (current_time + my_rank + 1);  // '+1' needed because seeds 0 and 1 give same results
    random_mat(A, N_loc*N, my_seed);
    my_seed += n_procs;
    random_mat(B, N_loc*N, my_seed);

    //////////////////////////////////////////////////////////////
    if (my_rank == 0) {
        FILE* file = fopen("A.bin", "wb");
        fwrite(A, sizeof(double), N_loc*N, file);
        fclose(file);
        file = fopen("B.bin", "wb");
        fwrite(B, sizeof(double), N_loc*N, file);
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int count=1; count<n_procs; count++) {
        if (my_rank == count) {
            FILE* file = fopen("A.bin", "ab");
            fwrite(A, sizeof(double), N_loc*N, file);
            fclose(file);
            file = fopen("B.bin", "ab");
            fwrite(B, sizeof(double), N_loc*N, file);
            fclose(file);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //printf("I'm %d and I have: %f  %f\n", my_rank, A[0], B[0]);

    //printf("I'm %d and I have N_loc=%d and last seed: %d\n", my_rank, N_loc, my_seed);
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

        ///////////////////////////////////////////////////////////////////////////
        //if (my_rank == 0)
	//    printf("-- ITERATION %d --\n", count);
	//MPI_Barrier(MPI_COMM_WORLD);

        //printf("I'm %d and I have %d rows and %d columns\n", my_rank, N_rows, N_cols);
        //MPI_Barrier(MPI_COMM_WORLD);
	
	//printf("I'm %d and my disp is: %d, %d, %d, %d\n", my_rank, displacements[0], displacements[1], displacements[2], displacements[3]);
	//printf("I'm %d and my sizes are: %d, %d, %d, %d\n", my_rank, counts_recv[0], counts_recv[1], counts_recv[2], counts_recv[3]);
	//MPI_Barrier(MPI_COMM_WORLD);
        ///////////////////////////////////////////////////////////////////////////

        // create block to send to other processes
        create_block(B, B_block, N_rows, N_cols, offset, N);

        // send and receive blocks
        MPI_Allgatherv(B_block, N_rows*N_cols, MPI_DOUBLE, B_col, counts_recv, displacements, MPI_DOUBLE, MPI_COMM_WORLD);

	/////////////////////////////////////////////////////////////////////////////
	//printf("I'm %d of %d\n", my_rank, n_procs);
	//MPI_Barrier(MPI_COMM_WORLD);
	/////////////////////////////////////////////////////////////////////////////

        // matmul ('row' and 'col' count rows and columns of the block of C)
        for (int row=0; row<N_rows; row++) {
            for (int col=0; col<N_cols; col++) {
                double acc=0;
                for (int k=0; k<N; k++)
                    acc += A[row*N + k] * B_col[col + k*N_cols];
                C[offset + row*N + col] = acc;
            }
        }

	// update offset of C blocks
	offset += N_cols;
    }

    //////////////////////////////////////////////////////////////
    if (my_rank == 0) {
        FILE* file = fopen("C.bin", "wb");
        fwrite(C, sizeof(double), N_loc*N, file);
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int count=1; count<n_procs; count++) {
        if (my_rank == count) {
            FILE* file = fopen("C.bin", "ab");
            fwrite(C, sizeof(double), N_loc*N, file);
            fclose(file);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //printf("I'm %d and I have: %f\n", my_rank, C[0]);
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
