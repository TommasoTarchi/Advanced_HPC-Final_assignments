#include <stdlib.h>
#include <mpi.h>


#define N 1000


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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    // compute local matrixes size
    int N_loc = N / n_procs;
    int N_rest = N % n_procs;
    if (my_rank < N_rest)
        N_loc++;

    // define array to store sizes of blocks to be received
    int* counts_recv = (int*) malloc(n_procs * sizeof(int));
    if (my_rank < N_rest && N_rest != 0) {
        for (int count=0; count<N_rest; count++)
            counts_recv[count] = N_loc * N;
        for (int count=N_rest; count<n_procs; count++)
            counts_recv[count] = (N_loc-1) * N;
    } else {
        for (int count=0; count<N_rest; count++)
            counts_recv[count] = (N_loc+1) * N;
        for (int count=N_rest; count<n_procs; count++)
            counts_recv[count] = N_loc * N;
    }

    // define displacement array (indicates where received block must be placed)
    int* displacements = (int*) malloc(n_procs * sizeof(int));
    if (my_rank < N_rest && N_rest != 0) {
        for (int count=0; count<N_rest+1; count++)
            displacements[count] = count * N_loc;
        for (int count=N_rest+1; count<n_procs; count++)
            displacements[count] = N_rest*N_loc + (count-N_rest-1)*(N_loc-1);
    } else {
        for (int count=0; count<N_rest; count++)
            displacements[count] = count * (N_loc+1);
        for (int count=N_rest; count<n_procs; count++)
            displacements[count] = N_rest*(N_loc+1) + (count-N_rest-1)*N_loc;
    }

    // allocate local matrices
    double* A = (double*) malloc(N_loc * N * sizeof(double));
    double* B = (double*) malloc(N_loc * N * sizeof(double));
    double* C = (double*) malloc(N_loc * N * sizeof(double));

    // initialize A and B
    random_mat(A, N_loc*N);
    random_mat(B, N_loc*N);

    // define quantities for blocks computation
    int offset = 0;  // first element of C to be updated
    int N_rows = N_loc;  // number of rows in C block
    int N_cols = N / n_procs;  // number of columns in C block
    if (N_rest != 0)
        N_cols++;

    // allocate auxiliary matrices
    double* B_block = (double*) malloc(N_rows * N_cols * sizeof(double));  // matrix to store process's block
    double* B_col = (double*) malloc(N * N_cols * sizeof(double));  // matrix to store received blocks

    for (int count=0; count<n_procs; count++) {

        // update number of columns and reallocate auxiliary matrices
        if (count == N_rest && N_rest != 0) {
            N_cols--;
            B_block = (double*) realloc(B_block, N_rows * N_cols * sizeof(double));
            B_col = (double*) realloc(B_col, N * N_cols * sizeof(double));
        }

        // create block to send to other processes
        create_block(B, B_block, N_rows, N_cols, offset, N);

        // send and receive blocks
        MPI_Allgatherv(B_block, N_loc*N_loc, MPI_DOUBLE, B_col, counts_recv, displacements, MPI_DOUBLE, MPI_COMM_WORLD);

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
