#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"


#define N 10
#define N_BLOCKS_Y 5  // GPU blocks along y
#define N_BLOCKS_X 5  // GPU blocks along x


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

// kernel for accumulation of C
__global__ void accumulate(double* mat, double* mat_block, int dim_y, int dim_x, int offset, int jump) {

    // get ID of GPU thread
    int id_y = blockIdx.y*blockDim.y + threadIdx.y;
    int id_x = blockIdx.x*blockDim.x + threadIdx.x;
    
    // compute indexes on mat_block and mat
    int idx_block = id_y*dim_x + id_x;
    int idx = offset + id_y*jump + id_x;

    // copy data
    mat[idx] = mat_block[idx_block];
}


int main(int argc, char** argv) {

    int my_rank, n_procs;

    // init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    // init cublas
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    // set devices and defines GPU threads 
    //
    //
    //

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
    
    // allocate needed local matrices on device and copy data
    double* A_dev;
    double* C_dev;
    cudaMalloc((void**) &A_dev, N_loc * N * sizeof(double));
    cudaMemcpy(A_dev, A, N_loc * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc((void**) &C_dev, N_loc * N * sizeof(double));

    // initialize A and B with personal seeds
    double current_time = MPI_Wtime();
    unsigned int my_seed = (unsigned int) (current_time + my_rank + 1);  // '+1' needed because seeds 0 and 1 give same results
    random_mat(A, N_loc*N, my_seed);
    my_seed += n_procs;
    random_mat(B, N_loc*N, my_seed);

    //////////////////////////////////////////////////////////////
    if (my_rank == 0) {
        FILE* file = fopen("A3.bin", "wb");
        fwrite(A, sizeof(double), N_loc*N, file);
        fclose(file);
        file = fopen("B3.bin", "wb");
        fwrite(B, sizeof(double), N_loc*N, file);
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int count=1; count<n_procs; count++) {
        if (my_rank == count) {
            FILE* file = fopen("A3.bin", "ab");
            fwrite(A, sizeof(double), N_loc*N, file);
            fclose(file);
            file = fopen("B3.bin", "ab");
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

        // allocate auxiliary matrices on device
        double* B_col_dev;
        double* C_block;
        cudaMalloc((void**) &B_col_dev, N * N_cols * sizeof(double));
        cudaMalloc((void**) &C_block, N_rows * N_cols * sizeof(double));

        // create block to send to other processes
        create_block(B, B_block, N_rows, N_cols, offset, N);

        // send and receive blocks
        MPI_Allgatherv(B_block, N_rows*N_cols, MPI_DOUBLE, B_col, counts_recv, displacements, MPI_DOUBLE, MPI_COMM_WORLD);

        // copy gathered data to device
        cudaMemcpy(B_col_dev, B_col, N*N_cols*sizeof(double), cudaMemcpyHostToDevice);

        ///////////////////////////////////////////////////////////////////////////
        //printf("I'm %d of %d\n", my_rank, n_procs);
        //MPI_Barrier(MPI_COMM_WORLD);
        //////////////////////////////////////////////////////////////////////////

        // matmul
        const double alpha = 1.0;
        const double beta = 0.0;
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N_rows, N_cols, N, &alpha, A_dev, N_rows, B_col_dev, N, &beta, C_block, N_rows);

        // accumulate results on C
        dim3 grid, block;
        grid.y = N_BLOCKS_Y;
        grid.x = N_BLOCKS_X;
        block.y = N_cols / grid.y;
        block.x = N_rows / grid.x;
        accumulate<<<grid, block>>>();

        // update offset of C blocks
        offset += N_cols;

        cudaFree(B_col_dev);
        cudaFree(C_block);
    }

    // copy accumulated computation from device to host
    cudaMemcpy(C, C_dev, N_loc * N * sizeof(double), cudaMemcpyDeviceToHost);
    
    //////////////////////////////////////////////////////////////
    if (my_rank == 0) {
        FILE* file = fopen("C3.bin", "wb");
        fwrite(C, sizeof(double), N_loc*N, file);
        fclose(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    for (int count=1; count<n_procs; count++) {
        if (my_rank == count) {
            FILE* file = fopen("C3.bin", "ab");
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
    cudaFree(A_dev);
    cudaFree(C_dev);

    cublasDestroy(handle);

    MPI_Finalize();

    return 0;
}
