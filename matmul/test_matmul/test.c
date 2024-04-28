/* 
 * this program reads three matrices A, B, C from corresponding binary 
 * files and checks whether A * B = C (row-column product)
 *
 * if compiled with -DPRINT it will print the upper left submatrix of
 * size FIRST_N
 * 
 * */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define MATMUL 2  // 0 for simple matmul, 1 for blas, 2 for cublas
#define N 27  // size of matrices
#define PRINT_N 10  // size of submatrix to be printed
#define EPS 1e-6  // tolerance in comparison between C and C_check elements


#if MATMUL == 0
    #define A_BIN "A_simple.bin"
    #define B_BIN "B_simple.bin"
    #define C_BIN "C_simple.bin"
#elif MATMUL == 1
    #define A_BIN "A_blas.bin"
    #define B_BIN "B_blas.bin"
    #define C_BIN "C_blas.bin"
#elif MATMUL == 2
    #define A_BIN "A_cublas.bin"
    #define B_BIN "B_cublas.bin"
    #define C_BIN "C_cublas.bin"
#else
    #error "MATMUL value must be 0, 1, or 2"
#endif


int main() {

    if (MATMUL == 0)
	printf("testing simple matmul...\n");
    else if (MATMUL == 1)
	printf("testing matmul using BLAS...\n");
    else if (MATMUL == 2)
	printf("testing matmul using cuBLAS...\n");


    // allocate matrices
    double* A = (double*) malloc(N * N * sizeof(double));
    double* B = (double*) malloc(N * N * sizeof(double));
    double* C = (double*) malloc(N * N * sizeof(double));
    double* C_check = (double*) malloc(N * N * sizeof(double));  // correct matrix

    // read output matrices of the parallel program
    FILE* file;
    file = fopen(A_BIN, "rb");
    fread(A, sizeof(double), N * N, file);
    fclose(file);
    file = fopen(B_BIN, "rb");
    fread(B, sizeof(double), N * N, file);
    fclose(file);
    file = fopen(C_BIN, "rb");
    fread(C, sizeof(double), N * N, file);
    fclose(file);

    // compute correct matrix-matrix multiplication result
    for (int row=0; row<N; row++) {
	for (int col=0; col<N; col++) {
	    double acc = 0;
	    for (int i=0; i<N; i++)
		acc += A[row*N + i] * B[col + i*N];
	    C_check[row*N + col] = acc;
	}
    }

    // comparison of results
    int error_counter = 0;
    for (int i=0; i<N*N; i++)
        if (fabs(C[i] - C_check[i]) > EPS)
            error_counter++;

    // print result
    printf("errors: %d out of %d elements\n\n", error_counter, N*N);

#ifdef PRINT
    // print first elements of matrices
    int first_n = PRINT_N;
    for (int i=0; i<first_n; i++) {
	for (int j=0; j<first_n; j++)
	    printf("%f  ", A[i*N + j]);
	printf("\n");
    }
    printf("\n\n");
    for (int i=0; i<first_n; i++) {
	for (int j=0; j<first_n; j++)
	    printf("%f  ", B[i*N + j]);
	printf("\n");
    }
    printf("\n\n");
    for (int i=0; i<first_n; i++) {
	for (int j=0; j<first_n; j++)
	    printf("%f  ", C[i*N + j]);
	printf("\n");
    }
    printf("\n\n");
    for (int i=0; i<first_n; i++) {
	for (int j=0; j<first_n; j++)
	    printf("%f  ", C_check[i*N + j]);
	printf("\n");
    }
#endif
    
    free(A);
    free(B);
    free(C);
    free(C_check);

    return 0;
}
