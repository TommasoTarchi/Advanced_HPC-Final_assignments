/* 
 * this program reads three matrices A, B, C from corresponding binary 
 * files and checks whether A * B = C (row-column product)
 * 
 * */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define N 8
#define PRINT_N 8
#define EPS 1e-6  // tolerance in comparison between C and C_check elements


int main() {

    // allocate matrices
    double* A = (double*) malloc(N * N * sizeof(double));
    double* B = (double*) malloc(N * N * sizeof(double));
    double* C = (double*) malloc(N * N * sizeof(double));
    double* C_check = (double*) malloc(N * N * sizeof(double));  // correct matrix

    // read output matrices of the parallel program
    FILE* file;
    file = fopen("A3.bin", "rb");
    fread(A, sizeof(double), N * N, file);
    fclose(file);
    file = fopen("B3.bin", "rb");
    fread(B, sizeof(double), N * N, file);
    fclose(file);
    file = fopen("C3.bin", "rb");
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
    printf("errors in matrix-matrix product: %d / %d\n", error_counter, N*N);

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
