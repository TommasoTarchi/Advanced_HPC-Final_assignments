/* 
 * this program reads three matrices A, B, C from corresponding binary 
 * files and checks whether A * B = C (row-column product)
 * 
 * */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#define N 100
#define EPS 1e-9  // tolerance in comparison between C and C_check elements


int main() {

    // allocate matrices
    double* A = (double*) malloc(N * N * sizeof(double));
    double* B = (double*) malloc(N * N * sizeof(double));
    double* C = (double*) malloc(N * N * sizeof(double));
    double* C_check = (double*) malloc(N * N * sizeof(double));  // correct matrix

    // read output matrices of the parallel program
    FILE* file;
    file = fopen("A.bin", "rb");
    fread(A, sizeof(double), N * N, file);
    fclose(file);
    file = fopen("B.bin", "rb");
    fread(B, sizeof(double), N * N, file);
    fclose(file);
    file = fopen("C.bin", "rb");
    fread(C, sizeof(double), N * N, file);
    fclose(file);

    // compute correct matrix
    for (int k=0; k<N*N; k++) {
        int row = k / N;
        int col = k % N;
        
        double acc = 0;
        for (int i=0; i<N; i++)
            for (int j=0; j<N; j++)
                acc += A[row*N + i] * B[col + j*N];
        
        C[k] = acc;
    }

    // comparison of results
    int error_counter = 0;
    for (int i=0; i<N*N; i++)
        if (fabs(C[i] - C_check[i]) > EPS)
            error_counter++;

    // print result
    printf("errors in matrix-matrix product: %d / %d\n", error_counter, N*N);

    return 0;
}
