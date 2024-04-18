#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#include "functions.h"


int main(int argc, char* argv[]){

    int my_rank, n_procs;

    // init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Status status;

    // indexes for loops
    size_t i, j, it;

    // initialize matrix
    double *matrix, *matrix_new, *tmp_matrix;

    // define needed variables
    size_t N = 0, iterations = 0, row_peek = 0, col_peek = 0;
    size_t byte_dimension = 0;

    // get input parameters
    if (argc != 5) {
        
        if (my_rank == 0)
            fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it n m\n");
        
        MPI_Barrier();
        return 1;
    }
    N = atoi(argv[1]);
    iterations = atoi(argv[2]);
    row_peek = atoi(argv[3]);
    col_peek = atoi(argv[4]);

    // compute local matrix dimensions
    size_t N_loc = N / n_procs;
    size_t N_rest = N % n_procs;
    size_t offset = my_rank * N_loc * N;  // position of first element of local matrix in global matrix
    if (my_rank < N_rest) {
        N_loc++;
        offset += my_rank * N;
    } else {
        offset += N_rest * N;
    }

    // print parameters
    if (my_rank == 0) {
        printf("matrix size = %zu\n", N);
        printf("number of iterations = %zu\n", iterations);
        printf("element for checking = Mat[%zu,%zu]\n",row_peek, col_peek);
    }

    // check for invalid peak indexes
    if ((row_peek > N) || (col_peek > N)) {

        if (my_rank == 0) {
            fprintf(stderr, "Cannot Peek a matrix element outside of the matrix dimension\n");
            fprintf(stderr, "Arguments n and m must be smaller than %zu\n", N);
        }

        MPI_Barrier();
        return 1;
    }

    // allocate local matrices
    byte_dimension = sizeof(double) * (N_loc + 2) * (N + 2);
    matrix = (double*) malloc(byte_dimension);
    matrix_new = (double*) malloc(byte_dimension);

    // CAPIRE A CHE SERVONO STE DUE RIGHE (E SE VANNO CAMBIATE CON L'AGGIUNTA DI MPI)
    memset(matrix, 0, byte_dimension);
    memset(matrix_new, 0, byte_dimension);

    // fill initial values (EVENTUALMENTE PARALLELIZZARE CON OPENMP O OPENACC)
    for (i=1; i<=N_loc; ++i)
        for (j=1; j<=N; ++j)
            matrix[(i * (N + 2)) + j] = 0.5;

    // set up borders (EVENTUALMENTE PARALLELIZZARE CON OPENMP O OPENACC)
    double increment = 100.0 / (N + 1);
    double increment_start = increment * (offset / N);
    for (i=1; i<=N_loc+1; ++i) {
        matrix[i * (N + 2)] = increment_start + i * increment;
        matrix_new[i * (N + 2)] = increment_start + i * increment;
    }
    if (my_rank == n_procs-1) {
        for (i=1; i<=N+1; i++) {
            matrix[((N_loc + 1) * (N + 2)) + (N + 1 - i)] = i * increment;
            matrix_new[((N_loc + 1) * (N + 2)) + (N + 1 - i)] = i * increment;
        }
    }

    // define variables for send-receive (using MPI_PROC_NULL 
    // as dummy destination-source)
    int destsource_up = my_rank - 1;
    int destsource_down = my_rank + 1;
    if (my_rank == 0)
        destsource_up = MPI_PROC_NULL;
    if (my_rank == n_procs-1)
        destsource_down = MPI_PROC_NULL;

    // start algorithm
    for (it=0; it<iterations; ++it) {

        // send and receive bordering data (backward first and 
        // forward then)
        MPI_Sendrecv(matrix, N+2, MPI_DOUBLE, destsource_up, destsource_up, &matrix[(N_loc+1)*(N+2)], N+2, MPI_DOUBLE, destsource_down, destsource_down, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&matrix[(N_loc+1)*(N+2)], N+2, MPI_DOUBLE, destsource_down, destsource_down, matrix, N+2, MPI_DOUBLE, destsource_up, destsource_up, MPI_COMM_WORLD, &status);

        // evolve state of cells
        evolve(matrix, matrix_new, N_loc, N);

        // swap the pointers
        tmp_matrix = matrix;
        matrix = matrix_new;
        matrix_new = tmp_matrix;
    }

    // print element for checking
    if ((offset / N < row_peek) && (row_peek < offset / N + N_loc)) {
        size_t true_row_peek = row_peek % (offset / N);
        printf("\nmatrix[%zu,%zu] = %f\n", row_peek, col_peek, matrix[(true_row_peek + 1) * (N + 2) + (col_peek + 1)]);
    }

    save_gnuplot(matrix, N);

    free(matrix);
    free(matrix_new);

    MPI_Finalize();

    return 0;
}
