/*
 * performs Jacobi evolution of a grid of cells in parallel
 *
 * compile with -DTIME for profiling: times for matrix 
 * initialization, communications and computations will be 
 * printed to CSV file called times_MPI-RMA.csv in profiling/
 * folder
 *
 * compile with -fopenmp -DOPENMP for further parallelization
 * of grid initialization and evolution using openMP
 *
 * base serial code is taken from prof. Ivan Girotto at ICTP
 *
 * MPI communications are performed using the Remote Memory
 * Access paradigm
 *
 * */


#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#include "functions.h"
#ifdef OPENMP
#include <omp.h>
#endif


int main(int argc, char* argv[]){

    int my_rank, n_procs;

    // init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
    MPI_Status status;

    // init MPI window
    MPI_Win win_up, win_down;
    MPI_Request request;

    // variables for timing
    //
    // (for clarity, we use t1, t2 to time initialization, t3, t4
    // to time communications, t5, t6 to time actual computation
    // and t7, t8, t9, t10 to time host-device communications)
#ifdef TIME
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;
    double t_comm = 0, t_comp = 0, t_host_dev_once = 0, t_host_dev_iter = 0;
#endif

    // indexes for loops
    size_t i, j, it;

    // initialize matrix and boundaries
    double *matrix, *matrix_new, *boundary_up, *boundary_down;

    // define needed variables
    size_t N = 0, iterations = 0, row_peek = 0, col_peek = 0;
    size_t byte_dimension = 0;

    // get input parameters
    if (argc != 5) {
        
        if (my_rank == 0)
            fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it n m\n");
        
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 1;
    }
    N = atoi(argv[1]);
    iterations = atoi(argv[2]);
    row_peek = (size_t) atoi(argv[3]);
    col_peek = (size_t) atoi(argv[4]);

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

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        return 1;
    }

#ifdef TIME
    t1 = MPI_Wtime();
#endif

    // allocate local matrices
    byte_dimension = sizeof(double) * N_loc * (N + 2);
    matrix = (double*) malloc(byte_dimension);
    matrix_new = (double*) malloc(byte_dimension);
    memset(matrix, 0, byte_dimension);
    memset(matrix_new, 0, byte_dimension);

    // allocate boundaries ('boundaries' is allocated and assigned to a window)
    MPI_Win_allocate((N + 2) * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &boundary_up, &win_up);
    MPI_Win_allocate((N + 2) * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &boundary_down, &win_down);
    memset(boundary_up, 0, (N + 2) * sizeof(double));
    memset(boundary_down, 0, (N + 2) * sizeof(double));

    // fill initial values
   #pragma omp parallel for collapse(2)
    for (i=0; i<N_loc; ++i)
        for (j=1; j<=N; ++j) {
            matrix[(i * (N + 2)) + j] = 0.5;
        }

    // set up borders
    double increment = 100.0 / (N + 1);
    double increment_start = increment * (offset / N);
    for (i=0; i<N_loc; ++i) {
        
	    matrix[i * (N + 2)] = increment_start + (i + 1) * increment;
        matrix_new[i * (N + 2)] = increment_start + (i + 1) * increment;
    }
    if (my_rank == n_procs-1) {
        
        for (i=1; i<=N+1; i++) {
            boundary_down[N + 1 - i] = i * increment;
        }
    }

#ifdef TIME
    t2 = MPI_Wtime();
#endif

    // define destinations
    int dest_up = my_rank - 1;
    int dest_down = my_rank + 1;

    // create groups for RMA communications
    //
    // (groups only contain processes exposing/accessing the windows)
    MPI_Group world_group, group_up, group_down;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    if (my_rank > 0)
        MPI_Group_incl(world_group, 1, &dest_up, &group_up);
    if (my_rank < n_procs-1)
        MPI_Group_incl(world_group, 1, &dest_down, &group_down);

    // start algorithm
    for (it=0; it<iterations; ++it) {

#ifdef TIME
        t3 = MPI_Wtime();
#endif

        // post exposure epoch
        if (my_rank > 0)
            MPI_Win_post(group_up, 0, win_up);
        if (my_rank < n_procs-1)
            MPI_Win_post(group_down, 0, win_down);

        if (my_rank > 0) {
            MPI_Win_start(group_up, 0, win_down);
            MPI_Put(matrix, N + 2, MPI_DOUBLE, dest_up, 0, N + 2, MPI_DOUBLE, win_down);
            MPI_Win_complete(win_down);
        }

        if (my_rank < n_procs-1) {
            MPI_Win_start(group_down, 0, win_up);
            MPI_Put(&matrix[(N_loc - 1) * (N + 2)], N + 2, MPI_DOUBLE, dest_down, 0, N + 2, MPI_DOUBLE, win_up);
            MPI_Win_complete(win_up);
        }

#ifdef TIME
        t4 = MPI_Wtime();
        t_comm += t4 - t3;
#endif

#ifdef TIME
        t5 = MPI_Wtime();
#endif

        // update interal cells
       #pragma omp parallel for collapse(2)
        for (i=1; i<N_loc-1; ++i)
            for (j=1; j<N+1; ++j)
                matrix_new[ ( i * ( N + 2 ) ) + j ] = ( 0.25 ) * 
                ( matrix[ ( ( i - 1 ) * ( N + 2 ) ) + j ] + 
                  matrix[ ( i * ( N + 2 ) ) + ( j + 1 ) ] + 	  
                  matrix[ ( ( i + 1 ) * ( N + 2 ) ) + j ] + 
                  matrix[ ( i * ( N + 2 ) ) + ( j - 1 ) ] ); 

#ifdef TIME
        t6 = MPI_Wtime();
        t_comp += t6 - t5;
#endif

#ifdef TIME
        t3 = MPI_Wtime();
#endif
    
        // wait for RMA operations to complete
        if (my_rank > 0)
            MPI_Win_wait(win_up);
        if (my_rank < n_procs-1)
            MPI_Win_wait(win_down);

#ifdef TIME
        t4 = MPI_Wtime();
        t_comm += t4 - t3;
#endif

#ifdef TIME
        t5 = MPI_Wtime();
#endif

        // update first row
       #pragma omp parallel for
        for (j=1; j<N+1; j++)
            matrix_new[ j ] = ( 0.25 ) *
            ( boundary_up[ j ] +
              matrix[ j - 1 ] +
              matrix[ j + 1 ] +
              matrix[ N + 2 + j ] );

        // update last row
       #pragma omp parallel for
        for (j=1; j<N+1; j++)
            matrix_new[ ( N_loc - 1 ) * ( N + 2) + j] = ( 0.25 ) *
            ( matrix[ ( N_loc - 2 ) * ( N + 2 ) + j ] +
              matrix[ ( N_loc - 1 ) * ( N + 2) + ( j - 1 ) ] +
              matrix[ ( N_loc - 1 ) * ( N + 2) + ( j + 1 ) ] +
              boundary_down[ j ] );

        // switch pointers to matrices
        double *tmp, *tmp_bound_up, *tmp_bound_down;
        tmp = matrix;
        matrix = matrix_new;
        matrix_new = tmp;

#ifdef TIME
        t6 = MPI_Wtime();
        t_comp += t6 - t5;
#endif

    }

    // print element for checking
    if (((offset / N) <= row_peek) && (row_peek < (offset / N + N_loc))) {
	
        size_t true_row_peek = row_peek;
        if (my_rank)
            true_row_peek = row_peek % (offset / N);
        
        printf("\nmatrix[%zu,%zu] = %f\n", row_peek, col_peek, matrix[true_row_peek * (N + 2) + col_peek + 1]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    // save data for plot (process 0 gathers data and prints them to file)
    // 
    // communications are not in RMA here (wouldn't be a big gain in efficiency)
    if (my_rank == 0) {

        // print process 0's data
        save_gnuplot_parallel_no_bounds(matrix, boundary_up, boundary_down, N_loc, N, my_rank, offset/N, n_procs);
        
        size_t col_offset_recv = N_loc;
        size_t N_loc_recv;
        for (int count=1; count<n_procs-1; count++) {

            N_loc_recv = N / n_procs + (count < N_rest);

            // receive central processes' data and print them
            MPI_Recv(matrix, N_loc_recv * (N+2), MPI_DOUBLE, count, count, MPI_COMM_WORLD, &status);
            save_gnuplot_parallel_no_bounds(matrix, boundary_up, boundary_down, N_loc_recv, N, count, col_offset_recv, n_procs);

            col_offset_recv += N_loc_recv;
        }
        
        N_loc_recv = N / n_procs + (n_procs-1 < N_rest);

        // receive last process's data and print them
        if (n_procs > 1) {
            MPI_Recv(matrix, N_loc_recv * (N+2), MPI_DOUBLE, n_procs-1, n_procs-1, MPI_COMM_WORLD, &status);
            MPI_Recv(boundary_down, N + 2, MPI_DOUBLE, n_procs-1, n_procs-1, MPI_COMM_WORLD, &status);
            save_gnuplot_parallel_no_bounds(matrix, boundary_up, boundary_down, N_loc_recv, N, n_procs-1, col_offset_recv, n_procs);
        }
    
    } else if (my_rank < n_procs-1) {

        // send data from central processes
        MPI_Send(matrix, N_loc * (N+2), MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD);

    } else {

        // send data from last process
        MPI_Send(matrix, N_loc * (N+2), MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD);
        MPI_Send(boundary_down, N + 2, MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD);
    }

    free(matrix);
    free(matrix_new);

    MPI_Win_free(&win_up);
    MPI_Win_free(&win_down);

    if (my_rank > 0)
        MPI_Group_free(&group_up);
    if (my_rank < n_procs-1)
        MPI_Group_free(&group_down);
    MPI_Group_free(&world_group);

    // gather measured times and print them
#ifdef TIME
    double* times;

    if (my_rank == 0)
        times = (double*) malloc(n_procs * 5 * sizeof(double));    
    else
        times = (double*) malloc(5 * sizeof(double));

    times[0] = t2 - t1;  // time for initialization
    times[1] = t_comm / (double) iterations;  // time for communications (average)
    times[2] = t_comp / (double) iterations;  // time for computation (average)
    times[3] = t_host_dev_once;  // time for initial and final host-device communications
    times[4] = t_host_dev_iter;  // time for iterated host-device communications

    MPI_Gather(times, 5, MPI_DOUBLE, times, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // save times
    if (my_rank == 0) {
        char csv_name[] = "profiling/times_MPI-RMA.csv";
        save_time(times, csv_name, n_procs);
    }

    free(times);
#endif

    MPI_Finalize();

    return 0;
}
