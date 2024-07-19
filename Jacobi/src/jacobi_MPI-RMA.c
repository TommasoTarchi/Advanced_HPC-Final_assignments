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
    MPI_Win win;
    MPI_Request request;

    // variables for timing
    //
    // (for clarity, we use t1 and t2 to time initialization,
    // t3 and t4 to time communications, and t5 and t6 to time
    // actual computation)
#ifdef TIME
    double t1, t2, t3, t4, t_comm = 0, t5, t6, t_comp = 0;
#endif

    // indexes for loops
    size_t i, j, it;

    // initialize matrix and boundaries
    double *matrix, *matrix_new, *boundaries, *boundaries_new;

    // define needed variables
    size_t N = 0, iterations = 0, row_peek = 0, col_peek = 0;
    size_t byte_dimension = 0;

    // get input parameters
    if (argc != 5) {
        
        if (my_rank == 0)
            fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it n m\n");
        
        MPI_Barrier(MPI_COMM_WORLD);
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
        return 1;
    }

    // allocate local matrices
    byte_dimension = sizeof(double) * N_loc * (N + 2);
    matrix = (double*) malloc(byte_dimension);
    matrix_new = (double*) malloc(byte_dimension);
    memset(matrix, 0, byte_dimension);
    memset(matrix_new, 0, byte_dimension);

    // allocate boundaries ('boundaries' is allocated and assigned to a window)
    MPI_Win_allocate(2 * (N + 2) * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &boundaries, &win);
    boundaries_new = (double*) malloc(2 * (N + 2) * sizeof(double));
    memset(boundaries, 0, 2 * (N + 2) * sizeof(double));
    memset(boundaries_new, 0, 2 * (N + 2) * sizeof(double));

#ifdef TIME
    t1 = MPI_Wtime();
#endif

    // fill initial values
   #pragma omp parallel for collapse(2)
    for (i=0; i<N_loc; ++i)
        for (j=1; j<=N; ++j) {
            matrix[(i * (N + 2)) + j] = 0.5;
            matrix_new[(i * (N + 2)) + j] = 0.5;
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
            boundaries[N + 2 + (N + 1 - i)] = i * increment;
            boundaries_new[N + 2 + (N + 1 - i)] = i * increment;
        }
    }

#ifdef TIME
    t2 = MPI_Wtime();
#endif

    // define destinations
    int dest_up = my_rank - 1;
    int dest_down = my_rank + 1;

    // create group for RMA communications
    MPI_Group world_group, neighbor_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    int *neighbors;
    if (my_rank == 0) {
	neighbors = (int*) malloc(2 * sizeof(int));
	neighbors[0] = my_rank;
	neighbors[1] = dest_down;
	MPI_Group_incl(world_group, 2, neighbors, &neighbor_group);
    } else if (my_rank == n_procs-1) {
	neighbors = (int*) malloc(2 * sizeof(int));
	neighbors[0] = dest_up;
	neighbors[1] = my_rank;
	MPI_Group_incl(world_group, 2, neighbors, &neighbor_group);
    } else {
	neighbors = (int*) malloc(3 * sizeof(int));
	neighbors[0] = dest_up;
	neighbors[1] = my_rank;
	neighbors[2] = dest_down;
	MPI_Group_incl(world_group, 3, neighbors, &neighbor_group);
    }

	///////////////////////////////////////////////////////////////////
	printf("Ready to start algorithm from rank %d\n", my_rank);
	///////////////////////////////////////////////////////////////////
    
    // start algorithm
    for (it=0; it<iterations; ++it) {

#ifdef TIME
        t3 = MPI_Wtime();
#endif

        ///////////////////////////////////////////////////////////////////
        int err = 0;
        ///////////////////////////////////////////////////////////////////

        // post exposure epoch
        MPI_Win_post(neighbor_group, 0, win);

        ///////////////////////////////////////////////////////////////////
        printf("Window posted at iteration %d from rank %d\n", it, my_rank);
        ///////////////////////////////////////////////////////////////////

        // start access epoch
        err = MPI_Win_start(neighbor_group, 0, win);
	if (err != MPI_SUCCESS) {
	    char error_string[BUFSIZ];
	    int length_of_error_string;
	    MPI_Error_string(err, error_string, &length_of_error_string);
	    fprintf(stderr, "Error in MPI_Win_start at iteration %zu from rank %d: %s\n", it, my_rank, error_string);
	    MPI_Abort(MPI_COMM_WORLD, err);
	}

        ///////////////////////////////////////////////////////////////////
        printf("Window started at iteration %d from rank %d\n", it, my_rank);
        ///////////////////////////////////////////////////////////////////

        // communicate boundaries
        if (my_rank > 0)
            err = MPI_Rput(matrix, N + 2, MPI_DOUBLE, dest_up, N + 2, N + 2, MPI_DOUBLE, win, &request);
		if (err != MPI_SUCCESS) {
		    char error_string[BUFSIZ];
		    int length_of_error_string;
		    MPI_Error_string(err, error_string, &length_of_error_string);
		    fprintf(stderr, "Error in MPI_Rput to upper neighbor at iteration %zu from rank %d: %s\n", it, my_rank, error_string);
		    MPI_Abort(MPI_COMM_WORLD, err);
		}
        if (my_rank < n_procs - 1)
            err = MPI_Rput(&matrix[(N_loc - 1) * (N + 2)], N + 2, MPI_DOUBLE, dest_down, 0, N + 2, MPI_DOUBLE, win, &request);
		if (err != MPI_SUCCESS) {
		    char error_string[BUFSIZ];
		    int length_of_error_string;
		    MPI_Error_string(err, error_string, &length_of_error_string);
		    fprintf(stderr, "Error in MPI_Rput to lower neighbor at iteration %zu from rank %d: %s\n", it, my_rank, error_string);
		    MPI_Abort(MPI_COMM_WORLD, err);
		}

        // complete access epoch
        MPI_Win_complete(win);

        // wait for RMA operations to complete
        MPI_Win_wait(win);

        ///////////////////////////////////////////////////////////////////
        printf("Communications at iteration %d completed from rank %d\n", it, my_rank);
        ///////////////////////////////////////////////////////////////////

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
            ( boundaries[ j ] +
              matrix[ j - 1 ] +
              matrix[ j + 1 ] +
              matrix[ N + 2 + j ] );

        // update interal cells
       #pragma omp parallel for collapse(2)
        for (i=1; i<N_loc-1; ++i)
            for (j=1; j<N+1; ++j)
                matrix_new[ ( i * ( N + 2 ) ) + j ] = ( 0.25 ) * 
                ( matrix[ ( ( i - 1 ) * ( N + 2 ) ) + j ] + 
                  matrix[ ( i * ( N + 2 ) ) + ( j + 1 ) ] + 	  
                  matrix[ ( ( i + 1 ) * ( N + 2 ) ) + j ] + 
                  matrix[ ( i * ( N + 2 ) ) + ( j - 1 ) ] ); 

        // update last row
       #pragma omp parallel for
        for (j=1; j<N+1; j++)
            matrix_new[ ( N_loc - 1 ) * ( N + 2) + j] = ( 0.25 ) *
            ( matrix[ ( N_loc - 2 ) * ( N + 2 ) + j ] +
              matrix[ ( N_loc - 1 ) * ( N + 2) + ( j - 1 ) ] +
              matrix[ ( N_loc - 1 ) * ( N + 2) + ( j + 1 ) ] +
              boundaries[ ( N + 2 ) * j ] );

        // switch pointers to matrices
        double *tmp, *tmp_bound;
        tmp = matrix;
        matrix = matrix_new;
        matrix_new = tmp;
        tmp_bound = boundaries;
        boundaries = boundaries_new;
        boundaries_new = tmp_bound;

	///////////////////////////////////////////////////////////////////
	printf("Iteration %d completed\n", it);
	///////////////////////////////////////////////////////////////////

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
        save_gnuplot_parallel_no_bounds(matrix, boundaries, N_loc, N, my_rank, offset/N, n_procs);
        
        size_t col_offset_recv = N_loc;
        size_t N_loc_recv;
        for (int count=1; count<n_procs-1; count++) {

            N_loc_recv = N / n_procs + (count < N_rest);

            // receive central processes' data and print them
            MPI_Recv(matrix, N_loc_recv * (N+2), MPI_DOUBLE, count, count, MPI_COMM_WORLD, &status);
            save_gnuplot_parallel_no_bounds(matrix, boundaries, N_loc_recv, N, count, col_offset_recv, n_procs);

            col_offset_recv += N_loc_recv;
        }
        
        N_loc_recv = N / n_procs + (n_procs-1 < N_rest);

        // receive last process's data and print them
        MPI_Recv(matrix, N_loc_recv * (N+2), MPI_DOUBLE, n_procs-1, n_procs-1, MPI_COMM_WORLD, &status);
        MPI_Recv(&boundaries[N+2], N + 2, MPI_DOUBLE, n_procs-1, n_procs-1, MPI_COMM_WORLD, &status);
        save_gnuplot_parallel_no_bounds(matrix, boundaries, N_loc_recv, N, n_procs-1, col_offset_recv, n_procs);
    
    } else if (my_rank < n_procs-1) {

        // send data from central processes
        MPI_Send(matrix, N_loc * (N+2), MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD);

    } else {

        // send data from last process
        MPI_Send(matrix, N_loc * (N+2), MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD);
        MPI_Send(&boundaries[N+2], N + 2, MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD);
    }

    free(matrix);
    free(matrix_new);

    MPI_Win_free(&win);
    free(boundaries_new);

    MPI_Group_free(&neighbor_group);
    MPI_Group_free(&world_group);
    free(neighbors);

    // gather measured times and print them
#ifdef TIME
    double* times;

    if (my_rank == 0)
        times = (double*) malloc(n_procs * 3 * sizeof(double));    
    else
        times = (double*) malloc(3 * sizeof(double));

    times[0] = t2 - t1;  // time for initialization
    times[1] = t_comm;  // time for communications
    times[2] = t_comp;  // time for computation

    MPI_Gather(times, 3, MPI_DOUBLE, times, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        // average communication and computation times over
        // iterations of the algorithm
        for (i=0; i<n_procs; i++) {
            times[1 + 3 * n_procs] /= (double) iterations;
            times[2 + 3 * n_procs] /= (double) iterations;
        }

        // save times
        char csv_name[] = "profiling/times_MPI-RMA.csv";
        save_time(times, csv_name, n_procs);
    }

    free(times);
#endif

    MPI_Finalize();

    return 0;
}
