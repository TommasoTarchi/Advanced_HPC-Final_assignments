/*
 * performs Jacobi evolution of a grid of cells in parallel
 *
 * compile with -DOPENACC for GPU acceleration with OpenACC
 *
 * compile with -DTIME for profiling: times for matrix 
 * initialization, communications and computations will be 
 * printed to CSV file called times_aware.csv in profiling/
 * folder
 *
 * compile with -fopenmp -DOPENMP for further parallelization 
 * of grid initialization using openMP
 *
 * base serial code is taken from prof. Ivan Girotto at ICTP
 *
 * MPI communications are CUDA-aware
 *
 * */


#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>
#include <openacc.h>
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

    // setting devices
#ifdef OPENACC
    int n_gpus = acc_get_num_devices(acc_device_nvidia);
    int i_gpu = my_rank % n_gpus;
    
    acc_set_device_num(i_gpu, acc_device_nvidia);
    acc_init(acc_device_nvidia);
#endif

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

    // initialize matrix
    double *matrix, *matrix_new;

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
    byte_dimension = sizeof(double) * (N_loc + 2) * (N + 2);
    matrix = (double*) malloc(byte_dimension);
    matrix_new = (double*) malloc(byte_dimension);
    memset(matrix, 0, byte_dimension);
    memset(matrix_new, 0, byte_dimension);

    // fill initial values
   #pragma omp parallel for collapse(2)
    for (i=1; i<=N_loc; ++i)
        for (j=1; j<=N; ++j) {
            matrix[(i * (N + 2)) + j] = 0.5;
            matrix_new[(i * (N + 2)) + j] = 0.5;
        }

    // set up borders
    double increment = 100.0 / (N + 1);
    double increment_start = increment * (offset / N);
   #pragma omp parallel for
    for (i=1; i<=N_loc+1; ++i) {
        
	    matrix[i * (N + 2)] = increment_start + i * increment;
        matrix_new[i * (N + 2)] = increment_start + i * increment;
    }
    if (my_rank == n_procs-1) {
        
       #pragma omp parallel for
        for (i=1; i<=N+1; i++) {
            matrix[((N_loc + 1) * (N + 2)) + (N + 1 - i)] = i * increment;
            matrix_new[((N_loc + 1) * (N + 2)) + (N + 1 - i)] = i * increment;
        }
    }

#ifdef TIME
    t2 = MPI_Wtime();
#endif

    // define variables for send-receive (using MPI_PROC_NULL 
    // as dummy destination-source)
    int destsource_up = my_rank - 1;
    int tag_up = my_rank - 1;
    int destsource_down = my_rank + 1;
    int tag_down = my_rank + 1;
    if (my_rank == 0) {
        destsource_up = MPI_PROC_NULL;
	    tag_up = -1;  // arbitrary
    }
    if (my_rank == n_procs-1) {
        destsource_down = MPI_PROC_NULL;
	    tag_down = -1;  // arbitrary
    }

#ifdef TIME
    t7 = MPI_Wtime();
#endif
    
    // copy matrix to device
    //
    // NOTICE: we cannot use copyin() for matrix_new beacuse if we did the algorithm
    // would not work for odd values of 'iterations', because of pointer swap
    size_t total_length = (N_loc+2) * (N+2);
   #pragma acc data copy(matrix[0:total_length], matrix_new[0:total_length])
    {

#ifdef TIME
        t8 = MPI_Wtime();
        t_host_dev_once += t8 - t7;
#endif

        // start algorithm
        for (it=0; it<iterations; ++it) {

#ifdef TIME
            t3 = MPI_Wtime();
#endif
    
            // send and receive bordering data (backward first and 
            // forward then) from device to device directly
            //
            // (the host_data clause specifies that, when executing
            // the code, the host has to use device pointers, and
            // the use_device clause specifies which pointers should
            // be "exposed" to host)
           #pragma acc host_data use_device(matrix)
            {
                MPI_Sendrecv(&matrix[N+2], N+2, MPI_DOUBLE, destsource_up, my_rank, &matrix[(N_loc+1)*(N+2)], N+2, MPI_DOUBLE, destsource_down, tag_down, MPI_COMM_WORLD, &status);        
                MPI_Sendrecv(&matrix[N_loc*(N+2)], N+2, MPI_DOUBLE, destsource_down, my_rank, matrix, N+2, MPI_DOUBLE, destsource_up, tag_up, MPI_COMM_WORLD, &status);
            }

#ifdef TIME
            t4 = MPI_Wtime();
            t_comm += t4 - t3;
#endif

#ifdef TIME
            t5 = MPI_Wtime();
#endif

	    
            // update system's state on device or host
#ifdef OPENACC
           #pragma acc parallel loop gang present(matrix[0:total_length], matrix_new[0:total_length]) collapse(2)
#else
           #pragma omp parallel for collapse(2)
#endif
            for (i=1; i<N_loc+1; ++i)
                for (j=1; j<N+1; ++j)
                    matrix_new[ ( i * ( N + 2 ) ) + j ] = ( 0.25 ) * 
                    ( matrix[ ( ( i - 1 ) * ( N + 2 ) ) + j ] + 
                      matrix[ ( i * ( N + 2 ) ) + ( j + 1 ) ] + 	  
                      matrix[ ( ( i + 1 ) * ( N + 2 ) ) + j ] + 
                      matrix[ ( i * ( N + 2 ) ) + ( j - 1 ) ] ); 

            // switch pointers to update main grid
#ifdef OPENACC
           #pragma acc serial present(matrix[0:total_length], matrix_new[0:total_length]) 
            {
                double* tmp;
                tmp = matrix;
                matrix = matrix_new;
                matrix_new = tmp;
            }
#endif
            double* tmp;
            tmp = matrix;
            matrix = matrix_new;
            matrix_new = tmp;

#ifdef TIME
            t6 = MPI_Wtime();
            t_comp += t6 - t5;
#endif

        }

#ifdef TIME
        t7 = MPI_Wtime();
#endif

    }

#ifdef TIME
    t8 = MPI_Wtime();
    t_host_dev_once += t8 - t7;
#endif
    
    // print element for checking
    if (((offset / N) <= row_peek) && (row_peek < (offset / N + N_loc))) {
	
        size_t true_row_peek = row_peek;
        if (my_rank)
            true_row_peek = row_peek % (offset / N);
        
        printf("\nmatrix[%zu,%zu] = %f\n", row_peek, col_peek, matrix[(true_row_peek + 1) * (N + 2) + (col_peek + 1)]);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    // save data for plot (process 0 gathers data and prints them to file)
    if (my_rank == 0) {

        // print process 0's data
        save_gnuplot_parallel(matrix, N_loc, N, my_rank, offset/N, n_procs);
        
        size_t col_offset_recv = N_loc;
        for (int count=1; count<n_procs; count++) {

            size_t N_loc_recv = N / n_procs + (count < N_rest);

            // receive data from other processes and print them
            MPI_Recv(matrix, (N_loc_recv+2) * (N+2), MPI_DOUBLE, count, count, MPI_COMM_WORLD, &status);
            save_gnuplot_parallel(matrix, N_loc_recv, N, count, col_offset_recv, n_procs);

            col_offset_recv += N_loc_recv;
        }
    
    } else {

        // send data from other processes
        MPI_Send(matrix, (N_loc+2) * (N+2), MPI_DOUBLE, 0, my_rank, MPI_COMM_WORLD);
    }

    free(matrix);
    free(matrix_new);

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
    times[4] = t_host_dev_iter / (double) iterations;  // time for iterated host-device communications

    MPI_Gather(times, 5, MPI_DOUBLE, times, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // save times
    if (my_rank == 0) {
        char csv_name[] = "profiling/times_aware.csv";
        save_time(times, csv_name, n_procs);
    }

    free(times);
#endif

    MPI_Finalize();

    return 0;
}
