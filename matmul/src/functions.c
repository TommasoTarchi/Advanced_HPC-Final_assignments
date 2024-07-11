#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "functions.h"
#ifdef OPENMP
#include <omp.h>
#endif


// randomly initializes elements of a matrix makes sure that each thread
// on each MPi process has a different seed
//
// returns the number of threads per process (needed if the function is
// used more than once within the same code, to not repeat seeds)
//
// NOTICE: in general we could have made code simpler by getting
// n_treads outside this function in the main, but we couldn't, since
// openMP seems to be not compatible with CUDA
int random_mat(double* mat, int mat_size, unsigned int seed, int rank) {

    // set factor to obtain elements with at most an order of 
    // magnitude ~10^6 (to avoid overflow)
    double exp = (6. - log10((double) mat_size)) / 2.;
    double factor = pow(10., exp);

    int n_threads = 1;
    int my_thread_id = 0;
   #pragma omp parallel
    {

#ifdef OPENMP
        // get number of threads on process
        n_threads = omp_get_num_threads();

        // get each thread's id (=0 if openMP is not enabled)
        my_thread_id = omp_get_thread_num();
#endif
        
        // setting a different seed for each thread
        struct drand48_data rand_gen;
        srand48_r(seed+(unsigned int)rank*(unsigned int)n_threads+(unsigned int)my_thread_id, &rand_gen);

       #pragma omp for
        for (int i=0; i<mat_size; i++) {
            
            // producing a uniformly distributed number between 0 and 1
            double random_number;
            drand48_r(&rand_gen, &random_number);

            // assigning random value to element in matrix
            mat[i] = random_number * factor;
	    }
    }

    return n_threads;
}

// creates a block (submatrix) at a given location of a larger matrix
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

// save computed profiling times to file (done by master process)
void save_time(double* times, char* csv_name, int n_procs) {

    // compute average times
    double* avg_times;
    avg_times = (double*) malloc(3 * sizeof(double));
    avg_times[0] = 0;
    avg_times[1] = 0;
    avg_times[2] = 0;
    for (int count=0; count<n_procs; count++) {
        avg_times[0] += times[3 * count] / (double) n_procs;
        avg_times[1] += times[1 + 3 * count] / (double) n_procs;
        avg_times[2] += times[2 + 3 * count] / (double) n_procs;
    }

    // print times
    char file_name[50];  // assume file_name no longer than 50 chars
    sprintf(file_name, "%s", csv_name);
    FILE* file = fopen(file_name, "a");
    fprintf(file, "%f,%f,%f\n", avg_times[0], avg_times[1], avg_times[2]);
    fclose(file);

    free(avg_times);
}
