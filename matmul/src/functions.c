#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "functions.h"


// randomly initializes elements of a matrix
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
    FILE* file = fopen(csv_name, "a");
    fprintf(file, "%f,%f,%f\n", avg_times[0], avg_times[1], avg_times[2]);
    fclose(file);

    free(avg_times);
}
