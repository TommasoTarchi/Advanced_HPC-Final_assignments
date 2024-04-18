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
