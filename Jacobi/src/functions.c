#include <stdio.h>
#include <sys/time.h>
#include "functions.h"


// save matrix state to file (each MPI process it's own grid)
void save_gnuplot_parallel(double *M, size_t dim_y, size_t dim_x, int rank, double y_offset, int n_procs) {
  
    size_t i, j;
    const double h = 0.1;
    FILE *file;

    y_offset *= h;

    if (rank == 0)
        file = fopen("plot/solution.dat", "w");
    else
        file = fopen("plot/solution.dat", "a");

    if (rank == 0)
        for (j=0; j<dim_x+2; ++j)
            fprintf(file, "%f\t%f\t%f\n", h*j, -0.0, M[i * (dim_x+2) + j]);

    for (i=1; i<dim_y+1; ++i)
        for (j=0; j<dim_x+2; ++j)
            fprintf(file, "%f\t%f\t%f\n", h*j, -y_offset-h*i, M[i * (dim_x+2) + j]);

    if (rank == n_procs-1)
        for (j=0; j<dim_x+2; ++j)
            fprintf(file, "%f\t%f\t%f\n", h*j, -y_offset-h*(dim_y+1), M[(dim_y+1) * (dim_x+2) + j]);

    fclose(file);
}

// save matrix state to file (serial case)
void save_gnuplot(double *M, size_t mat_size) {

    size_t i, j;
    const double h = 0.1;
    FILE *file = fopen("plot/solution.dat", "w");

    for (i=0; i<mat_size+2; i++) 
        for (j=0; j<mat_size+2; j++)
            fprintf(file, "%f\t%f\t%f\n", h*j, -h*i, M[i * (mat_size+2) + j]);

    fclose(file);
}

// return the elapsed time
// a Simple timer for measuring the walltime
double seconds() {

    struct timeval tmp;
    double sec;

    gettimeofday(&tmp, (struct timezone *)0);
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    
    return sec;
}
