#include <stdio.h>
#include <stdlib.h>
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
            fprintf(file, "%f\t%f\t%f\n", h*j, -0.0, M[j]);

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

// save matrix state to file (each MPI process it's own grid) in case boundaries
// are seprated from proper cells
void save_gnuplot_parallel_no_bounds(double *M, double *bound_up, double *bound_down, size_t dim_y, size_t dim_x, int rank, double y_offset, int n_procs) {
  
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
            fprintf(file, "%f\t%f\t%f\n", h*j, -0.0, bound_up[j]);

    for (i=0; i<dim_y; ++i)
        for (j=0; j<dim_x+2; ++j)
            fprintf(file, "%f\t%f\t%f\n", h*j, -y_offset-h*(i+1), M[i * (dim_x+2) + j]);

    if (rank == n_procs-1)
        for (j=0; j<dim_x+2; ++j)
            fprintf(file, "%f\t%f\t%f\n", h*j, -y_offset-h*(dim_y+1), bound_down[j]);

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

// save computed profiling times to file (done by master process)
void save_time(double* times, char* csv_name, int n_procs) {

    // compute average times
    double* avg_times;
    avg_times = (double*) malloc(5 * sizeof(double));
    avg_times[0] = 0;
    avg_times[1] = 0;
    avg_times[2] = 0;
    avg_times[3] = 0;
    avg_times[4] = 0;
    for (int count=0; count<n_procs; count++) {
        avg_times[0] += times[5 * count] / (double) n_procs;
        avg_times[1] += times[1 + 5 * count] / (double) n_procs;
        avg_times[2] += times[2 + 5 * count] / (double) n_procs;
        avg_times[3] += times[3 + 5 * count] / (double) n_procs;
        avg_times[4] += times[4 + 5 * count] / (double) n_procs;
    }

    // print times
    char file_name[50];  // assume file_name no longer than 50 chars
    sprintf(file_name, "%s", csv_name);
    FILE* file = fopen(file_name, "a");
    fprintf(file, "%f,%f,%f,%f,%f\n", avg_times[0], avg_times[1], avg_times[2], avg_times[3], avg_times[4]);
    fclose(file);

    free(avg_times);
}
