void save_gnuplot_parallel(double *M, size_t dim_y, size_t dim_x, int rank, double y_offset, int n_procs);
void save_gnuplot(double *M, size_t mat_size);
void save_gnuplot_parallel_no_bounds(double *M, double *bound_up, double *bound_down, size_t dim_y, size_t dim_x, int rank, double y_offset, int n_procs);
double seconds(void);
void save_time(double* times, char* csv_name, int n_procs);
