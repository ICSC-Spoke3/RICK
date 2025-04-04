#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
#include <heffte.h>
#include "ricklib.h"
#include <omp.h>

#define BACKEND Heffte_BACKEND_FFTW

struct my_double_complex
{
  double real;
  double imag;
}__attribute__((__packed__));

void fftw_data(
    int grid_size_x,
    int grid_size_y,
    int num_w_planes,
    int num_threads,
    MPI_Comm MYMPI_COMM,
    int size,
    int rank,
   #ifdef WRITE_DATA
    char *fftfile_writedata1,
    char *fftfile_writedata2,
   #endif
    double *grid,
    double *gridss)
{

  
  // FFT transform the data (using distributed FFTW)
  if (rank == 0)
  {
    printf("RICK FFT\n");
  }

  // double start = CPU_TIME_wt;

  int xaxis = grid_size_x;
  int yaxis = grid_size_y / size;

  /* Account for the case in which grid_size_y % size != 0 */
  long remy    = grid_size_y % size;
  long y_start = rank * yaxis + (rank < remy ? rank : remy);
  yaxis        = yaxis + (rank < remy ? 1 : 0);

  long y_end   = y_start + yaxis - 1;
  
  unsigned long long size_of_grid = 2 * num_w_planes * xaxis * yaxis;

  heffte_plan plan;
  struct my_double_complex *input;

  int inbox_low[3]  = {0, y_start, 0};
  int inbox_high[3] = {xaxis - 1, y_end, 0};
  
  
  // double norm = 1.0 / (double)(grid_size_x * grid_size_y);

  // Use the hybrid MPI-OpenMP FFTW

#ifdef HYBRID_FFTW
  fftw_plan_with_nthreads(num_threads);
  if (rank == 0)
    std::cout << "Using " << num_threads << " threads for the FFT" << std::endl;
#endif

  // map the 1D array of complex visibilities to a 2D array required by FFTW (complex[*][2])
  // x is the direction of contiguous data and maps to the second parameter
  // y is the parallelized direction and corresponds to the first parameter (--> n0)
  // and perform the FFT per w plane

  heffte_plan_options fft_options;
  int heffte_err;
  heffte_err = heffte_set_default_options(BACKEND, &fft_options);

  if (heffte_err != Heffte_SUCCESS)  printf("Heffte error in default options %d\n", heffte_err);

  heffte_err = heffte_plan_create(
				  BACKEND,
				  inbox_low, inbox_high,
				  NULL,
				  inbox_low, inbox_high,
				  NULL,
				  MPI_COMM_WORLD,
				  &fft_options,
				  &plan
				  );

  if (heffte_err != Heffte_SUCCESS)  printf("Heffte error in plan creation %d\n", heffte_err);

  unsigned int local_size  = yaxis * xaxis;

  unsigned int inbox_size  = heffte_size_inbox(plan);
    
  input  = (struct my_double_complex*)malloc(inbox_size*sizeof(struct my_double_complex));
  
  for (int iw = 0; iw < num_w_planes; iw++)
  {
    // printf("FFTing plan %d\n",iw);
    //  select the w-plane to transform

#ifdef HYBRID_FFTW
#pragma omp parallel for collapse(2) num_threads(num_threads) 
#endif
    for (unsigned int i = 0; i < local_size; i++)
      {
	input[i].real = grid[2*(i+iw*local_size)];
	input[i].imag = grid[2*(i+iw*local_size)+1];
      }
    

    // do the transform for each w-plane
    heffte_backward_z2z(plan, input, input, Heffte_SCALE_NONE);

    // save the transformed w-plane

#ifdef HYBRID_FFTW
#pragma omp parallel for collapse(2) num_threads(num_threads) 
#endif
    for (unsigned int i = 0; i < local_size; i++)
    {
      gridss[2*(i+iw*local_size)] = input[i].real;
      gridss[2*(i+iw*local_size)+1] = input[i].imag;
    }
   
  }

  heffte_plan_destroy(plan);
  free(input);
      
  if (size > 1)
  {
    MPI_Barrier(MYMPI_COMM);
  }


#ifdef WRITE_DATA

  if (rank == 0)
  {
    printf("WRITING FFT TRANSFORMED DATA\n");
  }
  MPI_File pFilereal;
  MPI_File pFileimg;

  double *gridss_real = (double *)malloc(size_of_grid / 2 * sizeof(double));
  double *gridss_img = (double *)malloc(size_of_grid / 2 * sizeof(double));

  for (unsigned int i = 0; i < size_of_grid / 2; i++)
    {
      gridss_real[i] = gridss[2 * i];
      gridss_img[i]  = gridss[2 * i + 1];
    }

  int ierr;
  ierr = MPI_File_open(MYMPI_COMM, fftfile_writedata1, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &pFilereal);

  if (ierr != MPI_SUCCESS)
    {
      if (rank == 0)
	fprintf(stderr, "Error: Could not open file '%s' for writing.\n", fftfile_writedata1);
    }

  ierr = MPI_File_open(MYMPI_COMM, fftfile_writedata2, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &pFileimg);

  if (ierr != MPI_SUCCESS)
    {
      if (rank == 0)
	fprintf(stderr, "Error: Could not open file '%s' for writing.\n", fftfile_writedata2);
    }

  /* TO BE REDEFINED IN CASE OF NON-TRIVIAL DD */
  int gsizes[3] = {num_w_planes, yaxis*size, xaxis};
  int lsizes[3] = {num_w_planes, yaxis, xaxis};
  int starts[3] = {0, rank*yaxis, 0};

  MPI_Datatype subarray;
  MPI_Type_create_subarray(3, gsizes, lsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray);
  MPI_Type_commit(&subarray);

  MPI_File_set_view(pFilereal, 0, MPI_DOUBLE, subarray, "native", MPI_INFO_NULL);
  MPI_File_set_view(pFileimg, 0, MPI_DOUBLE, subarray, "native", MPI_INFO_NULL);
    
  MPI_File_write_all(pFilereal, gridss_real, size_of_grid / 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
  MPI_File_write_all(pFileimg, gridss_img, size_of_grid / 2, MPI_DOUBLE, MPI_STATUS_IGNORE);

  MPI_Type_free(&subarray);

  MPI_File_close(&pFilereal);
  MPI_File_close(&pFileimg);

  MPI_Barrier(MYMPI_COMM);
  
  free(gridss_real);
  free(gridss_img);

 #endif // WRITE_DATA
  return;
}

/*
  for (int isector = 0; isector < size; isector++)
  {
    for (unsigned int i = 0; i < size_of_grid / 2; i++)
    {
      gridss_real[i] = gridss_w[2 * i];
      gridss_img[i] = gridss_w[2 * i + 1];
    }
    if (num_w_planes > 1)
    {
      for (int iw = 0; iw < num_w_planes; iw++)
        for (int iv = 0; iv < yaxis; iv++)
          for (int iu = 0; iu < xaxis; iu++)
          {
            unsigned int global_index = (iu + (iv + isector * yaxis) * xaxis + iw * grid_size_x * grid_size_y) * sizeof(double);
            unsigned int index = iu + iv * xaxis + iw * xaxis * yaxis;
            MPI_File_write_at_all(pFilereal, global_index, &gridss_real[index], 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
            MPI_File_write_at_all(pFileimg, global_index, &gridss_img[index], 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
          }
    }
    else
    {
      MPI_File_write_at_all(pFilereal, isector * (size_of_grid / 2) * sizeof(double), gridss_real, size_of_grid / 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
      MPI_File_write_at_all(pFileimg, isector * (size_of_grid / 2) * sizeof(double), gridss_img, size_of_grid / 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
  }

  MPI_File_close(&pFilereal);
  MPI_File_close(&pFileimg);

  free(gridss_real);
  free(gridss_img);
  free(gridss_w);
#endif // WRITE_DATA

  return;
}
*/
