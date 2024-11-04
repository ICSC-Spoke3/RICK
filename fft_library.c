#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
/* #include <omp.h>  to be included after checking the MPI version works */
#include <fftw3-mpi.h>

#define PI 3.14159265359

void fftw_data(
    int grid_size_x,
    int grid_size_y,
    int num_w_planes,
    int num_threads,
    MPI_Comm MYMPI_COMM,
    int size,
    int rank,
    double *grid,
    double *gridss)
{

  // FFT transform the data (using distributed FFTW)
  if (rank == 0)
    printf("RICK FFT\n");

  fftw_mpi_init();

  // double start = CPU_TIME_wt;

  int xaxis = grid_size_x;
  int yaxis = grid_size_y / size;

  fftw_plan plan;
  fftw_complex *fftwgrid;
  ptrdiff_t alloc_local, local_n0, local_0_start;
  double norm = 1.0 / (double)(grid_size_x * grid_size_y);

  // Use the hybrid MPI-OpenMP FFTW
#ifdef HYBRID_FFTW
  fftw_plan_with_nthreads(num_threads);
#endif
  // map the 1D array of complex visibilities to a 2D array required by FFTW (complex[*][2])
  // x is the direction of contiguous data and maps to the second parameter
  // y is the parallelized direction and corresponds to the first parameter (--> n0)
  // and perform the FFT per w plane
  alloc_local = fftw_mpi_local_size_2d(grid_size_y, grid_size_x, MYMPI_COMM, &local_n0, &local_0_start);
  fftwgrid = fftw_alloc_complex(alloc_local);
  plan = fftw_mpi_plan_dft_2d(grid_size_y, grid_size_x, fftwgrid, fftwgrid, MYMPI_COMM, FFTW_BACKWARD, FFTW_ESTIMATE);

  unsigned int fftwindex = 0;
  unsigned int fftwindex2D = 0;
  for (int iw = 0; iw < num_w_planes; iw++)
  {
    // printf("FFTing plan %d\n",iw);
    // select the w-plane to transform

#ifdef HYBRID_FFTW
#pragma omp parallel for collapse(2) num_threads(num_threads)
#endif
    for (int iv = 0; iv < yaxis; iv++)
    {
      for (int iu = 0; iu < xaxis; iu++)
      {
        fftwindex2D = iu + iv * xaxis;
        fftwindex = 2 * (fftwindex2D + iw * xaxis * yaxis);
        fftwgrid[fftwindex2D][0] = grid[fftwindex];
        fftwgrid[fftwindex2D][1] = grid[fftwindex + 1];
      }
    }

    // do the transform for each w-plane
    fftw_execute(plan);

    // save the transformed w-plane

#ifdef HYBRID_FFTW
#pragma omp parallel for collapse(2) num_threads(num_threads)
#endif
    for (int iv = 0; iv < yaxis; iv++)
    {
      for (int iu = 0; iu < xaxis; iu++)
      {
        fftwindex2D = iu + iv * xaxis;
        fftwindex = 2 * (fftwindex2D + iw * xaxis * yaxis);
        gridss[fftwindex] = norm * fftwgrid[fftwindex2D][0];
        gridss[fftwindex + 1] = norm * fftwgrid[fftwindex2D][1];
      }
    }
  }

#ifdef HYBRID_FFTW
  fftw_cleanup_threads();
#endif
  fftw_destroy_plan(plan);
  fftw_free(fftwgrid);

  if (size > 1)
  {
    MPI_Barrier(MYMPI_COMM);
  }

  return;
}
