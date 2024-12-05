#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <stdatomic.h>
#ifndef RICK_GPU
#include <fftw3-mpi.h>
#else
#include <cufftXt.h>
#include <cuda_runtime.h>
#endif
#include "ricklib.h"
#include <omp.h>

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


#ifndef RICK_GPU

  // FFT transform the data (using distributed FFTW)
  if (rank == 0)
  {
    std::cout << "RICK FFT" << std::endl;
  }

  FFT_INIT;

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
  if (rank == 0)
    std::cout << "Using " << num_threads << " threads for the FFT" << std::endl;
#endif

  // map the 1D array of complex visibilities to a 2D array required by FFTW (complex[*][2])
  // x is the direction of contiguous data and maps to the second parameter
  // y is the parallelized direction and corresponds to the first parameter (--> n0)
  // and perform the FFT per w plane
  alloc_local = fftw_mpi_local_size_2d(grid_size_y, grid_size_x, MYMPI_COMM, &local_n0, &local_0_start);
  fftwgrid = fftw_alloc_complex(alloc_local);
  plan = fftw_mpi_plan_dft_2d(grid_size_y, grid_size_x, fftwgrid, fftwgrid, MYMPI_COMM, FFTW_BACKWARD, FFTW_ESTIMATE);

  int fftwindex = 0;
  int fftwindex2D = 0;
  for (int iw = 0; iw < num_w_planes; iw++)
  {
    // printf("FFTing plan %d\n",iw);
    //  select the w-plane to transform

#ifdef HYBRID_FFTW
#pragma omp parallel for collapse(2) num_threads(num_threads) private(fftwindex, fftwindex2D)
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
#pragma omp parallel for collapse(2) num_threads(num_threads) private(fftwindex, fftwindex2D)
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

  FFT_CLEANUP;
  
  fftw_destroy_plan(plan);
  fftw_free(fftwgrid);

  if (size > 1)
  {
    MPI_Barrier(MYMPI_COMM);
  }

#else

	cudaError_t mmm;
	cufftResult_t status;


  // FFT transform the data (using distributed FFTW)
  if (rank == 0)
  {
    printf("RICK cuFFTXt\n");
  }

  // double start = CPU_TIME_wt;

  int xaxis = grid_size_x;
  int yaxis = grid_size_y / size;

  double norm = 1.0 / (double)(grid_size_x * grid_size_y);

  cufftDoubleComplex *fftwgrid;
  fftwgrid = (cufftDoubleComplex *)malloc(sizeof(cufftDoubleComplex) * 2 * num_w_planes * xaxis * yaxis);

  // Initialize devices
  int nDevices = 4;
  cudaGetDeviceCount(&nDevices);
  int deviceIds[nDevices];

  long fftwindex = 0;
  long fftwindex2D = 0;

  size_t workspace_sizes[nDevices];
  cudaLibXtDesc *fftwgrid_multig;

  cufftHandle plan;
  cufftCreate(&plan);

  cudaStream_t stream{};
  cudaStreamCreate(&stream);
  cufftSetStream(plan, stream);

  status = cufftXtSetGPUs(plan, nDevices, deviceIds);
  if (status != CUFFT_SUCCESS)
  {
    printf("!!! cufftXtSetGPUs ERROR %d !!!\n", status);
  }

  status = cufftMakePlan2d(plan, xaxis, yaxis, CUFFT_Z2Z, workspace_sizes);
  if (status != CUFFT_SUCCESS)
  {
    printf("!!! cufftMakePlan2d ERROR %d !!!\n", status);
  }
  cudaDeviceSynchronize();

  for (int iw = 0; iw < num_w_planes; iw++)
  {
    for (int iv = 0; iv < yaxis; iv++)
    {
      for (int iu = 0; iu < xaxis; iu++)
      {
        fftwindex2D = iu + iv * xaxis;
        fftwindex = 2 * (fftwindex2D + iw * xaxis * yaxis);
        fftwgrid[fftwindex2D].x = grid[fftwindex];
        fftwgrid[fftwindex2D].y = grid[fftwindex + 1];
      }
    }

    status = cufftXtMalloc(plan, &fftwgrid_multig, CUFFT_XT_FORMAT_INPLACE);
    if (status != CUFFT_SUCCESS)
    {
      printf("!!! cuufftXtMalloc ERROR %d !!!\n", status);
    }
    cudaDeviceSynchronize();

    status = cufftXtMemcpy(plan, fftwgrid_multig, fftwgrid, CUFFT_COPY_HOST_TO_DEVICE);
    if (status != CUFFT_SUCCESS)
    {
      printf("!!! cufftXtMemcpy fftwgrid_multig ERROR %d !!!\n", status);
    }
    cudaDeviceSynchronize();

    status = cufftXtExecDescriptorZ2Z(plan, fftwgrid_multig, fftwgrid_multig, CUFFT_INVERSE);
    if (status != CUFFT_SUCCESS)
    {
      printf("!!! cufftXtExec ERROR %d !!!\n", status);
    }
    cudaDeviceSynchronize();

    status = cufftXtMemcpy(plan, fftwgrid, fftwgrid_multig, CUFFT_COPY_DEVICE_TO_HOST);
    if (status != CUFFT_SUCCESS)
    {
      printf("!!! cufftXtMemcpy fftwgrid ERROR %d !!!\n", status);
    }
    cudaDeviceSynchronize();

    cufftXtFree(fftwgrid_multig);

    for (int iv = 0; iv < yaxis; iv++)
    {
      for (int iu = 0; iu < xaxis; iu++)
      {
        fftwindex2D = iu + iv * xaxis;
        fftwindex = 2 * (fftwindex2D + iw * xaxis * yaxis);
        gridss[fftwindex] = norm * fftwgrid[fftwindex2D].x;
        gridss[fftwindex + 1] = norm * fftwgrid[fftwindex2D].y;
      }
    }
  }

  cufftDestroy(plan);
  cudaStreamDestroy(stream);
  free(fftwgrid);

#endif

  return;
}