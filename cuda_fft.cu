#include <stdlib.h>
#include <stdio.h>
#include <cufftMp.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <complex.h>
#include "cuComplex.h"
#include "proto.h"
#include "errcodes.h"
#include <time.h>
#include <unistd.h>

#if defined(CUFFTMP) && defined(USE_FFTW)



__global__ void write_grid(
	int num_w_planes,
	int xaxis,
	int yaxis,
	cufftDoubleComplex * fftwgrid,
	double * grid,
	int iw)
{
  unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid<yaxis*xaxis)
    {
      unsigned int fftwindex2D = gid;
      unsigned int fftwindex = 2*(fftwindex2D + iw * xaxis * yaxis);
      fftwgrid[fftwindex2D].x = grid[fftwindex];
      fftwgrid[fftwindex2D].y = grid[fftwindex+1];
    }  
}


__global__ void write_gridss(
			     int num_w_planes,
			     int xaxis,
			     int yaxis,
			     cufftDoubleComplex * fftwgrid,
			     double * gridss,
			     double norm,
			     int iw)
  
{
  unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid<yaxis*xaxis)
    {
      unsigned int fftwindex2D = gid;
      unsigned int fftwindex = 2*(fftwindex2D + iw * xaxis * yaxis);
      gridss[fftwindex] = norm*fftwgrid[fftwindex2D].x;
      gridss[fftwindex+1] = norm*fftwgrid[fftwindex2D].y;
    }
}







void cuda_fft(
	      int num_w_planes,
	      int grid_size_x,
	      int grid_size_y,
	      int xaxis,
	      int yaxis,
	      double * grid,
	      double * gridss,
	      int rank,
	      MPI_Comm comm)
{

  int ndevices;
  cudaGetDeviceCount(&ndevices);
  cudaSetDevice(rank % ndevices);

  if ( rank == 0 ) {
    if (0 == ndevices) {
      return;
      //shutdown_wstacking(NO_ACCELERATORS_FOUND, "No accelerators found", __FILE__, __LINE__ );
    }
  }

  cudaError_t mmm;
  cufftResult_t status;

  cufftDoubleComplex *fftwgrid;

  
  // Alloco fftwgrid su GPU utilizzando cudaMalloc

  long long unsigned size_finta_fft = (long long unsigned)((long long unsigned)xaxis*(long long unsigned)yaxis);

  mmm=cudaMalloc(&fftwgrid, (size_t)(size_finta_fft*sizeof(cufftDoubleComplex)));
  if (mmm != cudaSuccess) {printf("!!! cuda_fft.cu cudaMalloc ERROR %d !!!\n", mmm);}

  int Nth = 32;
  myuint Nbl = (myuint)((yaxis*xaxis)/Nth + 1);  
  

  // Plan creation

  cufftHandle plan;
  status = cufftCreate(&plan);
  if (status != CUFFT_SUCCESS) {printf("!!! cufftCreate ERROR %d !!!\n", status);}

  cudaStream_t stream{};
  cudaStreamCreate(&stream);


  status = cufftMpAttachComm(plan, CUFFT_COMM_MPI, &comm);
  if (status != CUFFT_SUCCESS) {printf("!!! cufftMpAttachComm ERROR %d !!!\n", status);}

  status = cufftSetStream(plan, stream);
  if (status != CUFFT_SUCCESS) {printf("!!! cufftSetStream ERROR %d !!!\n", status);}

  size_t workspace;
  status = cufftMakePlan2d(plan, grid_size_x, grid_size_y, CUFFT_Z2Z, &workspace);
  if (status != CUFFT_SUCCESS) {printf("!!! cufftMakePlan2d ERROR %d !!!\n", status);}
  cudaDeviceSynchronize();


  double norm = 1.0/(double)(grid_size_x*grid_size_y);
  
  // Grid composition
  cudaLibXtDesc *fftwgrid_g;
  cudaLibXtDesc *fftwgrid_g2;

  
  status = cufftXtMalloc(plan, &fftwgrid_g2, CUFFT_XT_FORMAT_INPLACE);
  if (status != CUFFT_SUCCESS) {printf("!!! cufftXtMalloc 2 ERROR %d !!!\n", status);}
  cudaDeviceSynchronize();
  
  
  mmm = cudaStreamSynchronize(stream);
  if (mmm != cudaSuccess) {printf("!!! cudaStreamSynchronize ERROR %d !!!\n", mmm);}

  for (int iw = 0; iw < num_w_planes; iw++)
    {
        
      //printf("Task %d, FFTing plane %d...\n", rank, iw);

      //Define fftwgrid with a cuda kernel
      write_grid<<<Nbl, Nth>>>(num_w_planes, xaxis, yaxis, fftwgrid, grid, iw);
      cudaDeviceSynchronize();

      //Allocate the first descriptor inside the loop
      status = cufftXtMalloc(plan, &fftwgrid_g, CUFFT_XT_FORMAT_INPLACE);
      if (status != CUFFT_SUCCESS) {printf("!!! cufftXtMalloc ERROR %d !!!\n", status);}

      cudaStreamSynchronize(stream);

      //Copy the array to be transformed onto the descriptor structure array 
      mmm = cudaMemcpy(fftwgrid_g->descriptor->data[0], fftwgrid, (size_t)(size_finta_fft*sizeof(cufftDoubleComplex)), cudaMemcpyDeviceToDevice);
      if (mmm != cudaSuccess) {printf("!!! cudaMemcpy 1 ERROR %d !!!\n", mmm);}

      //Perform the FFT
      status = cufftXtExecDescriptor(plan, fftwgrid_g, fftwgrid_g, CUFFT_INVERSE);
      if (status != CUFFT_SUCCESS) {printf("!!! cufftXtExecDescriptor ERROR %d !!!\n", status);}

      mmm = cudaStreamSynchronize(stream);
      if (mmm != cudaSuccess) {printf("!!! cudaStreamSynchronize 2 ERROR %d !!!\n", mmm);}

      cudaDeviceSynchronize();

      //Put the data in the correct order as required by cufftMP
      status = cufftXtMemcpy(plan, fftwgrid_g2, fftwgrid_g, CUFFT_COPY_DEVICE_TO_DEVICE);
      if (status != CUFFT_SUCCESS) {printf("!!! cufftXtMemcpy dtd fftwgrid ERROR %d !!!\n", status);}

      //Copy the result descriptor structure array again onto the original fftwgrid
      mmm = cudaMemcpy(fftwgrid, fftwgrid_g2->descriptor->data[0], (size_t)(size_finta_fft*sizeof(cufftDoubleComplex)), cudaMemcpyDeviceToDevice);
      if (mmm != cudaSuccess) {printf("!!! cudaMemcpy 2 ERROR %d !!!\n", mmm);}

      //Write gridss starting from fftwgrid
      write_gridss<<<Nbl, Nth>>>(num_w_planes, xaxis, yaxis, fftwgrid, gridss, norm, iw);

      //Free the first descriptor
      status=cufftXtFree(fftwgrid_g);
      if (status != CUFFT_SUCCESS) {printf("!!! cudaFree fftwgrid_g ERROR %d !!!\n", mmm);}

      cudaDeviceSynchronize();
      
    }

  status=cufftXtFree(fftwgrid_g2);
  if (status != CUFFT_SUCCESS) {printf("!!! cudaFree fftwgrid_g2 ERROR %d !!!\n", mmm);}
  status = cufftDestroy(plan);
  if (status != CUFFT_SUCCESS) {printf("!!! cufftDestroy fftwgrid ERROR %d !!!\n", status);}
  
  mmm = cudaFree(grid);
  if (mmm != cudaSuccess) {printf("!!! cudaFree grid ERROR %d !!!\n", mmm);}
  
  mmm = cudaFree(fftwgrid);
  if (mmm != cudaSuccess) {printf("!!! cudaFree fftwgrid ERROR %d !!!\n", mmm);}

  cudaStreamDestroy(stream);

}
#endif
