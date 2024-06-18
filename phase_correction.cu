#ifdef _OPENMP
#include <omp.h>
#endif

#if !defined(ACCOMP)
#include "w-stacking.h"
#else
#include "w-stacking_omp.h"
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "errcodes.h"
#include "proto.h"

#ifdef __CUDACC__

__global__ void phase_g(int xaxis, 
		        int yaxis,
			int num_w_planes,
			double * gridss,
			double * image_real,
			double * image_imag,
			double wmin,
			double dw,
			double dwnorm,
			int xaxistot,
			int yaxistot,
			double resolution,
			int nbucket)
{
  long gid = blockIdx.x*blockDim.x + threadIdx.x;
  double add_term_real;
  double add_term_img;
  double wterm;
  long arraysize = (long)((xaxis*yaxis*num_w_planes)/nbucket);

  if(gid < arraysize)
    {
      long gid_aux = nbucket*gid;
      for(int iaux=0; iaux<nbucket; iaux++) 
	{
	  int iw = gid_aux/(xaxis*yaxis);
	  int ivaux = gid_aux%(xaxis*yaxis);
	  int iv = ivaux/xaxis;
	  int iu = ivaux%xaxis;
	  long index = 2*gid_aux;
	  long img_index = iu+iv*xaxis;

	  wterm = wmin + iw*dw;

#ifdef PHASE_ON
	  if (num_w_planes > 1)
	    {
	      double xcoord = (double)(iu-xaxistot/2);
	      if(xcoord < 0.0)xcoord = (double)(iu+xaxistot/2);
	      xcoord = sin(xcoord*resolution);
	      double ycoord = (double)(iv-yaxistot/2);
	      if(ycoord < 0.0)ycoord = (double)(iv+yaxistot/2);
	      ycoord = sin(ycoord*resolution);

	      double preal, pimag;
	      double radius2 = (xcoord*xcoord+ycoord*ycoord);

	      preal = cos(2.0*PI*wterm*(sqrt(1-radius2)-1.0));
	      pimag = sin(2.0*PI*wterm*(sqrt(1-radius2)-1.0));

	      double p,q,r,s;
	      p = gridss[index];
	      q = gridss[index+1];
	      r = preal;
	      s = pimag;

	      //printf("%d %d %d %ld %ld\n",iu,iv,iw,index,img_index);

	      add_term_real = (p*r-q*s)*dwnorm*sqrt(1-radius2);
	      add_term_img = (p*s+q*r)*dwnorm*sqrt(1-radius2);
	      atomicAdd(&(image_real[img_index]),add_term_real);
	      atomicAdd(&(image_imag[img_index]),add_term_img);
	    } else {
	    atomicAdd(&(image_real[img_index]),gridss[index]);
	    atomicAdd(&(image_imag[img_index]),gridss[index+1]);
	  }
#else
	  atomicAdd(&(image_real[img_index]),gridss[index]);
	  atomicAdd(&(image_imag[img_index]),gridss[index+1]);
#endif // end of PHASE_ON
	  gid_aux++;
	}
    }

}

#endif

void phase_correction(double* gridss, double* image_real, double* image_imag, int xaxis, int yaxis, int num_w_planes, int xaxistot, int yaxistot,
		      double resolution, double wmin, double wmax, int num_threads, int rank)
{
  double dw = (wmax-wmin)/(double)num_w_planes;
  double wterm = wmin+0.5*dw;
  double dwnorm = dw/(wmax-wmin);
  
#ifdef CUDACC
  
  // WARNING: nbucket MUST be chosen such that xaxis*yaxis*num_w_planes is a multiple of nbucket
  int nbucket = 1;
  int Nth = NTHREADS;
  long Nbl = (long)((num_w_planes*xaxis*yaxis)/Nth/nbucket) + 1;
  if(NWORKERS == 1) {Nbl = 1; Nth = 1;};
  
  int ndevices;
  cudaGetDeviceCount(&ndevices);
  cudaSetDevice(rank % ndevices);
  
  if ( rank == 0 ) {
    if (0 == ndevices) {
      
      shutdown_wstacking(NO_ACCELERATORS_FOUND, "No accelerators found", __FILE__, __LINE__ );
    }
    
  }
  printf("Running rank %d using GPU %d\n", rank, rank % ndevices);
#ifdef NVIDIA
  prtAccelInfo();
#endif
  
  cudaError_t mmm;
  double * image_real_g;
  double * image_imag_g;
  
#if !defined(CUFFTMP)
  double * gridss_g;
  
  long long unsigned size_finta = (long long unsigned)(2*(long long unsigned)num_w_planes*(long long unsigned)xaxis*(long long unsigned)yaxis);
  
  mmm = cudaMalloc(&gridss_g, (size_t)(size_finta*sizeof(double)));
  mmm = cudaMemcpy(gridss_g, gridss, (size_t)(size_finta*sizeof(double)), cudaMemcpyHostToDevice);
#endif
		   
  mmm=cudaMalloc(&image_real_g, xaxis*yaxis*sizeof(double));
  //printf("CUDA ERROR 2 %s\n",cudaGetErrorString(mmm));
  mmm=cudaMalloc(&image_imag_g, xaxis*yaxis*sizeof(double));
  //printf("CUDA ERROR 3 %s\n",cudaGetErrorString(mmm));
  
  //printf("CUDA ERROR 4 %s\n",cudaGetErrorString(mmm));
  mmm=cudaMemset(image_real_g, 0.0, xaxis*yaxis*sizeof(double));
  //printf("CUDA ERROR 5 %s\n",cudaGetErrorString(mmm));
  mmm=cudaMemset(image_imag_g, 0.0, xaxis*yaxis*sizeof(double));
  //printf("CUDA ERROR 6 %s\n",cudaGetErrorString(mmm));
		   
  // call the phase correction kernel
  phase_g <<<Nbl,Nth>>> (xaxis,
			 yaxis,
			 num_w_planes,
#if defined(CUFFTMP)
			 gridss,
#else
			 gridss_g,
#endif
			 image_real_g,
			 image_imag_g,
			 wmin,
			 dw,
			 dwnorm,
			 xaxistot,
			 yaxistot,
			 resolution,
			 nbucket);
		   
  mmm = cudaMemcpy(image_real, image_real_g, xaxis*yaxis*sizeof(double), cudaMemcpyDeviceToHost);
  //printf("CUDA ERROR 7 %s\n",cudaGetErrorString(mmm));
  mmm = cudaMemcpy(image_imag, image_imag_g, xaxis*yaxis*sizeof(double), cudaMemcpyDeviceToHost);
  //printf("CUDA ERROR 8 %s\n",cudaGetErrorString(mmm));
		   
#if !defined(CUFFTMP)
  cudaFree(gridss_g);
#else
  cudaFree(gridss);
#endif
		   
		   
#else
		   
#ifndef ACCOMP
		   
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
		   
#pragma omp parallel for collapse(2) private(wterm) 
  for (int iw=0; iw<num_w_planes; iw++)
    {
      for (int iv=0; iv<yaxis; iv++)
	for (int iu=0; iu<xaxis; iu++)
	  {
			     
	    unsigned int index = 2*(iu+iv*xaxis+xaxis*yaxis*iw);
	    unsigned int img_index = iu+iv*xaxis;
	    wterm = wmin + iw*dw;
#ifdef PHASE_ON
	    if (num_w_planes > 1)
	      {
		double xcoord = (double)(iu-xaxistot/2);
		if(xcoord < 0.0)xcoord = (double)(iu+xaxistot/2);
		xcoord = sin(xcoord*resolution);
		double ycoord = (double)(iv-yaxistot/2);
		if(ycoord < 0.0)ycoord = (double)(iv+yaxistot/2);
		ycoord = sin(ycoord*resolution);
				 
		double preal, pimag;
		double radius2 = (xcoord*xcoord+ycoord*ycoord);
		if(xcoord <= 1.0)
		  {
		    preal = cos(2.0*PI*wterm*(sqrt(1-radius2)-1.0));
		    pimag = sin(2.0*PI*wterm*(sqrt(1-radius2)-1.0));
		  } else {
		  preal = cos(-2.0*PI*wterm*(sqrt(radius2-1.0)-1));
		  pimag = 0.0;
		}
				 
		preal = cos(2.0*PI*wterm*(sqrt(1-radius2)-1.0));
		pimag = sin(2.0*PI*wterm*(sqrt(1-radius2)-1.0));

		double p,q,r,s;
		p = gridss[index];
		q = gridss[index+1];
		r = preal;
		s = pimag;

		//printf("%d %d %d %ld %ld\n",iu,iv,iw,index,img_index);
#pragma omp atomic
		image_real[img_index] += (p*r-q*s)*dwnorm*sqrt(1-radius2);
#pragma omp atomic
		image_imag[img_index] += (p*s+q*r)*dwnorm*sqrt(1-radius2);
	      } else {
#pragma omp atomic
	      image_real[img_index] += gridss[index];
#pragma omp atomic
	      image_imag[img_index] += gridss[index+1];
	    }
#else
#pragma omp atomic
	    image_real[img_index] += gridss[index];
#pragma omp atomic
	    image_imag[img_index] += gridss[index+1];
#endif // end of PHASE_ON

	  }
    }

#else
  omp_set_default_device(rank % omp_get_num_devices());
	
#if !defined(__clang__)

#pragma omp target teams distribute parallel for collapse(2) simd private(wterm) map(to:gridss[0:2*num_w_planes*xaxis*yaxis]) map(from:image_real[0:xaxis*yaxis]) map(from:image_imag[0:xaxis*yaxis])

#else

#pragma omp target teams distribute parallel for collapse(2) private(wterm) map(to:gridss[0:2*num_w_planes*xaxis*yaxis]) map(from:image_real[0:xaxis*yaxis]) map(from:image_imag[0:xaxis*yaxis])
#endif
	
  for (int iw=0; iw<num_w_planes; iw++)
    {
      for (int iv=0; iv<yaxis; iv++)
	for (int iu=0; iu<xaxis; iu++)
	  {

	    long index = 2*(iu+iv*xaxis+xaxis*yaxis*iw);
	    long img_index = iu+iv*xaxis;
	    wterm = wmin + iw*dw;
#ifdef PHASE_ON
	    if (num_w_planes > 1)
	      {
		double xcoord = (double)(iu-xaxistot/2);
		if(xcoord < 0.0)xcoord = (double)(iu+xaxistot/2);
		xcoord = sin(xcoord*resolution);
		double ycoord = (double)(iv-yaxistot/2);
		if(ycoord < 0.0)ycoord = (double)(iv+yaxistot/2);
		ycoord = sin(ycoord*resolution);

		double preal, pimag;
		double radius2 = (xcoord*xcoord+ycoord*ycoord);
		if(xcoord <= 1.0)
		  {
		    preal = cos(2.0*PI*wterm*(sqrt(1-radius2)-1.0));
		    pimag = sin(2.0*PI*wterm*(sqrt(1-radius2)-1.0));
		  } else {
		  preal = cos(-2.0*PI*wterm*(sqrt(radius2-1.0)-1));
		  pimag = 0.0;
		}

		preal = cos(2.0*PI*wterm*(sqrt(1-radius2)-1.0));
		pimag = sin(2.0*PI*wterm*(sqrt(1-radius2)-1.0));

		double p,q,r,s;
		p = gridss[index];
		q = gridss[index+1];
		r = preal;
		s = pimag;

		//printf("%d %d %d %ld %ld\n",iu,iv,iw,index,img_index);
#pragma omp atomic
		image_real[img_index] += (p*r-q*s)*dwnorm*sqrt(1-radius2);
#pragma omp atomic
		image_imag[img_index] += (p*s+q*r)*dwnorm*sqrt(1-radius2);
	      } else {
#pragma omp atomic
	      image_real[img_index] += gridss[index];
#pragma omp atomic
	      image_imag[img_index] += gridss[index+1];
	    }
#else
#pragma omp atomic
	    image_real[img_index] += gridss[index];
#pragma omp atomic
	    image_imag[img_index] += gridss[index+1];
#endif // end of PHASE_ON

	  }
    }
	
#endif	
#endif // end of __CUDACC__


}
