#ifdef _OPENMP
#include <omp.h>
#endif

#if !defined(ACCOMP)
#include "w-stacking.hip.hpp"
#else
#include "w-stacking_omp.h"
#endif

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "errcodes.h"
#include "proto.h"

#ifdef __HIPCC__

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

#ifdef HIPCC

	// WARNING: nbucket MUST be chosen such that xaxis*yaxis*num_w_planes is a multiple of nbucket
	int nbucket = 1;
	int Nth = NTHREADS;
        long Nbl = (long)((num_w_planes*xaxis*yaxis)/Nth/nbucket) + 1;
        if(NWORKERS == 1) {Nbl = 1; Nth = 1;};

	int ndevices;
	int m = hipGetDeviceCount(&ndevices);
	m = hipSetDevice(rank % ndevices);

	if ( rank == 0 ) {
	  if (0 == ndevices) {

	    shutdown_wstacking(NO_ACCELERATORS_FOUND, "No accelerators found", __FILE__, __LINE__ );
	  }

	}
	  printf("Running rank %d using GPU %d\n", rank, rank % ndevices);
	 #ifdef NVIDIA
	  prtAccelInfo();
	 #endif

	hipError_t mmm;
	double * image_real_g;
	double * image_imag_g;
	double * gridss_g;

	
	//Copy gridss on the device
	mmm=hipMalloc(&gridss_g, 2*num_w_planes*xaxis*yaxis*sizeof(double));
	mmm=hipMemcpy(gridss_g, gridss, 2*num_w_planes*xaxis*yaxis*sizeof(double), hipMemcpyHostToDevice);

	mmm=hipMalloc(&image_real_g, xaxis*yaxis*sizeof(double));
	//printf("HIP ERROR 2 %s\n",hipGetErrorString(mmm));
	mmm=hipMalloc(&image_imag_g, xaxis*yaxis*sizeof(double));
	//printf("HIP ERROR 3 %s\n",hipGetErrorString(mmm));

	//printf("HIP ERROR 4 %s\n",hipGetErrorString(mmm));
	mmm=hipMemset(image_real_g, 0.0, xaxis*yaxis*sizeof(double));
	//printf("HIP ERROR 5 %s\n",hipGetErrorString(mmm));
	mmm=hipMemset(image_imag_g, 0.0, xaxis*yaxis*sizeof(double));
	//printf("HIP ERROR 6 %s\n",hipGetErrorString(mmm));

	// call the phase correction kernel
	phase_g <<<Nbl,Nth>>> (xaxis,
                               yaxis,
			       num_w_planes,
                               gridss_g,
                               image_real_g,
                               image_imag_g,
                               wmin,
                               dw,
                               dwnorm,
                               xaxistot,
                               yaxistot,
                               resolution,
			       nbucket);

	mmm = hipMemcpy(image_real, image_real_g, xaxis*yaxis*sizeof(double), hipMemcpyDeviceToHost);
	//printf("HIP ERROR 7 %s\n",hipGetErrorString(mmm));
	mmm = hipMemcpy(image_imag, image_imag_g, xaxis*yaxis*sizeof(double), hipMemcpyDeviceToHost);
	//printf("HIP ERROR 8 %s\n",hipGetErrorString(mmm));

	mmm= hipFree(gridss_g);
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
#endif // end of __HIPCC__


}
