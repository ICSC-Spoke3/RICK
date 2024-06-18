#ifdef _OPENMP
#include <omp.h>
#endif
#include "w-stacking.hip.hpp"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef __HIPCC__
#include "allvars_rccl.hip.hpp"
#endif

#include "proto.h"

#ifdef ACCOMP
#pragma omp  declare target
#endif
#ifdef __HIPCC__
double __device__
#else
double
#endif
// Gaussian Kernel
gauss_kernel_norm(double norm, double std22, double u_dist, double v_dist)
{
     double conv_weight;
     conv_weight = norm * exp(-((u_dist*u_dist)+(v_dist*v_dist))*std22);
     return conv_weight;
}

void makeGaussKernel(double * kernel,
		     int KernelLen,
		     int increaseprecision,
		     double std22)
{

  double norm = std22/PI;
  int n = increaseprecision*KernelLen, mid = n / 2;
  for (int i = 0; i != mid + 1; i++) {
      double term = (double)i/(double)increaseprecision;
      kernel[mid + i] = sqrt(norm) * exp(-(term*term)*std22);
  }

  for (int i = 0; i != mid; i++) kernel[i] = kernel[n - 1 - i];
//  for (int i = 0; i < n; i++) printf("%f\n",kernel[i]);

}

// Kaiser-Bessel Kernel: it is adapted from WSClean
double bessel0(double x, double precision) {
  // Calculate I_0 = SUM of m 0 -> inf [ (x/2)^(2m) ]
  // This is the unnormalized bessel function of order 0.
  double d = 0.0, ds = 1.0, sum = 1.0;
  do {
    d += 2.0;
    ds *= x * x / (d * d);
    sum += ds;
  } while (ds > sum * precision);
  return sum;
}
void makeKaiserBesselKernel(double * kernel,
		            int KernelLen,
			    int increaseprecision,
                            double alpha,
                            double overSamplingFactor,
                            int withSinc) {
  int n = increaseprecision*KernelLen, mid = n / 2;
  double * sincKernel = (double*)malloc((mid + 1) * sizeof(*sincKernel));
  const double filterRatio = 1.0 / overSamplingFactor;
  sincKernel[0] = filterRatio;
  for (int i = 1; i != mid + 1; i++) {
    double x = i;
    sincKernel[i] =
        withSinc ? (sin(PI * filterRatio * x) / (PI * x)) : filterRatio;
  }
  const double normFactor = overSamplingFactor / bessel0(alpha, 1e-8);
  for (int i = 0; i != mid + 1; i++) {
    double term = (double)i / mid;
    kernel[mid + i] = sincKernel[i] *
                bessel0(alpha * sqrt(1.0 - (term * term)), 1e-8) *
                normFactor;
  }
  for (int i = 0; i != mid; i++) kernel[i] = kernel[n - 1 - i];
  //for (int i = 0; i < n; i++) printf("%f\n",kernel[i]);
}


#ifdef ACCOMP
#pragma omp end declare target
#endif

#ifdef __HIPCC__
//double __device__ gauss_kernel_norm(double norm, double std22, double u_dist, double v_dist)
//{
//     double conv_weight;
//     conv_weight = norm * exp(-((u_dist*u_dist)+(v_dist*v_dist))*std22);
//     return conv_weight;
//}

__global__ void convolve_g(
			   int num_w_planes,
			   myuint num_points,
			   myuint freq_per_chan,
			   myuint polarizations,
			   double* uu,
			   double* vv,
			   double* ww,
			   float* vis_real,
			   float* vis_img,
			   float* weight,
			   double dx,
			   double dw,
			   int KernelLen,
			   int grid_size_x,
			   int grid_size_y,
			   double* grid,
			  #if defined(GAUSS_HI_PRECISION)
			   double std22
			  #else
			   double std22,
			   double* convkernel
			  #endif
			   )
			   


{
  //printf("DENTRO AL KERNEL\n");
  myuint gid = blockIdx.x*blockDim.x + threadIdx.x;
  if(gid < num_points)
    {
      myuint i = gid;
      myull visindex = i*freq_per_chan*polarizations;
      double norm = std22/PI;

      int j, k;

      /* Convert UV coordinates to grid coordinates. */
      double pos_u = uu[i] / dx;
      double pos_v = vv[i] / dx;
      double ww_i  = ww[i] / dw;

      int grid_w = (int)ww_i;
      int grid_u = (int)pos_u;
      int grid_v = (int)pos_v;

      // check the boundaries
      myuint jmin = (grid_u > KernelLen - 1) ? grid_u - KernelLen : 0;
      myuint jmax = (grid_u < grid_size_x - KernelLen) ? grid_u + KernelLen : grid_size_x - 1;
      myuint kmin = (grid_v > KernelLen - 1) ? grid_v - KernelLen : 0;
      myuint kmax = (grid_v < grid_size_y - KernelLen) ? grid_v + KernelLen : grid_size_y - 1;


      // Convolve this point onto the grid.
      for (k = kmin; k <= kmax; k++)
        {

	  double v_dist = (double)k - pos_v;
	  int increaseprecision = 5;
	  
	  for (j = jmin; j <= jmax; j++)
            {
	      double u_dist = (double)j+0.5 - pos_u;
	      myuint iKer = 2 * (j + k*grid_size_x + grid_w*grid_size_x*grid_size_y);
	      int jKer = (int)(increaseprecision * (fabs(u_dist+(double)KernelLen)));
	      int kKer = (int)(increaseprecision * (fabs(v_dist+(double)KernelLen)));
	      
	     #ifdef GAUSS_HI_PRECISION
	      double conv_weight = gauss_kernel_norm(norm,std22,u_dist,v_dist);
	     #endif
	     #ifdef GAUSS
	      double conv_weight = convkernel[jKer]*convkernel[kKer];
	     #endif
	     #ifdef KAISERBESSEL
	      double conv_weight = convkernel[jKer]*convkernel[kKer];
	     #endif

	      // Loops over frequencies and polarizations
	      double add_term_real = 0.0;
	      double add_term_img = 0.0;
	      myull ifine = visindex;
	      for (myuint ifreq=0; ifreq<freq_per_chan; ifreq++)
		{
		  myuint iweight = visindex/freq_per_chan;
		  for (myuint ipol=0; ipol<polarizations; ipol++)
		    {
                      double vistest = (double)vis_real[ifine];
                      if (!isnan(vistest))
			{
			  add_term_real += weight[iweight] * vis_real[ifine] * conv_weight;
			  add_term_img += weight[iweight] * vis_img[ifine] * conv_weight;
			}
                      ifine++;
		      iweight++;
		    }
		}
	      atomicAdd(&(grid[iKer]),add_term_real);
	      atomicAdd(&(grid[iKer+1]),add_term_img);
            }
        }
    }
}
#endif
#ifdef ACCOMP
#pragma  omp declare target
#endif
void wstack(
     int num_w_planes,
     myuint num_points,
     myuint freq_per_chan,
     myuint polarizations,
     double* uu,
     double* vv,
     double* ww,
     float* vis_real,
     float* vis_img,
     float* weight,
     double dx,
     double dw,
     int w_support,
     int grid_size_x,
     int grid_size_y,
     double* grid,
     int num_threads,
    #ifdef HIPCC
     int rank,
     hipStream_t stream_stacking
    #else
     int rank
    #endif
	    )
	    
{
    myuint i;
    //myuint index;
    myull visindex;

    // initialize the convolution kernel
    // gaussian:
    int KernelLen = (w_support-1)/2;
    int increaseprecision = 5; // this number must be odd: increaseprecison*w_support must be odd (w_support must be odd)
    double std = 1.0;
    double std22 = 1.0/(2.0*std*std);
    double norm = std22/PI;
    double * convkernel = (double*)malloc(increaseprecision*w_support*sizeof(*convkernel));

    #ifdef GAUSS
    makeGaussKernel(convkernel,w_support,increaseprecision,std22);
    #endif
    #ifdef KAISERBESSEL
    double overSamplingFactor = 1.0;
    int withSinc = 0;
    double alpha = 8.6;
    makeKaiserBesselKernel(convkernel, w_support, increaseprecision, alpha, overSamplingFactor, withSinc);
    #endif


    // Loop over visibilities.
// Switch between HIP and GPU versions
   #ifdef __HIPCC__
    // Define the HIP set up
    int Nth = NTHREADS;
    myuint Nbl = (myuint)(num_points/Nth) + 1;
    if(NWORKERS == 1) {Nbl = 1; Nth = 1;};
    myull Nvis = num_points*freq_per_chan*polarizations;

    int ndevices;
    int num = hipGetDeviceCount(&ndevices);
    int res = hipSetDevice(rank % ndevices);

    if ( rank == 0 ) {
      if (0 == ndevices) {

	shutdown_wstacking(NO_ACCELERATORS_FOUND, "No accelerators found", __FILE__, __LINE__ );
      }
    }

     #ifdef NVIDIA
      prtAccelInfo();
     #endif

    // Create GPU arrays and offload them
    double * uu_g;
    double * vv_g;
    double * ww_g;
    float * vis_real_g;
    float * vis_img_g;
    float * weight_g;
    //double * grid_g;
    double * convkernel_g;
    
    //Create the event inside stream stacking
    //hipEvent_t event_kernel;
    
    //for (int i=0; i<100000; i++)grid[i]=23.0;
    hipError_t mmm;
    //mmm=hipEventCreate(&event_kernel);
    mmm=hipMalloc(&uu_g,num_points*sizeof(double));
    mmm=hipMalloc(&vv_g,num_points*sizeof(double));
    mmm=hipMalloc(&ww_g,num_points*sizeof(double));
    mmm=hipMalloc(&vis_real_g,Nvis*sizeof(float));
    mmm=hipMalloc(&vis_img_g,Nvis*sizeof(float));
    mmm=hipMalloc(&weight_g,(Nvis/freq_per_chan)*sizeof(float));
    //mmm=hipMalloc(&grid_g,2*num_w_planes*grid_size_x*grid_size_y*sizeof(double));
    
   #if !defined(GAUSS_HI_PRECISION)
    mmm=hipMalloc(&convkernel_g,increaseprecision*w_support*sizeof(double));
   #endif
    
    if (mmm != hipSuccess) {printf("!!! w-stacking.cu hipMalloc ERROR %d !!!\n", mmm);}
    //mmm=hipMemset(grid_g,0.0,2*num_w_planes*grid_size_x*grid_size_y*sizeof(double));
    if (mmm != hipSuccess) {printf("!!! w-stacking.cu hipMemset ERROR %d !!!\n", mmm);}


    mmm=hipMemcpyAsync(uu_g, uu, num_points*sizeof(double), hipMemcpyHostToDevice, stream_stacking);
    mmm=hipMemcpyAsync(vv_g, vv, num_points*sizeof(double), hipMemcpyHostToDevice, stream_stacking);
    mmm=hipMemcpyAsync(ww_g, ww, num_points*sizeof(double), hipMemcpyHostToDevice, stream_stacking);
    mmm=hipMemcpyAsync(vis_real_g, vis_real, Nvis*sizeof(float), hipMemcpyHostToDevice, stream_stacking);
    mmm=hipMemcpyAsync(vis_img_g, vis_img, Nvis*sizeof(float), hipMemcpyHostToDevice, stream_stacking);
    mmm=hipMemcpyAsync(weight_g, weight, (Nvis/freq_per_chan)*sizeof(float), hipMemcpyHostToDevice, stream_stacking);

    
   #if !defined(GAUSS_HI_PRECISION)
    mmm=hipMemcpyAsync(convkernel_g, convkernel, increaseprecision*w_support*sizeof(double), hipMemcpyHostToDevice, stream_stacking);
   #endif
    
    if (mmm != hipSuccess) {printf("!!! w-stacking.cu hipMemcpyAsync ERROR %d !!!\n", mmm);}
    
    // Call main GPU Kernel
   #if defined(GAUSS_HI_PRECISION)
    convolve_g <<<Nbl,Nth,0,stream_stacking>>> (
	       num_w_planes,
               num_points,
               freq_per_chan,
               polarizations,
               uu_g,
               vv_g,
               ww_g,
               vis_real_g,
               vis_img_g,
               weight_g,
               dx,
               dw,
               KernelLen,
               grid_size_x,
               grid_size_y,
               grid,
	       std22
						);

   #else
    convolve_g <<<Nbl,Nth,0,stream_stacking>>> (
               num_w_planes,
               num_points,
               freq_per_chan,
               polarizations,
               uu_g,
               vv_g,
               ww_g,
               vis_real_g,
               vis_img_g,
               weight_g,
               dx,
               dw,
               KernelLen,
               grid_size_x,
               grid_size_y,
               grid,
               std22,
               convkernel_g
                                                );
   #endif
    
    mmm=hipStreamSynchronize(stream_stacking);
    //Record the event
    //mmm=hipEventRecord(event_kernel,stream_stacking);

    //Wait until the kernel ends
    //mmm=hipStreamWaitEvent(stream_stacking,event_kernel);
    
    //for (int i=0; i<100000; i++)printf("%f\n",grid[i]);

    if (mmm != hipSuccess)
      printf("HIP ERROR %s\n",hipGetErrorString(mmm));

    mmm=hipFree(uu_g);
    mmm=hipFree(vv_g);
    mmm=hipFree(ww_g);
    mmm=hipFree(vis_real_g);
    mmm=hipFree(vis_img_g);
    mmm=hipFree(weight_g);
    //mmm=hipFree(grid_g);
    
   #if !defined(GAUSS_HI_PRECISION)
    mmm=hipFree(convkernel_g);
   #endif
    
// Switch between HIP and GPU versions
# else

#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#endif

   #if defined(ACCOMP) && (GPU_STACKING)
    omp_set_default_device(rank % omp_get_num_devices());
    myull Nvis = num_points*freq_per_chan*polarizations;
   #pragma omp target teams distribute parallel for private(visindex) map(to:uu[0:num_points], vv[0:num_points], ww[0:num_points], vis_real[0:Nvis], vis_img[0:Nvis], weight[0:Nvis/freq_per_chan]) map(tofrom:grid[0:2*num_w_planes*grid_size_x*grid_size_y])
#else
    #pragma omp parallel for private(visindex)
#endif
    for (i = 0; i < num_points; i++)
    {
#ifdef _OPENMP
	//int tid;
	//tid = omp_get_thread_num();
	//printf("%d\n",tid);
#endif

        visindex = i*freq_per_chan*polarizations;

        double sum = 0.0;
        int j, k;
	//if (i%1000 == 0)printf("%ld\n",i);

        /* Convert UV coordinates to grid coordinates. */
        double pos_u = uu[i] / dx;
        double pos_v = vv[i] / dx;
        double ww_i  = ww[i] / dw;
	
        int grid_w = (int)ww_i;
        int grid_u = (int)pos_u;
        int grid_v = (int)pos_v;

	// check the boundaries
	myuint jmin = (grid_u > KernelLen - 1) ? grid_u - KernelLen : 0;
	myuint jmax = (grid_u < grid_size_x - KernelLen) ? grid_u + KernelLen : grid_size_x - 1;
	myuint kmin = (grid_v > KernelLen - 1) ? grid_v - KernelLen : 0;
	myuint kmax = (grid_v < grid_size_y - KernelLen) ? grid_v + KernelLen : grid_size_y - 1;
        //printf("%d, %ld, %ld, %d, %ld, %ld\n",grid_u,jmin,jmax,grid_v,kmin,kmax);


        // Convolve this point onto the grid.
        for (k = kmin; k <= kmax; k++)
        {

            //double v_dist = (double)k+0.5 - pos_v;
            double v_dist = (double)k - pos_v;

            for (j = jmin; j <= jmax; j++)
            {
                //double u_dist = (double)j+0.5 - pos_u;
                double u_dist = (double)j - pos_u;
		myuint iKer = 2 * (j + k*grid_size_x + grid_w*grid_size_x*grid_size_y);
                int jKer = (int)(increaseprecision * (fabs(u_dist+(double)KernelLen)));
                int kKer = (int)(increaseprecision * (fabs(v_dist+(double)KernelLen)));

                #ifdef GAUSS_HI_PRECISION
		double conv_weight = gauss_kernel_norm(norm,std22,u_dist,v_dist);
                #endif
                #ifdef GAUSS
		double conv_weight = convkernel[jKer]*convkernel[kKer];
		//if(jKer < 0 || jKer >= 35 || kKer < 0 || kKer >= 35)
		//	printf("%f %d %f %d\n",fabs(u_dist+(double)KernelLen),jKer,fabs(v_dist+(double)KernelLen),kKer);
		//printf("%d %d %d %d %f %f %f %f %f\n",jKer, j, kKer, k, pos_u, pos_v, u_dist,v_dist,conv_weight);
                #endif
                #ifdef KAISERBESSEL
		double conv_weight = convkernel[jKer]*convkernel[kKer];
	        #endif
		// Loops over frequencies and polarizations
		double add_term_real = 0.0;
		double add_term_img = 0.0;
		myull ifine = visindex;
		// DAV: the following two loops are performend by each thread separately: no problems of race conditions
		for (myuint ifreq=0; ifreq<freq_per_chan; ifreq++)
		{
		   myuint iweight = visindex/freq_per_chan;
	           for (myuint ipol=0; ipol<polarizations; ipol++)
	           {
                      if (!isnan(vis_real[ifine]))
                      {
		         //printf("%f %ld\n",weight[iweight],iweight);
                         add_term_real += weight[iweight] * vis_real[ifine] * conv_weight;
		         add_term_img += weight[iweight] * vis_img[ifine] * conv_weight;
			 //if(vis_img[ifine]>1e10 || vis_img[ifine]<-1e10)printf("%f %f %f %f %ld %ld\n",vis_real[ifine],vis_img[ifine],weight[iweight],conv_weight,ifine,num_points*freq_per_chan*polarizations);
		      }
		      ifine++;
		      iweight++;
		   }
	        }
		// DAV: this is the critical call in terms of correctness of the results and of performance
		#pragma omp atomic
		grid[iKer] += add_term_real;
		#pragma omp atomic
		grid[iKer+1] += add_term_img;
            }
        }
	
    }
   #if defined(ACCOMP) && (GPU_STACKING)
   #pragma omp target exit data map(delete:uu[0:num_points], vv[0:num_points], ww[0:num_points], vis_real[0:Nvis], vis_img[0:Nvis], weight[0:Nvis/freq_per_chan], grid[0:2*num_w_planes*grid_size_x*grid_size_y])
   #endif
    // End switch between HIP and CPU versions
#endif
    //for (int i=0; i<100000; i++)printf("%f\n",grid[i]);
}

#ifdef ACCOMP
#pragma  omp end declare target
#endif

int test(int nnn)
{
	int mmm;

	mmm = nnn+1;
	return mmm;
}
