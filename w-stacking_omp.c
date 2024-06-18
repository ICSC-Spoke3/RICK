#include <omp.h>
#include "w-stacking_omp.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef NVIDIA
#include <cuda_runtime.h>
#endif

/* WARNING */

#if defined (_OPENMP) && (ACCOMP)

#pragma omp  declare target
double gauss_kernel_norm(double norm, double std22, double u_dist, double v_dist)
{
     double conv_weight;
     conv_weight = norm * exp(-((u_dist*u_dist)+(v_dist*v_dist))*std22);
     return conv_weight;
}

#pragma omp end declare target


//The function has been slightly modified in order to include the ranks of the tasks calling the GPUs [GL]
void wstack(
     int num_w_planes,
     uint num_points,
     uint freq_per_chan,
     uint polarizations,
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
     int rank) 
{
    //uint index;
    uint visindex;

    // initialize the convolution kernel
    // gaussian:
    int KernelLen = (w_support-1)/2;
    double std = 1.0;
    double std22 = 1.0/(2.0*std*std);
    double norm = std22/PI;

    // Loop over visibilities.
    omp_set_num_threads(num_threads);

    uint Nvis = num_points*freq_per_chan*polarizations;
    uint gpu_weight_dim = Nvis/freq_per_chan;
    uint gpu_grid_dim = 2*num_w_planes*grid_size_x*grid_size_y;

    /*
    //Checking for GPU memory
    uint uint int data_mem = num_points * (sizeof(uu) + sizeof(vv) + sizeof(ww)) + Nvis * (sizeof(vis_real) +
											   sizeof(vis_img) ) +
      gpu_weight_dim * sizeof(weight);
    uint uint int grid_mem = gpu_grid_dim * sizeof(grid);
    printf("Rank %d, Original data: %lld, Grid array: %lld, Total: %lld\n", rank, data_mem, grid_mem, data_mem + grid_mem);fflush(stdout);
    */

#pragma omp target teams distribute parallel for private(visindex)	\
  map(to:uu[0:num_points], vv[0:num_points], ww[0:num_points], vis_real[0:Nvis], vis_img[0:Nvis], weight[0:gpu_weight_dim]) \
  map(tofrom: grid[0:gpu_grid_dim]) device(rank % omp_get_num_devices()) //Works also if Ntasks > NGPUs [GL]

    for (uint i = 0; i < num_points; i++)
    {

        visindex = i*freq_per_chan*polarizations;

        double sum = 0.0;
        int j, k;
	//if (i%1000 == 0)printf("%ld\n",i);

        /* Convert UV coordinates to grid coordinates. */
        double pos_u = uu[i] / dx;
        double pos_v = vv[i] / dx;
        double ww_i  = ww[i] / dw;
	
	//Renormalization of ww_i to avoid out of bounds
 	ww_i = ww_i/(1+num_w_planes);

	int grid_w = (int)ww_i;
        int grid_u = (int)pos_u;
        int grid_v = (int)pos_v;

	// check the boundaries
	uint jmin = (grid_u > KernelLen - 1) ? grid_u - KernelLen : 0;
	uint jmax = (grid_u < grid_size_x - KernelLen) ? grid_u + KernelLen : grid_size_x - 1;
	uint kmin = (grid_v > KernelLen - 1) ? grid_v - KernelLen : 0;
	uint kmax = (grid_v < grid_size_y - KernelLen) ? grid_v + KernelLen : grid_size_y - 1;
        //printf("%d, %ld, %ld, %d, %ld, %ld\n",grid_u,jmin,jmax,grid_v,kmin,kmax);


        // Convolve this point onto the grid.
        for (k = kmin; k <= kmax; k++)
        {

            double v_dist = (double)k+0.5 - pos_v;

            for (j = jmin; j <= jmax; j++)
            {
                double u_dist = (double)j+0.5 - pos_u;
		uint iKer = 2 * (j + k*grid_size_x + grid_w*grid_size_x*grid_size_y);

		double conv_weight = gauss_kernel_norm(norm,std22,u_dist,v_dist);
		// Loops over frequencies and polarizations
		double add_term_real = 0.0;
		double add_term_img = 0.0;
		uint ifine = visindex;
		// DAV: the following two loops are performend by each thread separately: no problems of race conditions
		for (uint ifreq=0; ifreq<freq_per_chan; ifreq++)
		{
		   uint iweight = visindex/freq_per_chan;
	           for (uint ipol=0; ipol<polarizations; ipol++)
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

   #pragma omp target exit data map(delete:uu[0:num_points], vv[0:num_points], ww[0:num_points], vis_real[0:Nvis], vis_img[0:Nvis], weight[0:gpu_weight_dim],grid[0:gpu_grid_dim]) device(rank % omp_get_num_devices())

}


#ifdef NVIDIA
#define CUDAErrorCheck(funcall)                                         \
do {                                                                    \
  cudaError_t ierr = funcall;                                           \
  if (cudaSuccess != ierr) {                                            \
    fprintf(stderr, "%s(line %d) : CUDA RT API error : %s(%d) -> %s\n", \
    __FILE__, __LINE__, #funcall, ierr, cudaGetErrorString(ierr));      \
    exit(ierr);                                                         \
  }                                                                     \
} while (0)

static inline int _corePerSM(int major, int minor)
/**
 * @brief Give the number of CUDA cores per streaming multiprocessor (SM).
 *
 * The number of CUDA cores per SM is determined by the compute capability.
 *
 * @param major Major revision number of the compute capability.
 * @param minor Minor revision number of the compute capability.
 *
 * @return The number of CUDA cores per SM.
 */
{
  if (1 == major) {
    if (0 == minor || 1 == minor || 2 == minor || 3 == minor) return 8;
  }
  if (2 == major) {
    if (0 == minor) return 32;
    if (1 == minor) return 48;
  }
  if (3 == major) {
    if (0 == minor || 5 == minor || 7 == minor) return 192;
  }
  if (5 == major) {
    if (0 == minor || 2 == minor) return 128;
  }
  if (6 == major) {
    if (0 == minor) return 64;
    if (1 == minor || 2 == minor) return 128;
  }
  if (7 == major) {
    if (0 == minor || 2 == minor || 5 == minor) return 64;
  }
  return -1;
}

void getGPUInfo(int iaccel)
{
  int corePerSM;

 struct cudaDeviceProp dev;

  CUDAErrorCheck(cudaSetDevice(iaccel));
  CUDAErrorCheck(cudaGetDeviceProperties(&dev, iaccel));
  corePerSM = _corePerSM(dev.major, dev.minor);

  printf("\n");
  printf("============================================================\n");
  printf("CUDA Device name : \"%s\"\n", dev.name);
  printf("------------------------------------------------------------\n");
  printf("Comp. Capability : %d.%d\n", dev.major, dev.minor);
  printf("max clock rate   : %.0f MHz\n", dev.clockRate * 1.e-3f);
  printf("number of SMs    : %d\n", dev.multiProcessorCount);
  printf("cores  /  SM     : %d\n", corePerSM);
  printf("# of CUDA cores  : %d\n", corePerSM * dev.multiProcessorCount);
  printf("------------------------------------------------------------\n");
  printf("global memory    : %5.0f MBytes\n", dev.totalGlobalMem / 1048576.0f);
  printf("shared mem. / SM : %5.1f KBytes\n", dev.sharedMemPerMultiprocessor / 1024.0f);
  printf("32-bit reg. / SM : %d\n", dev.regsPerMultiprocessor);
  printf("------------------------------------------------------------\n");
  printf("max # of threads / SM    : %d\n", dev.maxThreadsPerMultiProcessor);
  printf("max # of threads / block : %d\n", dev.maxThreadsPerBlock);
  printf("max dim. of block        : (%d, %d, %d)\n",
      dev.maxThreadsDim[0], dev.maxThreadsDim[1], dev.maxThreadsDim[2]);
  printf("max dim. of grid         : (%d, %d, %d)\n",
      dev.maxGridSize[0],   dev.maxGridSize[1],   dev.maxGridSize[2]);
  printf("warp size                : %d\n", dev.warpSize);
  printf("============================================================\n");

  int z = 0, x = 2;
  #pragma omp target map(to:x) map(tofrom:z)
  {
    z=x+100;
  }
}

#endif  // closes ifdef(NVIDIA)

#endif  // closes initial if defined clause

