#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
#include "ricklib.h"
// #include <omp.h>

#define PI 3.14159265359
#define NTHREADS 32
#define NWORKERS -1
#define FILENAMELENGTH 30

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#define NOT_ENOUGH_MEM_STACKING 3

int *histo_send;
int **sectorarray;

#if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
float *visreal_stokes_ch;
float *visimg_stokes_ch;
float *weights_stokes_ch;
#else
float *visreal_ch;
float *visimg_ch;
float *weights_ch;
#endif

// Convolution kernels

// Gaussian Kernel
double gauss_kernel_norm(double norm, double std22, double u_dist, double v_dist)
{
  double conv_weight;
  conv_weight = norm * exp(-((u_dist * u_dist) + (v_dist * v_dist)) * std22);
  return conv_weight;
}

void makeGaussKernel(double *kernel,
                     int KernelLen,
                     int increaseprecision,
                     double std22)
{

  double norm = std22 / PI;
  int n = increaseprecision * KernelLen, mid = n / 2;
  for (int i = 0; i != mid + 1; i++)
  {
    double term = (double)i / (double)increaseprecision;
    kernel[mid + i] = sqrt(norm) * exp(-(term * term) * std22);
  }

  for (int i = 0; i != mid; i++)
    kernel[i] = kernel[n - 1 - i];
}

// Kaiser-Bessel Kernel: it is adapted from WSClean
double bessel0(double x, double precision)
{
  // Calculate I_0 = SUM of m 0 -> inf [ (x/2)^(2m) ]
  // This is the unnormalized bessel function of order 0.
  double d = 0.0, ds = 1.0, sum = 1.0;
  do
  {
    d += 2.0;
    ds *= x * x / (d * d);
    sum += ds;
  } while (ds > sum * precision);
  return sum;
}

void makeKaiserBesselKernel(double *kernel,
                            int KernelLen,
                            int increaseprecision,
                            double alpha,
                            double overSamplingFactor,
                            int withSinc)
{
  int n = increaseprecision * KernelLen, mid = n / 2;
  double *sincKernel = (double *)malloc((mid + 1) * sizeof(*sincKernel));
  const double filterRatio = 1.0 / overSamplingFactor;
  sincKernel[0] = filterRatio;
  for (int i = 1; i != mid + 1; i++)
  {
    double x = i;
    sincKernel[i] =
        withSinc ? (sin(PI * filterRatio * x) / (PI * x)) : filterRatio;
  }
  const double normFactor = overSamplingFactor / bessel0(alpha, 1e-8);
  for (int i = 0; i != mid + 1; i++)
  {
    double term = (double)i / mid;
    kernel[mid + i] = sincKernel[i] *
                      bessel0(alpha * sqrt(1.0 - (term * term)), 1e-8) *
                      normFactor;
  }
  for (int i = 0; i != mid; i++)
    kernel[i] = kernel[n - 1 - i];
}

/*

Weighting functions

*/

void weighting_uniform(
    unsigned int iKer,
    unsigned int visindex,
    float *weight,
    float *weight_uv)
{
  weight_uv[iKer] += weight[visindex]; // dentro una stessa cella uv
}

void weighting_briggs(
    unsigned int iKer,
    unsigned int visindex,
    float *weight,
    float *weight_uv,
    float *weight_uv_2)
{
  weight_uv[iKer] += weight[visindex]; // dentro una stessa cella uv
  weight_uv_2[iKer] += (weight[visindex] * weight[visindex]);
}

void channelselect(
    unsigned int Nmeasures,
    int freq_per_chan,
    #if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
    float *visreal_stokes,
    float *visimg_stokes,
    float *weights_stokes,
    #else
    float *visreal,
    float *visimg,
    float *weights,
    #endif
    int freq_index)
{
  #if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
  visreal_stokes_ch = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  visimg_stokes_ch = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  weights_stokes_ch = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  #else
  visreal_ch = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  visimg_ch = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  weights_ch = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  #endif

  for (unsigned int ichan = 0; ichan < Nmeasures; ichan++)
  {
    #if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
    visreal_stokes_ch[ichan] = visreal_stokes[ichan * freq_per_chan + freq_index];
    visimg_stokes_ch[ichan] = visimg_stokes[ichan * freq_per_chan + freq_index];
    weights_stokes_ch[ichan] = weights_stokes[ichan * freq_per_chan + freq_index];
    #else
    visreal_ch[ichan] = visreal[ichan * freq_per_chan + freq_index];
    visimg_ch[ichan] = visimg[ichan * freq_per_chan + freq_index];
    weights_ch[ichan] = weights[ichan * freq_per_chan + freq_index];
    #endif
  }

  #if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
  free(visreal_stokes);
  free(visimg_stokes);
  free(weights_stokes);
  #else
  free(visreal);
  free(visimg);
  free(weights);
  #endif
}

#ifdef RICK_GPU

__global__ void convolve_g(
    int num_w_planes,
    int num_points,
    int freq_per_chan,
    int polarizations,
    double *uu,
    double *vv,
    double *ww,
    float *vis_real,
    float *vis_img,
    float *weight,
    double dx,
    double dw,
    int KernelLen,
    int grid_size_x,
    int grid_size_y,
    double *grid,
#if defined(GAUSS_HI_PRECISION)
    double std22
#else
    double std22,
    double *convkernel
#endif
)

{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < num_points)
  {
    int i = gid;
    long visindex = i * freq_per_chan * polarizations;
    double norm = std22 / PI;

    int j, k;

    /* Convert UV coordinates to grid coordinates. */
    double pos_u = uu[i] / dx;
    double pos_v = vv[i] / dx;
    double ww_i = ww[i] / dw;

    int grid_w = (int)ww_i;
    int grid_u = (int)pos_u;
    int grid_v = (int)pos_v;

    // check the boundaries
    int jmin = (grid_u > KernelLen - 1) ? grid_u - KernelLen : 0;
    int jmax = (grid_u < grid_size_x - KernelLen) ? grid_u + KernelLen : grid_size_x - 1;
    int kmin = (grid_v > KernelLen - 1) ? grid_v - KernelLen : 0;
    int kmax = (grid_v < grid_size_y - KernelLen) ? grid_v + KernelLen : grid_size_y - 1;

    // Convolve this point onto the grid.
    for (k = kmin; k <= kmax; k++)
    {

      double v_dist = (double)k + 0.5 - pos_v;
      int increaseprecision = 5;

      for (j = jmin; j <= jmax; j++)
      {
        double u_dist = (double)j + 0.5 - pos_u;
        int iKer = 2 * (j + k * grid_size_x + grid_w * grid_size_x * grid_size_y);
        int jKer = (int)(increaseprecision * (fabs(u_dist + (double)KernelLen)));
        int kKer = (int)(increaseprecision * (fabs(v_dist + (double)KernelLen)));

#ifdef GAUSS_HI_PRECISION
        double conv_weight = gauss_kernel_norm(norm, std22, u_dist, v_dist);
#endif
#ifdef GAUSS
        double conv_weight = convkernel[jKer] * convkernel[kKer];
#endif
#ifdef KAISERBESSEL
        double conv_weight = convkernel[jKer] * convkernel[kKer];
#endif

        // Loops over frequencies and polarizations
        double add_term_real = 0.0;
        double add_term_img = 0.0;
        long ifine = visindex;
        for (int ifreq = 0; ifreq < freq_per_chan; ifreq++)
        {
          int iweight = visindex / freq_per_chan;
          for (int ipol = 0; ipol < polarizations; ipol++)
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
        atomicAdd(&(grid[iKer]), add_term_real);
        atomicAdd(&(grid[iKer + 1]), add_term_img);
      }
    }
  }
}
#endif

void initialize_array(
    int nsectors,
    int nmeasures,
    double w_supporth,
    double *vv,
    int yaxis,
    double dx)
{

  histo_send = (int *)calloc(nsectors + 1, sizeof(int));

  for (int iphi = 0; iphi < nmeasures; iphi++)
  {
    double vvh = vv[iphi];
    int binphi = (int)(vvh * nsectors);

    double updist = (double)((binphi + 1) * yaxis) * dx - vvh;
    double downdist = vvh - (double)(binphi * yaxis) * dx;

    (histo_send)[binphi]++;
    if (updist < w_supporth && updist >= 0.0)
      (histo_send)[binphi + 1]++;

    if (downdist < w_supporth && binphi > 0 && downdist >= 0.0)
      (histo_send)[binphi - 1]++;
  }

  sectorarray = (int **)malloc((nsectors + 1) * sizeof(int *));
  if (sectorarray == NULL)
  {
    fprintf(stderr, "Error allocating memory for sectorarray\n");
    exit(EXIT_FAILURE);
  }

  int *counter = (int *)calloc((nsectors + 1), sizeof(int));
  if (counter == NULL)
  {
    fprintf(stderr, "Error allocating memory for counter\n");
    exit(EXIT_FAILURE);
  }

  for (int sec = 0; sec < (nsectors + 1); sec++)
  {
    sectorarray[sec] = (int *)calloc((histo_send)[sec], sizeof(int));
  }

  for (int iphi = 0; iphi < nmeasures; iphi++)
  {
    double vvh = vv[iphi];
    int binphi = (int)(vvh * nsectors);
    double updist = (double)((binphi + 1) * yaxis) * dx - vvh;
    double downdist = vvh - (double)(binphi * yaxis) * dx;
    (sectorarray)[binphi][counter[binphi]] = iphi;
    counter[binphi]++;

    if (updist < w_supporth && updist >= 0.0)
    {
      (sectorarray)[binphi + 1][counter[binphi + 1]] = iphi;
      counter[binphi + 1]++;
    };
    if (downdist < w_supporth && binphi > 0 && downdist >= 0.0)
    {
      (sectorarray)[binphi - 1][counter[binphi - 1]] = iphi;
      counter[binphi - 1]++;
    };
  }

  free(counter);

#ifdef VERBOSE
  for (int iii = 0; iii < nsectors + 1; iii++)
    printf("HISTO %d %d %ld\n", rank, iii, histo_send[iii]);
#endif
}

void wstack(
    int num_w_planes,
    int num_points,
    int freq_per_chan,
    int polarizations,
    double *uu,
    double *vv,
    double *ww,
    float *vis_real,
    float *vis_img,
    float *weight,
    double dx,
    double dw,
    int w_support,
    int grid_size_x,
    int grid_size_y,
    double *grid,
    int num_threads,
    int rank,
    int size)
{
  int i;
  // int index;
  long visindex;

  // Initialize the convolution kernel
  // For simplicity, we use for the moment only the Gaussian kernel:
  int KernelLen = (w_support - 1) / 2;
  int increaseprecision = 5; // this number must be odd: increaseprecison*w_support must be odd (w_support must be odd)
  double std = 1.0;
  double std22 = 1.0 / (2.0 * std * std);
  double norm = std22 / PI;
  double *convkernel = (double *)malloc(increaseprecision * w_support * sizeof(*convkernel));

#ifdef GAUSS
  makeGaussKernel(convkernel, w_support, increaseprecision, std22);
#endif
#ifdef KAISERBESSEL
  double overSamplingFactor = 1.0;
  int withSinc = 0;
  double alpha = 8.6;
  makeKaiserBesselKernel(convkernel, w_support, increaseprecision, alpha, overSamplingFactor, withSinc);
#endif

#ifdef RICK_GPU
  int Nth = NTHREADS;
  int Nbl = (int)(num_points / Nth) + 1;
  if (NWORKERS == 1)
  {
    Nbl = 1;
    Nth = 1;
  };
  long Nvis = num_points * freq_per_chan;

  int ndevices;
  cudaGetDeviceCount(&ndevices);
  cudaSetDevice(rank % ndevices);

  // Create GPU arrays and offload them
  double *uu_g;
  double *vv_g;
  double *ww_g;
  float *vis_real_g;
  float *vis_img_g;
  float *weight_g;
  double *convkernel_g;
#if !defined(NCCL_REDUCE)
  double *grid_g;
#endif
#if !defined(NCCL_REDUCE)
  cudaStream_t stream_stacking;
  cudaStreamCreate(&stream_stacking);
#endif

  // Create the event inside stream stacking
  // cudaEvent_t event_kernel;

  // for (int i=0; i<100000; i++)grid[i]=23.0;
  cudaError_t mmm;
  // mmm=cudaEventCreate(&event_kernel);
  mmm = cudaMalloc(&uu_g, num_points * sizeof(double));
  mmm = cudaMalloc(&vv_g, num_points * sizeof(double));
  mmm = cudaMalloc(&ww_g, num_points * sizeof(double));
  mmm = cudaMalloc(&vis_real_g, Nvis * sizeof(float));
  mmm = cudaMalloc(&vis_img_g, Nvis * sizeof(float));
  mmm = cudaMalloc(&weight_g, (Nvis / freq_per_chan) * sizeof(float));
  // mmm=cudaMalloc(&grid_g,2*num_w_planes*grid_size_x*grid_size_y*sizeof(double));

#if !defined(NCCL_REDUCE)
  mmm = cudaMalloc(&grid_g, 2 * num_w_planes * grid_size_x * grid_size_y * sizeof(double));
#endif

#if !defined(GAUSS_HI_PRECISION)
  mmm = cudaMalloc(&convkernel_g, increaseprecision * w_support * sizeof(double));
#endif
  if (mmm != cudaSuccess)
  {
    printf("!!! w-stacking.cu cudaMalloc ERROR %d !!!\n", mmm);
  }

#if !defined(NCCL_REDUCE)
  mmm = cudaMemset(grid_g, 0.0, 2 * num_w_planes * grid_size_x * grid_size_y * sizeof(double));
  if (mmm != cudaSuccess)
  {
    printf("!!! w-stacking.cu cudaMemset ERROR %d !!!\n", mmm);
  }
#endif

  mmm = cudaMemcpyAsync(uu_g, uu, num_points * sizeof(double), cudaMemcpyHostToDevice, stream_stacking);
  mmm = cudaMemcpyAsync(vv_g, vv, num_points * sizeof(double), cudaMemcpyHostToDevice, stream_stacking);
  mmm = cudaMemcpyAsync(ww_g, ww, num_points * sizeof(double), cudaMemcpyHostToDevice, stream_stacking);
  mmm = cudaMemcpyAsync(vis_real_g, vis_real, Nvis * sizeof(float), cudaMemcpyHostToDevice, stream_stacking);
  mmm = cudaMemcpyAsync(vis_img_g, vis_img, Nvis * sizeof(float), cudaMemcpyHostToDevice, stream_stacking);
  mmm = cudaMemcpyAsync(weight_g, weight, (Nvis / freq_per_chan) * sizeof(float), cudaMemcpyHostToDevice, stream_stacking);

#if !defined(GAUSS_HI_PRECISION)
  mmm = cudaMemcpyAsync(convkernel_g, convkernel, increaseprecision * w_support * sizeof(double), cudaMemcpyHostToDevice, stream_stacking);
#endif

  if (mmm != cudaSuccess)
  {
    printf("!!! w-stacking.cu cudaMemcpyAsync ERROR %d !!!\n", mmm);
  }

  // Call main GPU Kernel
#if defined(GAUSS_HI_PRECISION)
  convolve_g<<<Nbl, Nth, 0, stream_stacking>>>(
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
#if !defined(NCCL_REDUCE)
      grid_g,
#else
      grid,
#endif
      std22);
#else
  convolve_g<<<Nbl, Nth, 0, stream_stacking>>>(
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
#if !defined(NCCL_REDUCE)
      grid_g,
#else
      grid,
#endif
      std22,
      convkernel_g);
#endif

  mmm = cudaStreamSynchronize(stream_stacking);
  // Record the event
  // mmm=cudaEventRecord(event_kernel,stream_stacking);

  // Wait until the kernel ends
  // mmm=cudaStreamWaitEvent(stream_stacking,event_kernel);

  // for (int i=0; i<100000; i++)printf("%f\n",grid[i]);

#if !defined(NCCL_REDUCE)
  mmm = cudaMemcpy(grid, grid_g, 2 * num_w_planes * grid_size_x * grid_size_y * sizeof(double), cudaMemcpyDeviceToHost);
#endif

  if (mmm != cudaSuccess)
    printf("CUDA ERROR %s\n", cudaGetErrorString(mmm));

  mmm = cudaFree(uu_g);
  mmm = cudaFree(vv_g);
  mmm = cudaFree(ww_g);
  mmm = cudaFree(vis_real_g);
  mmm = cudaFree(vis_img_g);
  mmm = cudaFree(weight_g);

#if !defined(NCCL_REDUCE)
  mmm = cudaFree(grid_g);
#endif

#if !defined(GAUSS_HI_PRECISION)
  mmm = cudaFree(convkernel_g);
#endif

#else // switch between CPU and GPU gridding

#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif

#if defined(ACCOMP) && (GPU_STACKING)
  omp_set_default_device(rank % omp_get_num_devices());
  long Nvis = num_points;
  printf("Nvis\n");
#pragma omp target teams distribute parallel for private(visindex) map(to : uu[0 : num_points], vv[0 : num_points], ww[0 : num_points], vis_real[0 : Nvis], vis_img[0 : Nvis], weight[0 : Nvis / freq_per_chan]) map(tofrom : grid[0 : 2 * num_w_planes * grid_size_x * grid_size_y])
#else
#pragma omp parallel for private(visindex)
#endif

#if defined(WEIGHTING_BRIGGS)
  float robust = 2.0;
#endif

#if defined(WEIGHTING_UNIFORM) || defined(WEIGHTING_BRIGGS)
  float *robustness;
  float *out_weight_uniform;
  float *weight_uv;
  float *weight_uv_2;

  out_weight_uniform = (float *)malloc(num_points * sizeof(float));
  weight_uv = (float *)malloc(grid_size_x * grid_size_y * num_w_planes * sizeof(float));
  weight_uv_2 = (float *)malloc(grid_size_x * grid_size_y * num_w_planes * sizeof(float));
  robustness = (float *)malloc(grid_size_x * grid_size_y * num_w_planes * size * sizeof(float));

  for (i = 0; i < num_points; i++)
  {
    visindex = i;

    double sum = 0.0;
    int j, k;

    // Convert UV coordinates to grid coordinates.
    double pos_u = uu[i] / dx;
    double pos_v = vv[i] / dx;
    double ww_i = ww[i] / dw;

    int grid_w = (int)ww_i;
    int grid_u = (int)pos_u;
    int grid_v = (int)pos_v;

    // check the boundaries
    unsigned int jmin = (grid_u > KernelLen - 1) ? grid_u - KernelLen : 0;
    unsigned int jmax = (grid_u < grid_size_x - KernelLen) ? grid_u + KernelLen : grid_size_x - 1;
    unsigned int kmin = (grid_v > KernelLen - 1) ? grid_v - KernelLen : 0;
    unsigned int kmax = (grid_v < grid_size_y - KernelLen) ? grid_v + KernelLen : grid_size_y - 1;
    // printf("%ld, %ld, %ld, %ld\n",jmin,jmax,kmin,kmax);

    // Convolve this point onto the grid.
    for (k = kmin; k <= kmax; k++)
    {
      for (j = jmin; j <= jmax; j++)
      {
        unsigned int iKer = 2 * (j + k * grid_size_x + grid_w * grid_size_x * grid_size_y);
#if defined(WEIGHTING_UNIFORM)
        weighting_uniform(iKer, visindex, weight, weight_uv);
        // To be done!!!!!!!!!!
        // float weights_stokes_sum = 0.0;
        // for (unsigned int i = 0; i < (Nmeasures * freq_per_chan); i++)
        //{
        //  weights_stokes_sum += weight_uv[i];
        //}
#elif defined(WEIGHTING_BRIGGS)
        weighting_briggs(iKer, visindex, weight, weight_uv, weight_uv_2);
#endif
      }
    }
  }
  // printf("Sum of all Stokes weights %f\n", weights_stokesI_sum);

#endif

  for (i = 0; i < num_points; i++)
  {
#ifdef _OPENMP
    // int tid;
    // tid = omp_get_thread_num();
    // printf("%d\n",tid);
#endif

    visindex = i;

    double sum = 0.0;
    int j, k;

    /* Convert UV coordinates to grid coordinates. */
    double pos_u = uu[i] / dx;
    double pos_v = vv[i] / dx;
    double ww_i = ww[i] / dw;

    int grid_w = (int)ww_i;
    int grid_u = (int)pos_u;
    int grid_v = (int)pos_v;

    // check the boundaries
    int jmin = (grid_u > KernelLen - 1) ? grid_u - KernelLen : 0;
    int jmax = (grid_u < grid_size_x - KernelLen) ? grid_u + KernelLen : grid_size_x - 1;
    int kmin = (grid_v > KernelLen - 1) ? grid_v - KernelLen : 0;
    int kmax = (grid_v < grid_size_y - KernelLen) ? grid_v + KernelLen : grid_size_y - 1;

    // Convolve this point onto the grid.
    for (k = kmin; k <= kmax; k++)
    {
      double v_dist = (double)k + 0.5 - pos_v;

      for (j = jmin; j <= jmax; j++)
      {
        double u_dist = (double)j + 0.5 - pos_u;
        int iKer = 2 * (j + k * grid_size_x + grid_w * grid_size_x * grid_size_y);
        int jKer = (int)(increaseprecision * (fabs(u_dist + (double)KernelLen)));
        int kKer = (int)(increaseprecision * (fabs(v_dist + (double)KernelLen)));

#if defined(WEIGHTING_UNIFORM)
        if (weight_uv[iKer] != 0.0)
        {
          out_weight_uniform[visindex] = 1.0 / weight_uv[iKer];
        }
        else
        {
          out_weight_uniform[visindex] = 0.0;
        }
#elif defined(WEIGHTING_BRIGGS)
        if (weight_uv[iKer] != 0.0)
        {
          robustness[iKer] = (pow((5.0 * (1.0 / (pow(10.0, robust)))), 2)) / (weight_uv_2[iKer] / weight_uv[iKer]);
          out_weight_uniform[visindex] = weight[visindex] / (1 + robustness[iKer] * weight_uv[iKer]);
        }
        else
        {
          out_weight_uniform[visindex] = 0.0;
        }
#endif

#ifdef GAUSS_HI_PRECISION
        double conv_weight = gauss_kernel_norm(norm, std22, u_dist, v_dist);
#endif
#ifdef GAUSS
        double conv_weight = convkernel[jKer] * convkernel[kKer];
        // printf("convkernel Gaussian = %f\n", convkernel[jKer]);
#endif
#ifdef KAISERBESSEL
        double conv_weight = convkernel[jKer] * convkernel[kKer];
#endif
        // Loops over frequencies and polarizations
        double add_term_real = 0.0;
        double add_term_img = 0.0;
        unsigned int ifine = visindex;
        // DAV: the following two loops are performend by each thread separately: no problems of race conditions
        for (int ifreq = 0; ifreq < 1; ifreq++)
        {
          // int iweight = visindex / freq_per_chan;
          // for (int ipol = 0; ipol < polarizations; ipol++)
          //{
          if (!isnan(vis_real[ifine]))
          {
#if defined(WEIGHTING_UNIFORM) || defined(WEIGHTING_BRIGGS)
            add_term_real += out_weight_uniform[ifine] * vis_real[ifine] * conv_weight;
            add_term_img += out_weight_uniform[ifine] * vis_img[ifine] * conv_weight;
#else
            add_term_real += weight[ifine] * vis_real[ifine] * conv_weight;
            add_term_img += weight[ifine] * vis_img[ifine] * conv_weight;
#endif
          }
          ifine++;
          // iweight++;
          //}
        }
        // DAV: this is the critical call in terms of correctness of the results and of performance
#pragma omp atomic
        grid[iKer] += add_term_real;
#pragma omp atomic
        grid[iKer + 1] += add_term_img;
      }
    }
  }
#if defined(ACCOMP) && (GPU_STACKING)
#pragma omp target exit data map(delete : uu[0 : num_points], vv[0 : num_points], ww[0 : num_points], vis_real[0 : Nvis], vis_img[0 : Nvis], weight[0 : Nvis / freq_per_chan], grid[0 : 2 * num_w_planes * grid_size_x * grid_size_y])
#endif

#if defined(WEIGHTING_UNIFORM)
  free(weight_uv);
  free(out_weight_uniform);
#elif defined(WEIGHTING_BRIGGS)
  free(weight_uv);
  free(weight_uv_2);
  free(out_weight_uniform);
  free(robustness);
#endif

#endif
}

void free_array(int *histo_send, int **sectorarrays, int nsectors)

{

  for (int i = nsectors - 1; i > 0; i--)
    free(sectorarrays[i]);

  free(sectorarrays);

  free(histo_send);

  return;
}

void gridding_data(
    double_t dx,
    double_t dw,
    int num_threads,
    int size,
    int rank,
    int xaxis,
    int yaxis,
    int grid_size_x,
    int grid_size_y,
    int size_of_grid,
    int num_w_planes,
    int w_support,
    double uvmin,
    double uvmax,
    int polarisations,
    int freq_per_chan,
    double *uu,
    double *vv,
    double *ww,
    double *grid,
    double *gridss,
    float *visreal,
    float *visimg,
    float *weights,
#if defined(WRITE_DATA)
    char *gridded_writedata1,
    char *gridded_writedata2,
#endif
    MPI_Comm MYMYMPI_COMM)
{

  double shift = (double)(dx * yaxis);

  // calculate the resolution in radians
  double resolution = 1.0 / MAX(fabs(uvmin), fabs(uvmax));

  // calculate the resolution in arcsec
  double resolution_asec = (3600.0 * 180.0) / MAX(fabs(uvmin), fabs(uvmax)) / PI;
  if (rank == 0)
    printf("Theoretical beam = %f rad, %f arcsec\n", resolution, resolution_asec);

  for (int isector = 0; isector < size; isector++)
  {
    // double start = CPU_TIME_wt;

    int Nsec = histo_send[isector];
    long Nweightss = Nsec;
    long Nvissec = Nsec;

    // EDR: this should probably be updated
    double_t *memory = (double *)malloc((Nsec * 3) * sizeof(double_t) +
                                        (Nvissec * 2 + Nweightss) * sizeof(float_t));

    double_t *uus = (double_t *)memory;
    double_t *vvs = (double_t *)uus + Nsec;
    double_t *wws = (double_t *)vvs + Nsec;
    float_t *weightss = (float_t *)((double_t *)wws + Nsec);
    float_t *visreals = (float_t *)weightss + Nweightss;
    float_t *visimgs = (float_t *)visreals + Nvissec;

    // select data for this sector
    int icount = 0;
    // int ip = 0;
    int inu = 0;

#warning "this loop should be threaded"
#warning "the counter of this loop should not be int"
    for (int iphi = histo_send[isector] - 1; iphi >= 0; iphi--)
    {

      int ilocal = sectorarray[isector][iphi];

      uus[icount] = uu[ilocal];
      vvs[icount] = vv[ilocal] - isector * shift;
      wws[icount] = ww[ilocal];
      /*
      for (int ipol = 0; ipol < polarisations; ipol++)
      {
        weightss[ip] = weights[ilocal * polarisations + ipol];
        ip++;
      }
      */
      for (int ifreq = 0; ifreq < 1; ifreq++)
      {
        visreals[inu] = visreal[ilocal * 1 + ifreq];
        visimgs[inu] = visimg[ilocal * 1 + ifreq];
        weightss[inu] = weights[ilocal * 1 + ifreq];
        inu++;
      }
      icount++;
    }

    double uumin = 1e20;
    double vvmin = 1e20;
    double uumax = -1e20;
    double vvmax = -1e20;

#pragma omp parallel reduction(min : uumin, vvmin) reduction(max : uumax, vvmax) num_threads(num_threads)
    {
      double my_uumin = 1e20;
      double my_vvmin = 1e20;
      double my_uumax = -1e20;
      double my_vvmax = -1e20;

#pragma omp for
      for (int ipart = 0; ipart < Nsec; ipart++)
      {
        my_uumin = MIN(my_uumin, uus[ipart]);
        my_uumax = MAX(my_uumax, uus[ipart]);
        my_vvmin = MIN(my_vvmin, vvs[ipart]);
        my_vvmax = MAX(my_vvmax, vvs[ipart]);
      }

      uumin = MIN(uumin, my_uumin);
      uumax = MAX(uumax, my_uumax);
      vvmin = MIN(vvmin, my_vvmin);
      vvmax = MAX(vvmax, my_vvmax);
    }

    // Make convolution on the grid

#ifdef VERBOSE
    printf("Processing sector %ld\n", isector);
#endif

    double *stacking_target_array;
    if (size > 1)
      stacking_target_array = gridss;
    else
      stacking_target_array = grid;

    // We have to call different GPUs per MPI task!!! [GL]
    // printf("Inside wstack() \n");
    wstack(num_w_planes,
           Nsec,
           freq_per_chan,
           polarisations,
           uus,
           vvs,
           wws,
           visreals,
           visimgs,
           weightss,
           dx,
           dw,
           w_support,
           xaxis,
           yaxis,
           stacking_target_array,
           num_threads,
           rank,
           size);

#ifdef VERBOSE
    printf("Processed sector %ld\n", isector);
#endif

    if (size > 1)
    {
      // Write grid in the corresponding remote slab

      int target_rank = (int)(isector % size);

      // Force to use MPI_Reduce when -fopenmp is not active
#ifdef _OPENMP
      // if (reduce_method == REDUCE_MPI)

      MPI_Reduce(gridss, grid, size_of_grid, MPI_DOUBLE, MPI_SUM, target_rank, MYMYMPI_COMM);
      /*
          else if (reduce_method == REDUCE_RING)
          {

            int ret = reduce_ring(target_rank);
            // grid    = (double*)Me.fwin.ptr; //Let grid point to the right memory location [GL]

            if (ret != 0)
            {
              char message[1000];
              sprintf(message, "Some problem occurred in the ring reduce "
                               "while processing sector %d",
                      isector);
              free(memory);
              shutdown_wstacking(ERR_REDUCE, message, __FILE__, __LINE__);
            }
          }
      */
#else
      MPI_Reduce(gridss, grid, size_of_grid, MPI_DOUBLE, MPI_SUM, target_rank, MYMYMPI_COMM);

#endif

      // Go to next sector
      memset(gridss, 0, 2 * num_w_planes * xaxis * yaxis * sizeof(double));
    }

    free(memory);
  }

  if (size > 1)
  {
    MPI_Barrier(MYMYMPI_COMM);
  }

#ifdef WRITE_DATA

  if (rank == 0)
  {
    printf("WRITING GRIDDED DATA\n");
  }

  MPI_File pFilereal;
  MPI_File pFileimg;

  double *gridss_real = (double *)malloc(size_of_grid / 2 * sizeof(double));
  double *gridss_img = (double *)malloc(size_of_grid / 2 * sizeof(double));

  MPI_File_open(MYMYMPI_COMM, gridded_writedata1, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &pFilereal);
  MPI_File_open(MYMYMPI_COMM, gridded_writedata2, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &pFileimg);

  for (int isector = 0; isector < size; isector++)
  {
#ifdef RING // Let the MPI_Get copy from the right location (Results must be checked!) [GL]
    MPI_Get(gridss, size_of_grid, MPI_DOUBLE, isector, 0, size_of_grid, MPI_DOUBLE, Me.win.win);
#endif
    for (unsigned int i = 0; i < size_of_grid / 2; i++)
    {
      gridss_real[i] = gridss[2 * i];
      gridss_img[i] = gridss[2 * i + 1];
    }
    if (num_w_planes > 1)
    {
      for (int iw = 0; iw < num_w_planes; iw++)
        for (int iv = 0; iv < yaxis; iv++)
          for (int iu = 0; iu < xaxis; iu++)
          {
            unsigned int global_index = (iu + (iv + isector * yaxis) * xaxis + iw * grid_size_x * grid_size_y) * sizeof(double);
            unsigned int index = iu + iv * xaxis + iw * xaxis * yaxis;
            MPI_File_write_at_all(pFilereal, global_index, &gridss_real[index], size_of_grid / 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
            MPI_File_write_at_all(pFileimg, global_index, &gridss_img[index], size_of_grid / 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
          }
    }
    else
    {
      for (int iw = 0; iw < num_w_planes; iw++)
      {
        unsigned int global_index = (xaxis * isector * yaxis + iw * grid_size_x * grid_size_y) * sizeof(double);
        unsigned int index = iw * xaxis * yaxis;
        MPI_File_write_at_all(pFilereal, global_index, &gridss_real[index], xaxis * yaxis, MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_write_at_all(pFileimg, global_index, &gridss_img[index], xaxis * yaxis, MPI_DOUBLE, MPI_STATUS_IGNORE);
      }
    }
  }
  MPI_File_close(&pFilereal);
  MPI_File_close(&pFileimg);

  free(gridss_real);
  free(gridss_img);

  if (size > 1)
  {
    MPI_Barrier(MYMYMPI_COMM);
  }
#endif // WRITE_DATA

  return;
}

void gridding(
    int rank,
    int size,
    int nmeasures,
    double *uu,
    double *vv,
    double *ww,
    double *grid,
    double *gridss,
    MPI_Comm MYMPI_COMM,
    int num_threads,
    int grid_size_x,
    int grid_size_y,
    int w_support,
    int num_w_planes,
    int polarisations,
    int freq_per_chan,
#if defined(WRITE_DATA)
    char *gridded_writedata1,
    char *gridded_writedata2,
#endif
#if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
    float *visreal_stokes,
    float *visimg_stokes,
    float *weights_stokes,
#else
    float *visreal,
    float *visimg,
    float *weights,
#endif
    double uvmin,
    double uvmax,
    int freq_index)
{

  if (rank == 0)
    printf("RICK GRIDDING DATA\n");

  // double start = CPU_TIME_wt;

  int xaxis = grid_size_x;
  int yaxis = grid_size_y / size;
  int size_of_grid = 2 * num_w_planes * xaxis * yaxis;

  // Pixel size in radians
  // float pixsize_x = 0.000002;

  double dx = 1.0 / (double)grid_size_x;
  double dw = 1.0 / (double)num_w_planes;
  double w_supporth = (double)((w_support - 1) / 2) * dx;

  // Create histograms and linked lists
  // Initialize linked list
  initialize_array(
      size,
      nmeasures,
      w_supporth,
      vv,
      yaxis,
      dx);

  channelselect(
      nmeasures,
      freq_per_chan,
#if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
      visreal_stokes,
      visimg_stokes,
      weights_stokes,
#else
      visreal,
      visimg,
      weights,
#endif
      freq_index);

  // Sector and Gridding data
  gridding_data(
      dx,
      dw,
      num_threads,
      size,
      rank,
      xaxis,
      yaxis,
      grid_size_x,
      grid_size_y,
      size_of_grid,
      num_w_planes,
      w_support,
      uvmin,
      uvmax,
      polarisations,
      freq_per_chan,
      uu,
      vv,
      ww,
      grid,
      gridss,
#if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
      visreal_stokes_ch,
      visimg_stokes_ch,
      weights_stokes_ch,
#else
      visreal_ch,
      visimg_ch,
      weights_ch,
#endif
#if defined(WRITE_DATA)
      gridded_writedata1,
      gridded_writedata2,
#endif
      MYMPI_COMM);

  free_array(histo_send, sectorarray, size);

  if (size > 1)
  {
    MPI_Barrier(MYMPI_COMM);
  }

  return;
}