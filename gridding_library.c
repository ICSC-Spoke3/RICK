#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
#include "ricklib.h"
#include <omp.h>

#define PI 3.14159265359
#define NTHREADS 32
#define NWORKERS -1
#define FILENAMELENGTH 30

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#define NOT_ENOUGH_MEM_STACKING 3

int *histo_send;
int **sectorarray;

float weights_stokes_sum;

#if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
float *visreal_stokes;
float *visimg_stokes;
float *weights_stokes;
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

/*

Stokes collapsing

*/

void stokes(
    myull Nmeasures,
    int freq_per_chan,
    float *visreal,
    float *visimg,
    float *weights)
{
  // In this way we select and combine correlations to form Stokes parameters

#if !defined(WEIGHTING_UNIFORM) || !defined(WEIGHTING_BRIGGS)
  weights_stokes_sum = 0.0;
#endif

#if defined(STOKESI)
  visreal_stokes = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  visimg_stokes  = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  weights_stokes = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  for (myull i = 0; i < (Nmeasures * freq_per_chan); i++)
  {
    visreal_stokes[i] = 0.5 * (visreal[i * 4] + visreal[(i * 4) + 3]);
    visimg_stokes[i] = 0.5 * (visimg[i * 4] + visimg[(i * 4) + 3]);
    weights_stokes[i] = 0.25 * (weights[i * 4] + weights[(i * 4) + 3]);
#if !defined(WEIGHTING_UNIFORM) || !defined(WEIGHTING_BRIGGS)
    weights_stokes_sum += weights_stokes[i];
#endif
  }
  // printf("Sum weights Stokes I %f\n", weights_stokes_sum);
#elif defined(STOKESQ)
  visreal_stokes = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  visimg_stokes = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  weights_stokes = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  for (myull i = 0; i < (Nmeasures * freq_per_chan); i++)
  {
    visreal_stokes[i] = 0.5 * (visreal[i * 4] - visreal[(i * 4) + 3]);
    visimg_stokes[i] = 0.5 * (visimg[i * 4] - visimg[(i * 4) + 3]);
    weights_stokes[i] = weights[i * 4];
#if !defined(WEIGHTING_UNIFORM) || !defined(WEIGHTING_BRIGGS)
    weights_stokes_sum += weights_stokes[i];
#endif
  }
#elif defined(STOKESU)
  visreal_stokes = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  visimg_stokes = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  weights_stokes = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  for (myull i = 0; i < (Nmeasures * freq_per_chan); i++)
  {
    visreal_stokes[i] = 0.5 * (visreal[(i * 4) + 1] + visreal[(i * 4) + 2]);
    visimg_stokes[i] = 0.5 * (visimg[(i * 4) + 1] + visimg[(i * 4) + 2]);
    weights_stokes[i] = weights[i * 4];
#if !defined(WEIGHTING_UNIFORM) || !defined(WEIGHTING_BRIGGS)
    weights_stokes_sum += weights_stokes[i];
#endif
  }
// #elif defined(STOKESV)
// float * visreal_stokesI = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
// float * visimg_stokesI = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
// float * weights_stokesI = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
#endif

  // printf("Sum of all Stokes I weights %f\n", weights_stokes_sum);

  free(weights);
  free(visimg);
  free(visreal);
    
}


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
    myull num_points,
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
    int y_start,
    int yaxis,
    double *grid,
    int num_threads,
    int rank,
    int size,
    int grid_size_y)
{
  myull i;
  // int index;
  myull visindex;

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

#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif

#if defined(ACCOMP) && (GPU_STACKING)
  omp_set_default_device(rank % omp_get_num_devices());
  myull Nvis = num_points * freq_per_chan;
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

  out_weight_uniform = (float *)malloc(num_points * freq_per_chan * sizeof(float));
  weight_uv = (float *)malloc(grid_size_x * grid_size_y * num_w_planes * sizeof(float));
  weight_uv_2 = (float *)malloc(grid_size_x * grid_size_y * num_w_planes * sizeof(float));
  robustness = (float *)malloc(grid_size_x * grid_size_y * num_w_planes * size * sizeof(float));

  for (i = 0; i < num_points; i++)
  {
    visindex = i * freq_per_chan;

    double sum = 0.0;
    myuint j, k;

    // Convert UV coordinates to grid coordinates.
    double pos_u = uu[i] / dx;
    double pos_v = vv[i] / dx;
    double ww_i = ww[i] / dw;

    int grid_w = (int)ww_i;
    int grid_u = (int)pos_u;
    int grid_v = (int)pos_v;

    // check the boundaries
    myuint jmin = (grid_u > KernelLen - 1) ? grid_u - KernelLen : 0;
    myuint jmax = (grid_u < grid_size_x - KernelLen) ? grid_u + KernelLen : grid_size_x - 1;
    myuint kmin = (grid_v > KernelLen - 1) ? grid_v - KernelLen : 0;
    myuint kmax = (grid_v < grid_size_y - KernelLen) ? grid_v + KernelLen : grid_size_y - 1;
    // printf("%ld, %ld, %ld, %ld\n",jmin,jmax,kmin,kmax);

    // Convolve this point onto the grid.
    for (k = kmin; k <= kmax; k++)
    {
      for (j = jmin; j <= jmax; j++)
      {
        myuint iKer = 2 * (j + k * grid_size_x + grid_w * grid_size_x * grid_size_y);
#if defined(WEIGHTING_UNIFORM)
        weighting_uniform(iKer, visindex, weight, weight_uv);
        // To be done!!!!!!!!!!
        // float weights_stokes_sum = 0.0;
        // for (myull i = 0; i < (Nmeasures * freq_per_chan); i++)
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

  myuint y_end = y_start + yaxis - 1;
  
  for (i = 0; i < num_points; i++)
  {
#ifdef _OPENMP
    // int tid;
    // tid = omp_get_thread_num();
    // printf("%d\n",tid);
#endif

    visindex = i * freq_per_chan;

    double sum = 0.0;
    myuint j, k;

    /* Convert UV coordinates to grid coordinates. */
    double pos_u = uu[i] / dx;
    double pos_v = vv[i] / dx;
    double ww_i = ww[i] / dw;

    int grid_w = (int)ww_i;
    int grid_u = (int)pos_u;
    int grid_v = (int)pos_v;

    // check the boundaries
    myuint jmin = (grid_u > KernelLen - 1) ? grid_u - KernelLen : 0;
    myuint jmax = (grid_u < grid_size_x - KernelLen) ? grid_u + KernelLen : grid_size_x - 1;
    //myuint kmin = (grid_v > y_start - 1) ? grid_v - KernelLen : y_start - KernelLen;
    //myuint kmax = (grid_v < y_start + yaxis) ? grid_v + KernelLen : y_start + yaxis - 1;
    
    myuint kmin;
    myuint kmax;

    if (y_start == 0)
      kmin = (grid_v > KernelLen - 1) ? grid_v - KernelLen : 0;
    else
      kmin = (grid_v < y_start) ? y_start - KernelLen : grid_v - KernelLen;

    if (y_end == grid_size_y)
      kmax = (grid_v < grid_size_y - KernelLen) ? grid_v + KernelLen : grid_size_y - 1;
    else
      kmax = (grid_v > y_end) ? y_end + KernelLen - 1 : grid_v + KernelLen;
    
    // Convolve this point onto the grid.
    for (k = kmin; k <= kmax; k++)
    {
      /* Avoid using points in the ghost region */
      
      if (k < y_start || k > y_end)
	continue;
          
      double v_dist = (double)k + 0.5 - pos_v;
      int kKer = (int)(increaseprecision * (fabs(v_dist + (double)KernelLen)));
      
      
      for (j = jmin; j <= jmax; j++)
      {
        double u_dist = (double)j + 0.5 - pos_u;
        myull iKer = 2 * (j + (k-y_start) * grid_size_x + grid_w * grid_size_x * yaxis);
        int jKer = (int)(increaseprecision * (fabs(u_dist + (double)KernelLen)));
	
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
        myull ifine = visindex;
        // DAV: the following two loops are performend by each thread separately: no problems of race conditions
        for (int ifreq = 0; ifreq < freq_per_chan; ifreq++)
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
    int y_start,
    int yaxis,
    myull size_of_grid,
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
    float *visreal,
    float *visimg,
    float *weights,
#if defined(WRITE_DATA)
    char *gridded_writedata1,
    char *gridded_writedata2,
#endif
    MPI_Comm MYMYMPI_COMM,
    myull total_size,
    int grid_size_y)
{

  double shift = (double)(dx * yaxis);

  // calculate the resolution in radians
  double resolution = 1.0 / MAX(fabs(uvmin), fabs(uvmax));

  // calculate the resolution in arcsec
  double resolution_asec = (3600.0 * 180.0) / MAX(fabs(uvmin), fabs(uvmax)) / PI;
  if (rank == 0)
    printf("RESOLUTION = %f rad, %f arcsec\n", resolution, resolution_asec);
  
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
    for (myull ipart = 0; ipart < total_size; ipart++)
      {
        my_uumin = MIN(my_uumin, uu[ipart]);
        my_uumax = MAX(my_uumax, uu[ipart]);
        my_vvmin = MIN(my_vvmin, vv[ipart]);
        my_vvmax = MAX(my_vvmax, vv[ipart]);
      }

    uumin = MIN(uumin, my_uumin);
    uumax = MAX(uumax, my_uumax);
    vvmin = MIN(vvmin, my_vvmin);
    vvmax = MAX(vvmax, my_vvmax);
  }
    

  // Make convolution on the grid


  // printf("Inside wstack() \n");
  wstack(num_w_planes,
	 total_size,
	 freq_per_chan,
	 polarisations,
	 uu,
	 vv,
	 ww,
	 visreal,
	 visimg,
	 weights,
	 dx,
	 dw,
	 w_support,
	 xaxis,
	 y_start,
	 yaxis,
	 grid,
	 num_threads,
	 rank,
	 size,
	 grid_size_y);


  /* Wait until all MPI tasks perform the gridding */
  MPI_Barrier(MYMYMPI_COMM);

 #ifdef WRITE_DATA

  if (rank == 0)
    {
      printf("WRITING GRIDDED DATA\n");
    }

  MPI_File pFilereal;
  MPI_File pFileimg;

  double *gridss_real = (double *)malloc(size_of_grid / 2 * sizeof(double));
  double *gridss_img = (double *)malloc(size_of_grid / 2 * sizeof(double));

  for (myull i = 0; i < size_of_grid / 2; i++)
    {
      gridss_real[i] = grid[2 * i];
      gridss_img[i]  = grid[2 * i + 1];
    }
    
  int ierr;
  ierr = MPI_File_open(MYMYMPI_COMM, gridded_writedata1, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &pFilereal);

  if (ierr != MPI_SUCCESS)
    {
      if (rank == 0)
	fprintf(stderr, "Error: Could not open file '%s' for writing.\n", gridded_writedata1);
    }
  
  ierr = MPI_File_open(MYMYMPI_COMM, gridded_writedata2, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &pFileimg);

  if (ierr != MPI_SUCCESS)
    {
      if (rank == 0)
	fprintf(stderr, "Error: Could not open file '%s' for writing.\n", gridded_writedata2);
    }
  
  /* TO BE REDEFINED IN CASE OF NON-TRIVIAL DD */
  int gsizes[3] = {num_w_planes, grid_size_y, xaxis};
  int lsizes[3] = {num_w_planes, yaxis, xaxis};
  int starts[3] = {0, y_start, 0};

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

  MPI_Barrier(MYMYMPI_COMM);
  
  free(gridss_real);
  free(gridss_img);

 #endif // WRITE_DATA
  return;
}

  /*
  for (int isector = 0; isector < size; isector++)
  {
#ifdef RING // Let the MPI_Get copy from the right location (Results must be checked!) [GL]
    MPI_Get(gridss, size_of_grid, MPI_DOUBLE, isector, 0, size_of_grid, MPI_DOUBLE, Me.win.win);
#endif
    for (myull i = 0; i < size_of_grid / 2; i++)
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
            myull global_index = (iu + (iv + isector * yaxis) * xaxis + iw * grid_size_x * grid_size_y) * sizeof(double);
            myull index = iu + iv * xaxis + iw * xaxis * yaxis;
            MPI_File_write_at_all(pFilereal, global_index, &gridss_real[index], 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
            MPI_File_write_at_all(pFileimg, global_index, &gridss_img[index], 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
          }
    }
    else
    {
      for (int iw = 0; iw < num_w_planes; iw++)
      {
        myull global_index = (xaxis * isector * yaxis + iw * grid_size_x * grid_size_y) * sizeof(double);
        unsigned int index = iw * xaxis * yaxis;
        MPI_File_write_at_all(pFilereal, global_index, &gridss_real[index], xaxis * yaxis, MPI_DOUBLE, MPI_STATUS_IGNORE);
        MPI_File_write_at_all(pFileimg, global_index, &gridss_img[index], xaxis * yaxis, MPI_DOUBLE, MPI_STATUS_IGNORE);
      }
    }
  }
  */
void gridding(
    int rank,
    int size,
    myull nmeasures,
    double *uu,
    double *vv,
    double *ww,
    double *grid,
    MPI_Comm MYMPI_COMM,
    int num_threads,
    int xaxis,
    int grid_size_y,
    int y_start,
    int yaxis,
    int w_support,
    int num_w_planes,
    int polarisations,
    int freq_per_chan,
#if defined(WRITE_DATA)
    char *gridded_writedata1,
    char *gridded_writedata2,
#endif
    float *visreal,
    float *visimg,
    float *weights,
    double uvmin,
    double uvmax,
    myull total_size)
{

  if (rank == 0)
    printf("RICK GRIDDING DATA\n");

  // double start = CPU_TIME_wt;

  myull size_of_grid = 2 * num_w_planes * xaxis * yaxis;

  double dx = 1.0 / (double)xaxis;
  double dw = 1.0 / (double)num_w_planes;
  double w_supporth = (double)((w_support - 1) / 2) * dx;

 #if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
  // Collapse correlations into Stokes parameters
  stokes(
      total_size,
      freq_per_chan,
      visreal,
      visimg,
      weights);
 #endif

  // Sector and Gridding data
  gridding_data(
      dx,
      dw,
      num_threads,
      size,
      rank,
      xaxis,
      y_start,
      yaxis,
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
     #if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
      visreal_stokes,
      visimg_stokes,
      weights_stokes,
     #else
      visreal,
      visimg,
      weights,
     #endif
#if defined(WRITE_DATA)
      gridded_writedata1,
      gridded_writedata2,
#endif
      MYMPI_COMM,
      total_size,
      grid_size_y);

  MPI_Barrier(MYMPI_COMM);

 #if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
  free(weights_stokes);
  free(visimg_stokes);
  free(visreal_stokes);
 #else
  free(weights);
  free(visimg);
  free(visreal);
 #endif
  
  return;
}
