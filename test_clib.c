#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
#include <fftw3-mpi.h>
#ifdef FITSIO
#include "fitsio.h"
#endif

#define PI 3.14159265359

#define NFILES 100
#define FILENAMELENGTH 30

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

int *histo_send;
int **sectorarray;

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

void initialize_array(
    int nsectors,
    int nmeasures,
    double w_supporth,
    double *vv,
    int yaxis,
    double dx)
{
  printf("Beginning of _initialize_array_ function\n");
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

  int *counter = calloc((nsectors + 1), sizeof(int));
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
    int rank)
{
  int i;
  // int index;
  unsigned long long visindex;

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
  myull Nvis = num_points * freq_per_chan * polarizations;
#pragma omp target teams distribute parallel for private(visindex) map(to : uu[0 : num_points], vv[0 : num_points], ww[0 : num_points], vis_real[0 : Nvis], vis_img[0 : Nvis], weight[0 : Nvis / freq_per_chan]) map(tofrom : grid[0 : 2 * num_w_planes * grid_size_x * grid_size_y])
#else
#pragma omp parallel for private(visindex)
#endif

  printf("Before _for_ loop into _wstack_ function\n");
  for (i = 0; i < num_points; i++)
  {
#ifdef _OPENMP
    // int tid;
    // tid = omp_get_thread_num();
    // printf("%d\n",tid);
#endif

    visindex = i * freq_per_chan * polarizations;

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
        unsigned long long ifine = visindex;
        // DAV: the following two loops are performend by each thread separately: no problems of race conditions
        for (int ifreq = 0; ifreq < freq_per_chan; ifreq++)
        {
          int iweight = visindex / freq_per_chan;
          for (int ipol = 0; ipol < polarizations; ipol++)
          {
            if (!isnan(vis_real[ifine]))
            {
              add_term_real += weight[iweight] * vis_real[ifine] * conv_weight;
              add_term_img += weight[iweight] * vis_img[ifine] * conv_weight;
            }
            ifine++;
            iweight++;
          }
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
}

void free_array(int *histo_send, int **sectorarrays, int nsectors)

{
  printf("Beginning of _free_array_ function\n");
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
    MPI_Comm MYMYMPI_COMM)
//
// actually performs the gridding of the data
//
{

  double shift = (double)(dx * yaxis);

  // calculate the resolution in radians
  double resolution = 1.0 / MAX(fabs(uvmin), fabs(uvmax));

  // calculate the resolution in arcsec
  double resolution_asec = (3600.0 * 180.0) / MAX(fabs(uvmin), fabs(uvmax)) / PI;
  if (rank == 0)
    printf("RESOLUTION = %f rad, %f arcsec\n", resolution, resolution_asec);

  for (int isector = 0; isector < size; isector++)
  {
    // double start = CPU_TIME_wt;

    int Nsec = histo_send[isector];
    int Nweightss = Nsec * polarisations;
    unsigned long long Nvissec = Nweightss * freq_per_chan;
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
    int ip = 0;
    int inu = 0;

#warning "this loop should be threaded"
#warning "the counter of this loop should not be int"
    for (int iphi = histo_send[isector] - 1; iphi >= 0; iphi--)
    {

      int ilocal = sectorarray[isector][iphi];

      uus[icount] = uu[ilocal];
      vvs[icount] = vv[ilocal] - isector * shift;
      wws[icount] = ww[ilocal];
      for (int ipol = 0; ipol < polarisations; ipol++)
      {
        weightss[ip] = weights[ilocal * polarisations + ipol];
        ip++;
      }
      for (int ifreq = 0; ifreq < polarisations * freq_per_chan; ifreq++)
      {
        visreals[inu] = visreal[ilocal * polarisations * freq_per_chan + ifreq];
        visimgs[inu] = visimg[ilocal * polarisations * freq_per_chan + ifreq];
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

    printf("Calling _wstack_ function\n");

    // We have to call different GPUs per MPI task!!! [GL]
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
           rank);

#ifdef VERBOSE
    printf("Processed sector %ld\n", isector);
#endif

    if (size > 1)
    {
      // Write grid in the corresponding remote slab

      int target_rank = (int)(isector % size);

      // Force to use MPI_Reduce when -fopenmp is not active
#ifdef _OPENMP
      if (reduce_method == REDUCE_MPI)

        MPI_Reduce(gridss, grid, size_of_grid, MPI_DOUBLE, MPI_SUM, target_rank, MYMYMPI_COMM);

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
    float *visreal,
    float *visimg,
    float *weights,
    double uvmin,
    double uvmax)
{

  if (rank == 0)
    printf("RICK GRIDDING DATA\n");

  // double start = CPU_TIME_wt;

  int xaxis = grid_size_x;
  int yaxis = grid_size_y / size;
  int size_of_grid = 2 * num_w_planes * xaxis * yaxis;

  double dx = 1.0 / (double)grid_size_x;
  double dw = 1.0 / (double)num_w_planes;
  double w_supporth = (double)((w_support - 1) / 2) * dx;

  printf("Calling _initialize_array_ function\n");

  // Create histograms and linked lists

  // Initialize linked list
  initialize_array(
      size,
      nmeasures,
      w_supporth,
      vv,
      yaxis,
      dx);

  printf("Calling _gridding_data_ function\n");
  // Sector and Gridding data
  gridding_data(
      dx,
      dw,
      num_threads,
      size,
      rank,
      xaxis,
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
      gridss,
      visreal,
      visimg,
      weights,
      MYMPI_COMM);

  free_array(histo_send, sectorarray, size);

  if (size > 1)
  {
    MPI_Barrier(MYMPI_COMM);
  }

  return;
}

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
  {
    printf("RICK FFT\n");
  }

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

void phase_correction(
    double *gridss,
    double *image_real,
    double *image_imag,
    double num_w_planes,
    int xaxistot,
    int yaxistot,
    double wmin,
    double wmax,
    double uvmin,
    double uvmax,
    int num_threads,
    int size,
    int rank,
    MPI_Comm MYMPI_COMM)
{
  printf("RICK PHASE CORRECTION\n");

  double dw = (wmax - wmin) / (double)num_w_planes;
  double wterm = wmin + 0.5 * dw;
  double dwnorm = dw / (wmax - wmin);

  int xaxis = xaxistot;
  int yaxis = yaxistot / size;

  double resolution = 1.0 / MAX(fabs(uvmin), fabs(uvmax));

  FILE *pFilereal;
  FILE *pFileimg;
  char fftfile2[FILENAMELENGTH] = "fft_real.bin";
  char fftfile3[FILENAMELENGTH] = "fft_img.bin";

#ifdef FITSIO
  fitsfile *fptreal;
  fitsfile *fptrimg;
  int status;
  char testfitsreal[FILENAMELENGTH] = "ricklib_real.fits";
  char testfitsimag[FILENAMELENGTH] = "ricklib_img.fits";
#endif

#ifndef ACCOMP

#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif

#pragma omp parallel for collapse(2) private(wterm)
  for (int iw = 0; iw < num_w_planes; iw++)
  {
    for (int iv = 0; iv < yaxis; iv++)
      for (int iu = 0; iu < xaxis; iu++)
      {

        unsigned int index = 2 * (iu + iv * xaxis + xaxis * yaxis * iw);
        unsigned int img_index = iu + iv * xaxis;
        wterm = wmin + iw * dw;
#ifdef PHASE_ON
        if (num_w_planes > 1)
        {
          double xcoord = (double)(iu - xaxistot / 2);
          if (xcoord < 0.0)
            xcoord = (double)(iu + xaxistot / 2);
          xcoord = sin(xcoord * resolution);
          double ycoord = (double)(iv - yaxistot / 2);
          if (ycoord < 0.0)
            ycoord = (double)(iv + yaxistot / 2);
          ycoord = sin(ycoord * resolution);

          double preal, pimag;
          double radius2 = (xcoord * xcoord + ycoord * ycoord);
          if (xcoord <= 1.0)
          {
            preal = cos(2.0 * PI * wterm * (sqrt(1 - radius2) - 1.0));
            pimag = sin(2.0 * PI * wterm * (sqrt(1 - radius2) - 1.0));
          }
          else
          {
            preal = cos(-2.0 * PI * wterm * (sqrt(radius2 - 1.0) - 1));
            pimag = 0.0;
          }

          preal = cos(2.0 * PI * wterm * (sqrt(1 - radius2) - 1.0));
          pimag = sin(2.0 * PI * wterm * (sqrt(1 - radius2) - 1.0));

          double p, q, r, s;
          p = gridss[index];
          q = gridss[index + 1];
          r = preal;
          s = pimag;

          // printf("%d %d %d %ld %ld\n",iu,iv,iw,index,img_index);
#pragma omp atomic
          image_real[img_index] += (p * r - q * s) * dwnorm * sqrt(1 - radius2);
#pragma omp atomic
          image_imag[img_index] += (p * s + q * r) * dwnorm * sqrt(1 - radius2);
        }
        else
        {
#pragma omp atomic
          image_real[img_index] += gridss[index];
          // printf("image_real[%d] = %f\n", img_index, image_real[img_index]);
#pragma omp atomic
          image_imag[img_index] += gridss[index + 1];
        }
#else
#pragma omp atomic
        image_real[img_index] += gridss[index];
        // printf("image_real[%d] = %f\n", img_index, image_real[img_index]);
#pragma omp atomic
        image_imag[img_index] += gridss[index + 1];
#endif // end of PHASE_ON
      }
  }

#else
  omp_set_default_device(rank % omp_get_num_devices());

#if !defined(__clang__)

#pragma omp target teams distribute parallel for collapse(2) simd private(wterm) map(to : gridss[0 : 2 * num_w_planes * xaxis * yaxis]) map(from : image_real[0 : xaxis * yaxis]) map(from : image_imag[0 : xaxis * yaxis])

#else

#pragma omp target teams distribute parallel for collapse(2) private(wterm) map(to : gridss[0 : 2 * num_w_planes * xaxis * yaxis]) map(from : image_real[0 : xaxis * yaxis]) map(from : image_imag[0 : xaxis * yaxis])
#endif

  for (int iw = 0; iw < num_w_planes; iw++)
  {
    for (int iv = 0; iv < yaxis; iv++)
      for (int iu = 0; iu < xaxis; iu++)
      {

        long index = 2 * (iu + iv * xaxis + xaxis * yaxis * iw);
        long img_index = iu + iv * xaxis;
        wterm = wmin + iw * dw;
#ifdef PHASE_ON
        if (num_w_planes > 1)
        {
          double xcoord = (double)(iu - xaxistot / 2);
          if (xcoord < 0.0)
            xcoord = (double)(iu + xaxistot / 2);
          xcoord = sin(xcoord * resolution);
          double ycoord = (double)(iv - yaxistot / 2);
          if (ycoord < 0.0)
            ycoord = (double)(iv + yaxistot / 2);
          ycoord = sin(ycoord * resolution);

          double preal, pimag;
          double radius2 = (xcoord * xcoord + ycoord * ycoord);
          if (xcoord <= 1.0)
          {
            preal = cos(2.0 * PI * wterm * (sqrt(1 - radius2) - 1.0));
            pimag = sin(2.0 * PI * wterm * (sqrt(1 - radius2) - 1.0));
          }
          else
          {
            preal = cos(-2.0 * PI * wterm * (sqrt(radius2 - 1.0) - 1));
            pimag = 0.0;
          }

          preal = cos(2.0 * PI * wterm * (sqrt(1 - radius2) - 1.0));
          pimag = sin(2.0 * PI * wterm * (sqrt(1 - radius2) - 1.0));

          double p, q, r, s;
          p = gridss[index];
          q = gridss[index + 1];
          r = preal;
          s = pimag;

          // printf("%d %d %d %ld %ld\n",iu,iv,iw,index,img_index);
#pragma omp atomic
          image_real[img_index] += (p * r - q * s) * dwnorm * sqrt(1 - radius2);
#pragma omp atomic
          image_imag[img_index] += (p * s + q * r) * dwnorm * sqrt(1 - radius2);
        }
        else
        {
#pragma omp atomic
          image_real[img_index] += gridss[index];
#pragma omp atomic
          image_imag[img_index] += gridss[index + 1];
        }
#else
#pragma omp atomic
        image_real[img_index] += gridss[index];
        printf("image_real[%d] = %f\n", img_index, image_real[img_index]);
#pragma omp atomic
        image_imag[img_index] += gridss[index + 1];
#endif // end of PHASE_ON
      }
  }

#endif

  if (rank == 0)
  {

#ifdef FITSIO
    printf("REMOVING RESIDUAL FITS FILE\n");
    remove(testfitsreal);
    remove(testfitsimag);

    printf("FITS CREATION\n");
    status = 0;

    fits_create_file(&fptrimg, testfitsimag, &status);
    fits_create_img(fptrimg, DOUBLE_IMG, naxis, naxes, &status);
    fits_close_file(fptrimg, &status);

    status = 0;

    fits_create_file(&fptreal, testfitsreal, &status);
    fits_create_img(fptreal, DOUBLE_IMG, naxis, naxes, &status);
    fits_close_file(fptreal, &status);
#endif

    pFilereal = fopen(fftfile2, "wb");
    pFileimg = fopen(fftfile3, "wb");
    fclose(pFilereal);
    fclose(pFileimg);
  }

  if (size > 1)
  {
    MPI_Barrier(MYMPI_COMM);
  }

  if (rank == 0)
    printf("WRITING IMAGE\n");

#ifdef FITSIO
  myuint *fpixel = (myuint *)malloc(sizeof(myuint) * naxis);
  myuint *lpixel = (myuint *)malloc(sizeof(myuint) * naxis);
#endif

#ifdef FITSIO

  fpixel[0] = 1;
  fpixel[1] = rank * yaxis + 1;
  lpixel[0] = xaxis;
  lpixel[1] = (rank + 1) * yaxis;

  status = 0;
  fits_open_image(&fptreal, testfitsreal, READWRITE, &status);
  fits_write_subset(fptreal, TDOUBLE, fpixel, lpixel, image_real, &status);
  fits_close_file(fptreal, &status);

  status = 0;
  fits_open_image(&fptrimg, testfitsimag, READWRITE, &status);
  fits_write_subset(fptrimg, TDOUBLE, fpixel, lpixel, image_imag, &status);
  fits_close_file(fptrimg, &status);

#endif // FITSIO

  pFilereal = fopen(fftfile2, "wb");
  pFileimg = fopen(fftfile3, "wb");

  long global_index = rank * (xaxis * yaxis) * sizeof(long);

  fseek(pFilereal, global_index, SEEK_SET);
  fwrite(image_real, xaxis * yaxis, sizeof(double), pFilereal);
  fseek(pFileimg, global_index, SEEK_SET);
  fwrite(image_imag, xaxis * yaxis, sizeof(double), pFileimg);

  fclose(pFilereal);
  fclose(pFileimg);

  if (size > 1)
  {
    MPI_Barrier(MYMPI_COMM);
  }
}

int main(int argc, char **argv)
{
  int rank;
  int size;

  // Define main filenames
  FILE *pFile;
  FILE *pFile1;
  FILE *pFilereal;
  FILE *pFileimg;
  // Global filename to be composed
  char filename[1000];

  // MS paths
  char datapath[900];
  char datapath_multi[NFILES][900];
  char ufile[FILENAMELENGTH] = "ucoord.bin";
  char vfile[FILENAMELENGTH] = "vcoord.bin";
  char wfile[FILENAMELENGTH] = "wcoord.bin";
  char weightsfile[FILENAMELENGTH] = "weights.bin";
  char visrealfile[FILENAMELENGTH] = "visibilities_real.bin";
  char visimgfile[FILENAMELENGTH] = "visibilities_img.bin";
  char metafile[FILENAMELENGTH] = "meta.txt";

  // Visibilities related variables
  double *uu;
  double *vv;
  double *ww;
  float *weights;
  float *visreal;
  float *visimg;

  int Nmeasures;
  int Nvis;
  int Nweights;
  int freq_per_chan;
  int polarisations;
  int Ntimes;
  double dt;
  double thours;
  long baselines;
  double uvmin;
  double uvmax;
  double wmin;
  double wmax;
  double resolution;

  // Mesh related parameters: global size
  int grid_size_x = 2048;
  int grid_size_y = 2048;
  // Split Mesh size (auto-calculated)
  int local_grid_size_x;
  int local_grid_size_y;
  int xaxis;
  int yaxis;

  // Number of planes in the w direction
  int num_w_planes = 1;

  // Size of the convoutional kernel support
  int w_support = 7;

  double dx = 1.0 / (double)grid_size_x;
  double dw = 1.0 / (double)num_w_planes;

  double w_supporth = (double)((w_support - 1) / 2) * dx;

  long naxis = 2;
  long naxes[2] = {grid_size_x, grid_size_y};

#ifdef USE_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (rank == 0)
    printf("Running with %d MPI tasks\n", size);
#else
  rank = 0;
  size = 1;
#endif

  xaxis = grid_size_x;
  yaxis = grid_size_y / size;

  int ndatasets = 1;
  strcpy(datapath_multi[0], "./newgauss2noconj_t201806301100_SBL180.binMS/");

  strcpy(datapath, datapath_multi[0]);
  // Read metadata
  strcpy(filename, datapath);
  strcat(filename, metafile);
  pFile = fopen(filename, "r");
  fscanf(pFile, "%d", &Nmeasures);
  fscanf(pFile, "%d", &Nvis);
  fscanf(pFile, "%d", &freq_per_chan);
  fscanf(pFile, "%d", &polarisations);
  fscanf(pFile, "%d", &Ntimes);
  fscanf(pFile, "%lf", &dt);
  fscanf(pFile, "%lf", &thours);
  fscanf(pFile, "%ld", &baselines);
  fscanf(pFile, "%lf", &uvmin);
  fscanf(pFile, "%lf", &uvmax);
  fscanf(pFile, "%lf", &wmin);
  fscanf(pFile, "%lf", &wmax);
  fclose(pFile);

  Nvis = Nmeasures * freq_per_chan * polarisations;
  Nweights = Nmeasures * polarisations;

  long nm_pe = (long)(Nmeasures / size);
  long remaining = Nmeasures % size;
  long startrow = rank * nm_pe;
  if (rank == size - 1)
    nm_pe = nm_pe + remaining;

  if (rank == 0)
  {
    printf("N. measurements %d\n", Nmeasures);
    printf("N. visibilities %d\n", Nvis);
  }

  uu = (double *)calloc(Nmeasures, sizeof(double));
  vv = (double *)calloc(Nmeasures, sizeof(double));
  ww = (double *)calloc(Nmeasures, sizeof(double));
  weights = (float *)calloc(Nweights, sizeof(float));
  visreal = (float *)calloc(Nvis, sizeof(float));
  visimg = (float *)calloc(Nvis, sizeof(float));

  if (rank == 0)
    printf("READING DATA\n");
  // Read data
  strcpy(filename, datapath);
  strcat(filename, ufile);
  // printf("Reading %s\n",filename);

  pFile = fopen(filename, "rb");
  fseek(pFile, startrow * sizeof(double), SEEK_SET);
  fread(uu, Nmeasures * sizeof(double), 1, pFile);
  fclose(pFile);

  strcpy(filename, datapath);
  strcat(filename, vfile);
  // printf("Reading %s\n",filename);

  pFile = fopen(filename, "rb");
  fseek(pFile, startrow * sizeof(double), SEEK_SET);
  fread(vv, Nmeasures * sizeof(double), 1, pFile);
  fclose(pFile);

  strcpy(filename, datapath);
  strcat(filename, wfile);
  // printf("Reading %s\n",filename);

  pFile = fopen(filename, "rb");
  fseek(pFile, startrow * sizeof(double), SEEK_SET);
  fread(ww, Nmeasures * sizeof(double), 1, pFile);
  fclose(pFile);

  strcpy(filename, datapath);
  strcat(filename, weightsfile);
  pFile = fopen(filename, "rb");
  fseek(pFile, startrow * polarisations * sizeof(float), SEEK_SET);
  fread(weights, (Nweights) * sizeof(float), 1, pFile);
  fclose(pFile);

  strcpy(filename, datapath);
  strcat(filename, visrealfile);
  pFile = fopen(filename, "rb");
  fseek(pFile, startrow * freq_per_chan * polarisations * sizeof(float), SEEK_SET);
  fread(visreal, Nvis * sizeof(float), 1, pFile);
  fclose(pFile);
  strcpy(filename, datapath);
  strcat(filename, visimgfile);
#ifdef VERBOSE
  printf("Reading %s\n", filename);
#endif
  pFile = fopen(filename, "rb");
  fseek(pFile, startrow * freq_per_chan * polarisations * sizeof(float), SEEK_SET);
  fread(visimg, Nvis * sizeof(float), 1, pFile);
  fclose(pFile);

  long size_of_grid = 2 * num_w_planes * xaxis * yaxis;

  double *grid;
  grid = (double *)calloc(size_of_grid, sizeof(double));
  double *gridss;
  gridss = (double *)calloc(size_of_grid, sizeof(double));

  double *image_real = (double *)calloc(xaxis * yaxis, sizeof(double));
  double *image_imag = (double *)calloc(xaxis * yaxis, sizeof(double));

#ifdef USE_MPI
  MPI_Win slabwin;
  MPI_Win_create(grid, size_of_grid * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &slabwin);
  MPI_Win_fence(0, slabwin);
#endif

  gridding(
      rank,
      size,
      Nmeasures,
      uu,
      vv,
      ww,
      grid,
      gridss,
      MPI_COMM_WORLD,
      1,
      grid_size_x,
      grid_size_y,
      w_support,
      num_w_planes,
      polarisations,
      freq_per_chan,
      visreal,
      visimg,
      weights,
      uvmin,
      uvmax);

  fftw_data(
      grid_size_x,
      grid_size_y,
      num_w_planes,
      1,
      MPI_COMM_WORLD,
      size,
      rank,
      grid,
      gridss);

  phase_correction(
      gridss,
      image_real,
      image_imag,
      num_w_planes,
      grid_size_x,
      grid_size_y,
      wmin,
      wmax,
      uvmin,
      uvmax,
      1,
      size,
      rank,
      MPI_COMM_WORLD);

  MPI_Finalize();

  printf("End of main\n");
}
