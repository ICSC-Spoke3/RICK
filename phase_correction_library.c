#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
#include "ricklib.h"
#ifdef FITSIO
#include "fitsio.h"
#endif
// #include <omp.h>

#define PI 3.14159265359
#define FILENAMELENGTH 30
#define NTHREADS 32
#define NWORKERS -1

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#ifdef RICK_GPU

__global__ void phase_g(int xaxis,
                        int yaxis,
                        int num_w_planes,
                        double *gridss,
                        double *image_real,
                        double *image_imag,
                        double wmin,
                        double dw,
                        double dwnorm,
                        int xaxistot,
                        int yaxistot,
                        double resolution,
                        int nbucket)
{
  long gid = blockIdx.x * blockDim.x + threadIdx.x;
  double add_term_real;
  double add_term_img;
  double wterm;
  long arraysize = (long)((xaxis * yaxis * num_w_planes) / nbucket);

  if (gid < arraysize)
  {
    long gid_aux = nbucket * gid;
    for (int iaux = 0; iaux < nbucket; iaux++)
    {
      int iw = gid_aux / (xaxis * yaxis);
      int ivaux = gid_aux % (xaxis * yaxis);
      int iv = ivaux / xaxis;
      int iu = ivaux % xaxis;
      long index = 2 * gid_aux;
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

        preal = cos(2.0 * PI * wterm * (sqrt(1 - radius2) - 1.0));
        pimag = sin(2.0 * PI * wterm * (sqrt(1 - radius2) - 1.0));

        double p, q, r, s;
        p = gridss[index];
        q = gridss[index + 1];
        r = preal;
        s = pimag;

        add_term_real = (p * r - q * s) * dwnorm * sqrt(1 - radius2);
        add_term_img = (p * s + q * r) * dwnorm * sqrt(1 - radius2);
        atomicAdd(&(image_real[img_index]), add_term_real);
        atomicAdd(&(image_imag[img_index]), add_term_img);
      }
      else
      {
        atomicAdd(&(image_real[img_index]), gridss[index]);
        atomicAdd(&(image_imag[img_index]), gridss[index + 1]);
      }
#else
      atomicAdd(&(image_real[img_index]), gridss[index]);
      atomicAdd(&(image_imag[img_index]), gridss[index + 1]);
#endif // end of PHASE_ON
      gid_aux++;
    }
  }
}

#endif

void phase_correction(
    double *gridss,
    double *image_real,
    double *image_imag,
    int num_w_planes,
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
  if (rank == 0)
    printf("RICK PHASE CORRECTION\n");

  double dw = (wmax - wmin) / (double)num_w_planes;
  double wterm = wmin + 0.5 * dw;
  double dwnorm = dw / (wmax - wmin);

  int xaxis = xaxistot;
  int yaxis = yaxistot / size;

  double resolution = 1.0 / MAX(fabs(uvmin), fabs(uvmax));

#ifdef RICK_GPU

  // WARNING: nbucket MUST be chosen such that xaxis*yaxis*num_w_planes is a multiple of nbucket
  int nbucket = 1;
  int Nth = NTHREADS;
  long Nbl = (long)((num_w_planes * xaxis * yaxis) / Nth / nbucket) + 1;
  if (NWORKERS == 1)
  {
    Nbl = 1;
    Nth = 1;
  };

  int ndevices;
  cudaGetDeviceCount(&ndevices);
  cudaSetDevice(rank % ndevices);

  cudaError_t mmm;
  double *image_real_g;
  double *image_imag_g;

  double *gridss_g;

  long long unsigned size_finta = (long long unsigned)(2 * (long long unsigned)num_w_planes * (long long unsigned)xaxis * (long long unsigned)yaxis);

  mmm = cudaMalloc(&gridss_g, (size_t)(size_finta * sizeof(double)));
  mmm = cudaMemcpy(gridss_g, gridss, (size_t)(size_finta * sizeof(double)), cudaMemcpyHostToDevice);

  mmm = cudaMalloc(&image_real_g, xaxis * yaxis * sizeof(double));
  // printf("CUDA ERROR 2 %s\n",cudaGetErrorString(mmm));
  mmm = cudaMalloc(&image_imag_g, xaxis * yaxis * sizeof(double));
  // printf("CUDA ERROR 3 %s\n",cudaGetErrorString(mmm));

  // printf("CUDA ERROR 4 %s\n",cudaGetErrorString(mmm));
  mmm = cudaMemset(image_real_g, 0.0, xaxis * yaxis * sizeof(double));
  // printf("CUDA ERROR 5 %s\n",cudaGetErrorString(mmm));
  mmm = cudaMemset(image_imag_g, 0.0, xaxis * yaxis * sizeof(double));
  // printf("CUDA ERROR 6 %s\n",cudaGetErrorString(mmm));

  // call the phase correction kernel
  phase_g<<<Nbl, Nth>>>(xaxis,
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

  mmm = cudaMemcpy(image_real, image_real_g, xaxis * yaxis * sizeof(double), cudaMemcpyDeviceToHost);
  // printf("CUDA ERROR 7 %s\n",cudaGetErrorString(mmm));
  mmm = cudaMemcpy(image_imag, image_imag_g, xaxis * yaxis * sizeof(double), cudaMemcpyDeviceToHost);
  // printf("CUDA ERROR 8 %s\n",cudaGetErrorString(mmm));

  cudaFree(gridss_g);

#endif

  MPI_File pFilereal;
  MPI_File pFileimg;
  char fftfile2[FILENAMELENGTH] = "output_real.bin";
  char fftfile3[FILENAMELENGTH] = "output_img.bin";

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

  // OMP debugging verification
  /*
#pragma omp parallel
  {
    printf("Hello from thread %d out of %d of task %d\n", omp_get_thread_num(), omp_get_num_threads(), rank);
  }
  */
#pragma omp parallel for collapse(2) private(wterm)
  for (int iw = 0; iw < num_w_planes; iw++)
  {
    for (int iv = 0; iv < yaxis; iv++)
      for (int iu = 0; iu < xaxis; iu++)
      {

        int index = 2 * (iu + iv * xaxis + xaxis * yaxis * iw);
        int img_index = iu + iv * xaxis;
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
#pragma omp atomic
        image_imag[img_index] += gridss[index + 1];
#endif // end of PHASE_ON
      }
  }

#endif

  if (rank == 0)
  {
    printf("WRITING IMAGE\n");

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
  }

  if (size > 1)
  {
    MPI_Barrier(MYMPI_COMM);
  }

#ifdef FITSIO
  unsigned int *fpixel = (unsigned int *)malloc(sizeof(unsigned int) * naxis);
  unsigned int *lpixel = (unsigned int *)malloc(sizeof(unsigned int) * naxis);
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

  MPI_File_open(MYMPI_COMM, fftfile2, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &pFilereal);
  MPI_File_open(MYMPI_COMM, fftfile3, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &pFileimg);

  MPI_Offset offset = (MPI_Offset)rank * xaxis * yaxis * sizeof(double);

  MPI_File_write_at(pFilereal, offset, image_real, xaxis * yaxis, MPI_DOUBLE, MPI_STATUS_IGNORE);
  MPI_File_write_at(pFileimg, offset, image_imag, xaxis * yaxis, MPI_DOUBLE, MPI_STATUS_IGNORE);

  MPI_File_close(&pFilereal);
  MPI_File_close(&pFileimg);

  if (size > 1)
  {
    MPI_Barrier(MYMPI_COMM);
  }
}
