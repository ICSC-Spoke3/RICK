#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
#include "ricklib.h"
#include <omp.h>
#ifdef FITSIO
#include "fitsio.h"
#endif
//#include <omp.h>  /*to be included after checking the MPI version works */

#define PI 3.14159265359
#define FILENAMELENGTH 30

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

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
  if (rank==0)
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

  //OMP debugging verification
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
