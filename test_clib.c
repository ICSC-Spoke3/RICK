#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
// #include <omp.h>
#include "ricklib.h"
#ifdef FITSIO
#include "fitsio.h"
#endif

#define PI 3.14159265359

#define NFILES 100
#define FILENAMELENGTH 50

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

int main(int argc, char **argv)
{
  int rank;
  int size;

  // Define main filenames
  FILE *pFile;
  FILE *pFile1;

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
  char channels[FILENAMELENGTH] = "freq_file.bin";
  char metafile[FILENAMELENGTH] = "meta.txt";

#if defined(WRITE_DATA)
  char gridded_writedata1[FILENAMELENGTH] = "gridded_data_real.bin";
  char gridded_writedata2[FILENAMELENGTH] = "gridded_data_img.bin";
  char fftfile_writedata1[FILENAMELENGTH] = "ffted_data_real.bin";
  char fftfile_writedata2[FILENAMELENGTH] = "ffted_data_img.bin";
#endif

  // Visibilities related variables
  double *uu;
  double *vv;
  double *ww;
  float *weights;
  float *visreal;
  float *visimg;
  float *channels_ref;

  int Nmeasures;
  long Nvis;
  int Nweights;
  int freq_per_chan;
  int polarisations;
  int nweights1, nweights2, nweights3;
  int Ntimes;
  double dt;
  double thours;
  long baselines;
  double uvmin;
  double uvmax;
  double wmin;
  double wmax;
  double resolution;

  // Mesh related parameters

  // Grid size in pixels
  int grid_size_x = 4096;
  int grid_size_y = 4096;

  // Split Mesh size (auto-calculated)
  int local_grid_size_x;
  int local_grid_size_y;
  int xaxis;
  int yaxis;

  // Number of planes in the w direction
  int num_w_planes = 1;

  // Size of the convoutional kernel support in pixels
  int w_support = 7;

  double dx = 1.0 / (double)grid_size_x;
  double dw = 1.0 / (double)num_w_planes;

  double w_supporth = (double)((w_support - 1) / 2) * dx;

  long naxis = 2;
  long naxes[2] = {grid_size_x, grid_size_y};

  int num_threads;

#ifdef USE_MPI
#ifdef _DOPENMP // Use MPI and OpenMP
  int thread_level;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
#else
  MPI_Init(&argc, &argv);
#endif //_OPENMP
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef _OPENMP
  num_threads = omp_get_max_threads();
#else
  num_threads = 1;
#endif //_OPENMP

  if (rank == 0)
  {
    printf("\n");
    printf("RRR   III   CCCC  K   K\n");
    printf("R  R   I   C      K  K \n");
    printf("RRR    I   C      KK   \n");
    printf("R  R   I   C      K  K \n");
    printf("R   R III   CCCC  K   K    Library\n");
    printf("\n");
    printf("Radio Imaging Code Kernels Library (v2.0.0)\n");
    printf("\n");

    printf("Running with %d MPI tasks\n", size);
  }
#else
  rank = 0;
  size = 1;

#ifdef _OPENMP
  num_threads = omp_get_max_threads();
#else
  num_threads = 1;
#endif //_OPENMP
#endif // USE_MPI

  xaxis = grid_size_x;
  yaxis = grid_size_y / size;

  int ndatasets = 1;
  strcpy(datapath_multi[0], "/Users/e.derubeis/hpc_imaging/data/hba_8hours_150_simulated.binMS/");

  strcpy(datapath, datapath_multi[0]);
  // Read metadata
  strcpy(filename, datapath);
  strcat(filename, metafile);
  pFile = fopen(filename, "r");
  fscanf(pFile, "%u", &Nmeasures);
  fscanf(pFile, "%ld", &Nvis);
  fscanf(pFile, "%d", &freq_per_chan);
  fscanf(pFile, "%d", &polarisations);
  fscanf(pFile, "%u", &Ntimes);
  fscanf(pFile, "%lf", &dt);
  fscanf(pFile, "%lf", &thours);
  fscanf(pFile, "%ld", &baselines);
  fscanf(pFile, "%lf", &uvmin);
  fscanf(pFile, "%lf", &uvmax);
  fscanf(pFile, "%lf", &wmin);
  fscanf(pFile, "%lf", &wmax);
  fscanf(pFile, "%d", &nweights1);
  fscanf(pFile, "%d", &nweights2);
  fscanf(pFile, "%d", &nweights3);
  fclose(pFile);

  Nvis = Nmeasures * freq_per_chan * polarisations;
  Nweights = nweights1 * nweights2 * nweights3;

  long nm_pe = (long)(Nmeasures / size);
  long remaining = Nmeasures % size;
  long startrow = rank * nm_pe;
  if (rank == size - 1)
    nm_pe = nm_pe + remaining;

  Nmeasures = nm_pe;
  Nvis = Nmeasures * freq_per_chan * polarisations;
  Nweights = nweights1 * nweights2 * nweights3;

  if (rank == 0)
  {
    printf("N. measurements %d\n", Nmeasures);
    printf("N. visibilities %ld\n", Nvis);
  }

  uu = (double *)calloc(Nmeasures, sizeof(double));
  vv = (double *)calloc(Nmeasures, sizeof(double));
  ww = (double *)calloc(Nmeasures, sizeof(double));
  weights = (float *)calloc(Nweights, sizeof(float));
  visreal = (float *)calloc(Nvis, sizeof(float));
  visimg = (float *)calloc(Nvis, sizeof(float));
  channels_ref = (float *)calloc(freq_per_chan, sizeof(float));

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
  fseek(pFile, startrow * nweights2 * nweights3 * sizeof(float), SEEK_SET);
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

  strcpy(filename, datapath);
  strcat(filename, channels);
  pFile = fopen(filename, "rb");
  fseek(pFile, 0, SEEK_SET);
  fread(channels_ref, freq_per_chan * sizeof(float), 1, pFile);
  fclose(pFile);

#ifdef VERBOSE
  // Add this debug print
  if (rank == 0)
  {
    printf("Read %d frequencies:\n", freq_per_chan);
    for (int i = 0; i < freq_per_chan; i++)
    {
      printf("  channels_ref[%d] = %f\n", i, channels_ref[i]);
    }
  }
#endif

  long size_of_grid = 2 * num_w_planes * xaxis * yaxis;

  double *grid;
  grid = (double *)calloc(size_of_grid, sizeof(double));
  double *gridss;
  gridss = (double *)calloc(size_of_grid, sizeof(double));

  double *image_real = (double *)calloc(xaxis * yaxis, sizeof(double));
  double *image_imag = (double *)calloc(xaxis * yaxis, sizeof(double));

#if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)

  // float weights_stokes_sum;
  float *visreal_stokes;
  float *visimg_stokes;
  float *weights_stokes;

  visreal_stokes = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  visimg_stokes = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));
  weights_stokes = (float *)malloc(Nmeasures * freq_per_chan * sizeof(float));

  stokes_collapse(
      Nmeasures,
      freq_per_chan,
      visreal,
      visimg,
      weights,
      visreal_stokes,
      visimg_stokes,
      weights_stokes);

#endif

  // float listfreq[] = {1.20237732e+08, 1.20286560e+08, 1.20335388e+08, 1.20384216e+08};

  for (int freq_index = 0; freq_index < freq_per_chan; freq_index++)
  {
    double *uu_in_m = (double *)malloc(Nmeasures * sizeof(double));
    double *vv_in_m = (double *)malloc(Nmeasures * sizeof(double));
    double *ww_in_m = (double *)malloc(Nmeasures * sizeof(double));

    // printf("channels ref [%d] = %f\n", freq_index, channels_ref[freq_index]);
    float wavelength = (float)(299792458.0) / channels_ref[freq_index];
    printf("wavelength = %f\n", wavelength);
    for (int i = 0; i < Nmeasures; i++)
    {
      uu_in_m[i] = uu[i] / (double)wavelength;
      // printf("%f \n", uu_in_m[i]);
      vv_in_m[i] = vv[i] / (double)wavelength;
      ww_in_m[i] = ww[i] / (double)wavelength;
    }

    gridding(
        rank,
        size,
        Nmeasures,
        uu_in_m,
        vv_in_m,
        ww_in_m,
        grid,
        gridss,
        MPI_COMM_WORLD,
        num_threads,
        grid_size_x,
        grid_size_y,
        w_support,
        num_w_planes,
        polarisations,
        freq_per_chan,
#if defined(WRITE_DATA)
        gridded_writedata1,
        gridded_writedata2,
#endif
#if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
        visreal_stokes,
        visimg_stokes,
        weights_stokes,
#else
        visreal,
        visimg,
        weights,
#endif
        uvmin,
        uvmax,
        freq_index);

    free(uu_in_m);
    free(vv_in_m);
    free(ww_in_m);
  }

  fftw_data(
      grid_size_x,
      grid_size_y,
      num_w_planes,
      num_threads,
      MPI_COMM_WORLD,
      size,
      rank,
#if defined(WRITE_DATA)
      fftfile_writedata1,
      fftfile_writedata2,
#endif
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
      num_threads,
      size,
      rank,
      MPI_COMM_WORLD);

  if (rank == 0)
    printf("End of main\n");

#ifdef USE_MPI
  MPI_Finalize();
#endif
}