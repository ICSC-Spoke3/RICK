#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
#include <omp.h>
#include "ricklib.h"
#ifdef FITSIO
#include "fitsio.h"
#endif

#define PI 3.14159265359

#define NFILES 100
#define FILENAMELENGTH 50

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))


/* FUNCTION FOR TIMINGS */
void write_timings(int rank, timing_t timing)
{
  double time_IO, time_gridding, time_fft, time_phase, time_total;
  
  MPI_Reduce(&timing.IO, &time_IO, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timing.gridding, &time_gridding, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timing.fft, &time_fft, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timing.phase, &time_phase, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timing.total, &time_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0)
    {
      printf("%40s time: %g sec\n", "I/O (reading)", time_IO);
      printf("%40s time: %g sec\n", "Gridding", time_gridding);
      printf("%40s time: %g sec\n", "FFT", time_fft);
      printf("%40s time: %g sec\n", "Phase correction", time_phase);
      printf("%40s time: %g sec\n", "Total", time_total);
    }
}


int main(int argc, char **argv)
{
  int rank;
  int size;

  // Define main filenames
  FILE *pFile;
  MPI_File pFile1;

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

  int Nmeasures;
  long Nvis;
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
  int grid_size_x = 4096;
  int grid_size_y = 4096;
  // Split Mesh size (auto-calculated)
  int local_grid_size_x;
  int local_grid_size_y;
  int xaxis;
  int yaxis;

  // Number of planes in the w direction
  int num_w_planes = 8;

  // Size of the convoutional kernel support
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

  /* Account for the case in which grid_size_y % size != 0 */
  long remy   = grid_size_y % size;
  long starty = rank * yaxis + (rank < remy ? rank : remy);
  yaxis       = yaxis + (rank < remy ? 1 : 0);
  
  int ndatasets = 1;
  strcpy(datapath_multi[0], "/u/glacopo/RICK_PMT/newgauss2noconj_t201806301100_SBL180.binMS/");
  //strcpy(datapath_multi[0], "/home/giovanni/RICK_LIBRARY/RICK/data/newgauss2noconj_t201806301100_SBL180.binMS/");

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
  fclose(pFile);

  Nvis = Nmeasures * freq_per_chan * polarisations;
  Nweights = Nmeasures * freq_per_chan * polarisations;

  long Nmeasures_tot = Nmeasures;
  long Nvis_tot      = Nvis;
  long Nweights_tot  = Nweights;
  
  long nm_pe    = (long)(Nmeasures / size);
  long rem      = Nmeasures % size;
  long startrow = rank * nm_pe + (rank < rem ? rank : rem);
  nm_pe         = nm_pe + (rank < rem ? 1 : 0);
  
  Nmeasures = nm_pe;
  Nvis = Nmeasures * freq_per_chan * polarisations;
  Nweights = Nmeasures * freq_per_chan * polarisations;

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

  /* DEFINE THE TIMINGS STRUCTURE */
  timing_t timing;
 
  if (rank == 0)
    printf("READING DATA WITH MPI-I/O\n");

  double total_start = WALLCLOCK_TIME;

  strcpy(filename, datapath);
  strcat(filename, ufile);

  int ierr;
  ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &pFile1);

  if (ierr != MPI_SUCCESS)
    {
      if (rank == 0)
        fprintf(stderr, "Error: Could not open file '%s' for reading.\n", filename);
    }

  MPI_Offset offset = sizeof(double) * startrow;

  MPI_File_read_at(pFile1, offset, uu, Nmeasures, MPI_DOUBLE, MPI_STATUS_IGNORE);

  MPI_File_close(&pFile1);

   
  strcpy(filename, datapath);
  strcat(filename, vfile);
    
  ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &pFile1);

  if (ierr != MPI_SUCCESS)
    {
      if (rank == 0)
        fprintf(stderr, "Error: Could not open file '%s' for reading.\n", filename);
    }

  MPI_File_read_at(pFile1, offset, vv, Nmeasures, MPI_DOUBLE, MPI_STATUS_IGNORE);

  MPI_File_close(&pFile1);

  strcpy(filename, datapath);
  strcat(filename, wfile);
    
  ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &pFile1);

  if (ierr != MPI_SUCCESS)
    {
      if (rank == 0)
        fprintf(stderr, "Error: Could not open file '%s' for reading.\n", filename);
    }

  MPI_File_read_at(pFile1, offset, ww, Nmeasures, MPI_DOUBLE, MPI_STATUS_IGNORE);

  MPI_File_close(&pFile1);

  strcpy(filename, datapath);
  strcat(filename, weightsfile);
    
  ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &pFile1);

  if (ierr != MPI_SUCCESS)
    {
      if (rank == 0)
        fprintf(stderr, "Error: Could not open file '%s' for reading.\n", filename);
    }

  MPI_Offset offset_w = sizeof(float) * (startrow * freq_per_chan * polarisations);
  
  MPI_File_read_at(pFile1, offset_w, weights, Nweights, MPI_FLOAT, MPI_STATUS_IGNORE);
    
  MPI_File_close(&pFile1);
 
  strcpy(filename, datapath);
  strcat(filename, visrealfile);
    
  ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &pFile1);

  if (ierr != MPI_SUCCESS)
    {
      if (rank == 0)
        fprintf(stderr, "Error: Could not open file '%s' for reading.\n", filename);
    }

  MPI_File_read_at(pFile1, offset_w, visreal, Nvis, MPI_FLOAT, MPI_STATUS_IGNORE);

  MPI_File_close(&pFile1);

  strcpy(filename, datapath);
  strcat(filename, visimgfile);
    
  ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &pFile1);

  if (ierr != MPI_SUCCESS)
    {
      if (rank == 0)
        fprintf(stderr, "Error: Could not open file '%s' for reading.\n", filename);
    }

  MPI_File_read_at(pFile1, offset_w, visimg, Nvis, MPI_FLOAT, MPI_STATUS_IGNORE);
  
  MPI_File_close(&pFile1);

  MPI_Barrier(MPI_COMM_WORLD);

  timing.IO += WALLCLOCK_TIME - total_start;
  
  long size_of_grid = 2 * num_w_planes * xaxis * yaxis;

  double *grid;
  grid = (double *)calloc(size_of_grid, sizeof(double));
  double *gridss;
  gridss = (double *)calloc(size_of_grid, sizeof(double));

  double *image_real = (double *)calloc(xaxis * yaxis, sizeof(double));
  double *image_imag = (double *)calloc(xaxis * yaxis, sizeof(double));

  double gridding_start = WALLCLOCK_TIME;
  
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
      visreal,
      visimg,
      weights,
      uvmin,
      uvmax);

  timing.gridding += WALLCLOCK_TIME - gridding_start;

  double fft_start = WALLCLOCK_TIME;
  
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

  timing.fft += WALLCLOCK_TIME - fft_start;

  double phase_start = WALLCLOCK_TIME;
  
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

  timing.phase += WALLCLOCK_TIME - phase_start;
  
  if (rank == 0)
    printf("End of main\n");

  timing.total = WALLCLOCK_TIME - total_start;

  write_timings(rank, timing);
  
#ifdef USE_MPI
  MPI_Finalize();
#endif
}
