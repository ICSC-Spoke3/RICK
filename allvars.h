/* file to store global variables*/

#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
#include <mpi.h>

#ifdef FITSIO
#include "fitsio.h"
#endif

#if defined (_OPENMP)
#include <omp.h>
#endif



#if defined(USE_FFTW) && !defined(CUFFTMP) // use MPI fftw
#include <fftw3-mpi.h>
#endif

#if defined(ACCOMP)               
#include "w-stacking_omp.h"
#else
#include "w-stacking.h"
#endif 


#if defined(NVIDIA)
#include <cuda_runtime.h>
#endif

#include "fft.h"
#include "numa.h"
#include "timing.h"
#include "errcodes.h"

#define PI 3.14159265359
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define NOVERBOSE
#define NFILES 100

#define NAME_LEN 50
#define LONGNAME_LEN 1000


#define REDUCE_MPI  0
#define REDUCE_RING 1

#if defined(DEBUG)
#define dprintf(LEVEL, T, t, ...) if( (verbose_level >= (LEVEL)) &&	\
				      ( ((t) ==-1 ) || ((T)==(t)) ) ) {	\
    printf(__VA_ARGS__); fflush(stdout); }

#else
#define dprintf(...)
#endif

typedef double double_t;
#if defined(DOUBLE_PRECISION)
typedef double float_t;
#else
typedef float float_t;
#endif

typedef unsigned int       myuint;
typedef unsigned long long myull;


extern struct io
{
	FILE * pFile;
        FILE * pFile1;
        FILE * pFilereal;
        FILE * pFileimg;
} file;

extern struct ip
{
	char ufile[NAME_LEN];
  	char vfile[NAME_LEN];
  	char wfile[NAME_LEN];
  	char weightsfile[NAME_LEN];
  	char visrealfile[NAME_LEN];
  	char visimgfile[NAME_LEN];
  	char metafile[NAME_LEN];
        char paramfile[NAME_LEN];
} in;

extern struct op
{
	char outfile[NAME_LEN];
        char outfile1[NAME_LEN];
        char outfile2[NAME_LEN];
        char outfile3[NAME_LEN];
        char fftfile[NAME_LEN];
        char gridded_writedata1[NAME_LEN];
        char gridded_writedata2[NAME_LEN];
        char fftfile_writedata1[NAME_LEN];
        char fftfile_writedata2[NAME_LEN];
        char fftfile2[NAME_LEN];
        char fftfile3[NAME_LEN];
        char logfile[NAME_LEN];
        char extension[NAME_LEN];
        char timingfile[NAME_LEN];
  
} out, outparam;

extern struct meta
{

  myuint   Nmeasures;
  myull   Nvis;
  myuint   Nweights;
  myuint   freq_per_chan;
  myuint   polarisations;
  myuint   Ntimes;
  double dt;
  double thours;
  myuint   baselines;
  double uvmin;
  double uvmax;
  double wmin;
  double wmax;
} metaData;


extern struct parameter
{
  int  num_threads;
  int  ndatasets;
  char datapath_multi[NFILES][LONGNAME_LEN];
  int  grid_size_x;
  int  grid_size_y;
  int  num_w_planes;
  int  w_support;
  int  reduce_method;
} param;

extern struct fileData
{
        double * uu;
        double * vv;
        double * ww;
        float * weights;
        float * visreal;
        float * visimg;
}data;


extern char filename[LONGNAME_LEN], buf[NAME_LEN], num_buf[NAME_LEN];
extern char datapath[LONGNAME_LEN];
extern int  xaxis, yaxis;
extern int  rank;
extern int  size;
extern myuint nsectors;
extern myuint startrow;
extern double_t resolution, dx, dw, w_supporth;

extern myuint **sectorarray;
extern myuint  *histo_send;
extern int    verbose_level; 


extern myuint    size_of_grid;
extern double_t *grid_pointers, *grid, *gridss, *gridss_real, *gridss_img, *gridss_w, *grid_gpu, *gridss_gpu;

extern MPI_Comm MYMPI_COMM_WORLD;
extern MPI_Win  slabwin;
