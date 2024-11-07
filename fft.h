#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
#include <omp.h>  
#include <fftw3-mpi.h>

#ifdef _OPENMP
#define HYBRID_FFTW
#endif

#define PI 3.14159265359

#if defined( HYBRID_FFTW )
#define FFT_INIT    { fftw_init_threads(); fftw_mpi_init();}
#define FFT_CLEANUP fftw_cleanup_threads()
#else
#define FFT_INIT    fftw_mpi_init()
#define FFT_CLEANUP fftw_cleanup()
#endif

