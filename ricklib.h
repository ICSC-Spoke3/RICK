#include <mpi.h>

#ifdef _OPENMP
#define HYBRID_FFTW
#endif

#if !defined(RICK_GPU) && defined(HYBRID_FFTW)
#include <fftw3-mpi.h>
#define FFT_INIT             \
    {                        \
        fftw_init_threads(); \
        fftw_mpi_init();     \
    }
#define FFT_CLEANUP fftw_cleanup_threads()
#elif !defined(RICK_GPU) && !defined(HYBRID_FFTW)
#include <fftw3-mpi.h>
#define FFT_INIT fftw_mpi_init()
#define FFT_CLEANUP fftw_cleanup()
#endif

void gridding(
    int,
    int,
    int,
    double *,
    double *,
    double *,
    double *,
    double *,
    MPI_Comm,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
#if defined(WRITE_DATA)
    char *,
    char *,
#endif
#if defined(STOKESI) || defined(STOKESQ) || defined(STOKESU)
    float *,
    float *,
    float *,
#else
    float *,
    float *,
    float *,
#endif
    double,
    double,
    int);

void fftw_data(
    int,
    int,
    int,
    int,
    MPI_Comm,
    int,
    int,
#if defined(WRITE_DATA)
    char *,
    char *,
#endif
    double *,
    double *);

void phase_correction(
    double *,
    double *,
    double *,
    int,
    int,
    int,
    double,
    double,
    double,
    double,
    int,
    int,
    int,
    MPI_Comm);

void stokes_collapse(
    unsigned int,
    int,
    float *,
    float *,
    float *,
    float *,
    float *,
    float *);