#include <mpi.h>
#include <time.h>

#ifdef _OPENMP
#define HYBRID_FFTW
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
    float *,
    float *,
    float *,
    double,
    double);

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

/* TIMINGS */

#define WALLCLOCK_TIME ({ struct timespec myts; (clock_gettime( CLOCK_REALTIME, &myts ), (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9);})


typedef struct {
  double IO;         // time spent in IO (MPI-I/O)
  double gridding;   // time spent in gridding (wstack + reduce (so far)) Contains MPI-I/O when WRITE_DATA is defined;
  
  double fft;        // time spent in FFT Contains MPI-I/O when WRITE_DATA is defined;
  double phase;      // time spent in phase correction Contains MPI-I/O for image writing
  double total;      // total runtime
} timing_t;

extern timing_t timing;      // wall-clock process timing
