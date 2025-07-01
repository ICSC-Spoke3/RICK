#include <mpi.h>
#include <time.h>
#include <limits.h>

#ifdef _OPENMP
#define HYBRID_FFTW
#endif

typedef unsigned int       myuint;
typedef unsigned long long myull;

int MPI_File_read_at_custom(MPI_File, MPI_Offset, void *,
			    myull, MPI_Datatype,
			    MPI_Status *);

int MPI_Sendrecv_custom(void *, myull, MPI_Datatype,
			int, int, void *, myull,
			MPI_Datatype, int, int,
			MPI_Comm, MPI_Status *);

void gridding(
    int,
    int,
    myull,
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
    double,
    myull);

    void fftw_data(
    int,
    int,
    int,
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
    double *);

    void phase_correction(
    double *,
    double *,
    double *,
    int,
    int,
    int,
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
  double check;      // time spent to consturct bucket_sort[] array
  double bucket;     // time spent to perform all bucket sortings with cache misses
  double communication; // time spent in P2P communication for distributing visibilities
  double fft;        // time spent in FFT Contains MPI-I/O when WRITE_DATA is defined;
  double phase;      // time spent in phase correction Contains MPI-I/O for image writing
  double total;      // total runtime
} timing_t;

extern timing_t timing;      // wall-clock process timing


#define MPI_COUNT_T MPI_UNSIGNED_LONG_LONG
#define SIGMA_FACTOR_Y 6.0

/* If nothing is defined, the HeFFTe will go for FFTW backend */

#if defined(NVIDIA) /////
#define BACKEND Heffte_BACKEND_CUFFT
#elif defined(AMD)
#define BACKEND Heffte_BACKEND_ROCFFT
#elif defined(INTEL)
#define BACKEND Heffte_BACKEND_MKL
#else
#define BACKEND Heffte_BACKEND_FFTW
#endif /////

/* If NVIDIA or AMD or INTEL are defined, switch on OpenMP acceleration */

#if defined(NVIDIA) || defined(AMD) || defined(INTEL)
#define OMP_ACCELERATION
#endif //NVIDIA OR AMD OR INTEL

extern double* grid;

#if defined(OMP_ACCELERATION)
extern int devID;
#pragma omp declare target(grid)
#endif //OMP_ACCELERATION
