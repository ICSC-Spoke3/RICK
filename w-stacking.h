#ifndef W_PROJECT_H_
#define W_PROJECT_H_

#define NWORKERS -1    //100
#define NTHREADS 32
#define PI 3.14159265359
#define REAL_TYPE double

#include <mpi.h>

#ifdef __CUDACC__
extern "C"
#endif

#ifdef __cplusplus
extern "C" {
void wstack(
     long long unsigned,
     unsigned int,
     unsigned int,
     unsigned int,
     double*,
     double*,
     double*,
     float*,
     float*,
     float*,
     double,
     double,
     int,
     long long unsigned,
     long long unsigned,
     double*,
     int,
     int,
     cudaStream_t);
}
#else 
void wstack(
     int,
     unsigned int,
     unsigned int,
     unsigned int,
     double*,
     double*,
     double*,
     float*,
     float*,
     float*,
     double,
     double,
     int,
     int,
     int,
     double*,
     int,
     int);
#endif



#ifdef __CUDACC__
extern "C"
#endif
int test(int nnn);

#ifdef __CUDACC__
extern "C"
#endif
void phase_correction(
     double*,
     double*,
     double*,
     int,
     int,
     int,
     int,
     int,
     double,
     double,
     double,
     int,
     int);

double gauss_kernel_norm(
  double norm,
  double std22,
  double u_dist,
  double v_dist);

#ifdef ACCOMP
#pragma omp declare target (gauss_kernel_norm)
#endif

#ifdef __CUDACC__
extern "C"
#endif

void cuda_fft(
	int,
	int,
	int,
	int,
	int,
	double*,
	double*,
	int,
	MPI_Comm);


#endif
