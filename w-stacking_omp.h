#ifndef W_PROJECT_H_
#define W_PROJECT_H_
#endif

#define NWORKERS -1    //100
#define NTHREADS 32
#define PI 3.14159265359
#define REAL_TYPE double

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
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

int test(int nnn);

double gauss_kernel_norm(
  double norm,
  double std22,
  double u_dist,
  double v_dist);

#ifdef __cplusplus
extern "C" {
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
}
#else
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
#endif

#ifdef ACCOMP
#ifdef NVIDIA
void getGPUInfo(int);
#endif
#pragma omp declare target (gauss_kernel_norm)
#endif

#ifdef NVIDIA
#ifdef __cplusplus
extern "C" {
#endif

#ifndef PRTACCELINFO_H
#define PRTACCELINFO_H

void prtAccelInfo(int iaccel);
/**<
 * @brief Print some basic info of an accelerator.
 *
 * Strictly speaking, \c prtAccelInfo() can only print the basic info of an
 * Nvidia CUDA device.
 *
 * @param iaccel The index of an accelerator.
 *
 * @return \c void.
 */

#endif

#ifdef __cplusplus
}
#endif
#endif
