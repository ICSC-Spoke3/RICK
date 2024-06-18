/* function declaration */
#include <mpi.h>

/* init.c */

void init(int i);
void op_filename();
void read_parameter_file(char *);
void fileName(char datapath[900], char file[30]);
void readMetaData(char fileLocal[1000]);
void metaData_calculation();
void allocate_memory();
void readData();

#ifdef __cplusplus
extern "C" {
  void shutdown_wstacking( int, char *, char *, int);
}


#else
void shutdown_wstacking( int, char *, char *, int);
#endif

#ifdef __cplusplus
extern "C" {
  void gridding          (void);
  void gridding_data     (void);
  void write_gridded_data(void);
}

#else
/*  gridding.c */

void gridding          (void);
void gridding_data     (void);
void write_gridded_data(void);
#endif

#ifdef __cplusplus
extern "C" {
  void fftw_data();

 #ifdef CUFFTMP
  void cuda_fft( int, int, int, int, int, double*, double*, int, MPI_Comm );
 #endif
  
  void write_fftw_data();
  void write_result();
}
#else

/* fourier_transform.c */

void fftw_data();

#ifdef CUFFTMP
void cuda_fft( int, int, int, int, int, double*, double*, int, MPI_Comm );
#endif

void write_fftw_data();
void write_result();
#endif
