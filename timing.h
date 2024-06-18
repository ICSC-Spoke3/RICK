
#include <time.h>

#define CPU_TIME_wt ({ struct timespec myts; (clock_gettime( CLOCK_REALTIME, &myts ), (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9);})
#define CPU_TIME_pr ({ struct timespec myts; (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &myts ), (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9);})
#define CPU_TIME_th ({ struct timespec myts; (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9);})


#if defined(_OPENMP)
#define TAKE_TIME_START( T ) {			\
    wt_timing.T = CPU_TIME_wt;			\
    pr_timing.T = CPU_TIME_pr; }

#define TAKE_TIME_STOP( T ) {			\
    pr_timing.T = CPU_TIME_pr - pr_timing.T;	\
    wt_timing.T = CPU_TIME_wt - wt_timing.T; }

#define TAKE_TIME( Twt, Tpr ) { Twt = CPU_TIME_wt; Tpr = CPU_TIME_pr; }
#define ADD_TIME( T, Twt, Tpr ) {		\
    pr_timing.T += CPU_TIME_pr - Tpr;		\
    wt_timing.T += CPU_TIME_wt - Twt;		\
    Tpr = CPU_TIME_pr; Twt = CPU_TIME_wt; }

#else

#define TAKE_TIME_START( T ) wt_timing.T = CPU_TIME_wt

#define TAKE_TIME_STOP( T )  wt_timing.T = CPU_TIME_wt - wt_timing.T

#define TAKE_TIME( Twt, ... ) Twt = CPU_TIME_wt;
#define ADD_TIME( T, Twt, ... ) { wt_timing.T += CPU_TIME_wt - Twt; Twt = CPU_TIME_wt;}

#endif

#define TAKE_TIMEwt_START( T) wt_timing.T = CPU_TIME_wt
#define TAKE_TIMEwt_STOP( T) wt_timing.T = CPU_TIME_wt - wt_timing.T
#define TAKE_TIMEwt( Twt ) Twt = CPU_TIME_wt;
#define ADD_TIMEwt( T, Twt ) { wt_timing.T += CPU_TIME_wt - Twt; Twt = CPU_TIME_wt; }


#if defined(__GNUC__) && !defined(__ICC) && !defined(__INTEL_COMPILER)
#define PRAGMA_IVDEP _Pragma("GCC ivdep")
#else
#define PRAGMA_IVDEP _Pragma("ivdep")
#endif

#define STRINGIFY(a) #a
#define UNROLL(N) _Pragma(STRINGIFY(unroll(N)))


#define CPU_TIME_tr ({ struct timespec myts; (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9);})

#define CPU_TIME_rt ({ struct timespec myts; (clock_gettime( CLOCK_REALTIME, &myts ), (double)myts.tv_sec + (double)myts.tv_nsec * 1e-9);})

#define CPU_TIME_STAMP(t, s) { struct timespec myts; clock_gettime( CLOCK_REALTIME, &myts ); printf("STAMP t %d %s - %ld %ld\n", (t), s, myts.tv_sec, myts.tv_nsec);}




typedef struct {
  double setup;      // time spent in initialization, init()
  double init;       // time spent in initializing arrays
  double gridding;   // time spent in gridding;
  double mpi;        // time spent in mpi communications
  double fftw;       //
  double cufftmp;    //
  double kernel;     //
  double mmove;      //
  double reduce;     //
  double reduce_mpi; //
  double reduce_sh ; //
  double compose;    //
  double phase;      //
  double write;      //
  double total;
  double offload;} timing_t;

extern timing_t timing_wt;      // wall-clock process timing, at Task 0


extern double start_tot;
extern double reduce_shmem_time;
extern double reduce_mpi_time;
