

#if defined( USE_FFTW ) && !defined( CUFFTMP )

#if defined( HYBRID_FFTW )
#define FFT_INIT    { fftw_init_threads(); fftw_mpi_init();}
#define FFT_CLEANUP fftw_cleanup_threads()
#else
#define FFT_INIT    fftw_mpi_init()
#define FFT_CLEANUP fftw_cleanup()
#endif

#else

#define FFT_INIT
#define FFT_CLEANUP

#endif
