

#include "allvars.h"
#include "proto.h"

void shutdown_wstacking( int errcode, char *message, char *fname, int linenum )
{
  if ( ( rank == 0 ) &&
       ( errcode > 0 ) )
    printf("an error occurred at line %d of file %s\n"
	   "error message is: %s\n"
	   "error code is: %d\n",
	   linenum, fname, message, errcode);
  
  FFT_CLEANUP;

  if( param.reduce_method == REDUCE_RING )
    numa_shutdown(rank, 0, &MYMPI_COMM_WORLD, &Me);


  if( grid_pointers != NULL )
    free( grid_pointers );
  
  MPI_Finalize();
  
  if( errcode > 0 )
    exit(errcode); 
}


// Main Code
int main(int argc, char * argv[])
{


  if(argc > 1)
    {
      strcpy(in.paramfile, argv[1]);
    }
  else
    {
      fprintf(stderr, "please, specify a parameter file as first argument at command line\n");
      exit(1);
    }
 
  /* Initializing MPI Environment */

  double time_tot = CPU_TIME_wt;
  
 #ifdef _OPENMP
  {
    int thread_level;
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &thread_level );
    if ( thread_level < MPI_THREAD_FUNNELED )
      {
	printf("the supported thread level is smaller than MPI_THREAD_FUNNELLED (%d vs %d)\n",
	       thread_level, MPI_THREAD_FUNNELED);

       #if defined(HYBRID_FFTW)
	shutdown_wstacking(NO_THREADS_SUPPORT, "FFTW with threads could not be supported. Better to stop here.",
		 __FILE__, __LINE__ );

       #endif
      }
  }
 #else
  MPI_Init(&argc,&argv);
 #endif
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if(rank == 0)
  {
    printf("\n");
    printf("RRR   III   CCCC  K   K\n");
    printf("R  R   I   C      K  K \n");
    printf("RRR    I   C      KK   \n");
    printf("R  R   I   C      K  K \n");
    printf("R   R III   CCCC  K   K\n");
    printf("\n");
    printf("Radio Imaging Code Kernels (v2.0.0)\n");
    printf("\n");

    printf("Running with %d MPI tasks\n", size);
  }
  MPI_Comm_dup(MPI_COMM_WORLD, &MYMPI_COMM_WORLD);

  FFT_INIT;    

 #ifdef ACCOMP
  if ( rank == 0 ) {
    if (0 == omp_get_num_devices()) {

      shutdown_wstacking(NO_ACCELERATORS_FOUND, "No accelerators found", __FILE__, __LINE__ );
    }
    printf("Number of available GPUs %d\n", omp_get_num_devices());
   #ifdef NVIDIA
    prtAccelInfo();
   #endif
  }  
 #endif

#ifdef FITSIO
  fitsfile *fptreal;
  fitsfile *fptrimg;
  int status;
  char testfitsreal[NAME_LEN] = "parallel_real.fits";
  char testfitsimag[NAME_LEN] = "parallel_img.fits";

  myuint naxis = 2;
  myuint naxes[2] = { grid_size_x, grid_size_y };
 #endif

  
  /* Reading Parameter file */

  read_parameter_file(in.paramfile);
  
  if ( param.num_threads == 0 )
    {
     #if defined(_OPENMP)
     #pragma omp parallel
     #pragma omp single
      param.num_threads = omp_get_num_threads();
     #else
      param.num_threads = 1;
     #endif

      if( rank == 0 )
      printf("number of threads set to %d\n", param.num_threads);
    }

  
  for(int ifiles=0; ifiles<param.ndatasets; ifiles++)
    {
      if(rank == 0)
	printf( "\nDataset %d\n", ifiles);
     
      /*INIT function */
      init(ifiles);

      /* GRIDDING function */
      gridding();

      /* WRITE_GRIDDED_DATA function */
      write_gridded_data();

      /* FFTW_DATA function */
      fftw_data();

      /* WRITE_FFTW_DATA function */
      write_fftw_data();

      if(rank == 0)
	printf("*************************************************************\n"); 

    }

  /* WRITE_RESULT function */
  timing_wt.total = CPU_TIME_wt - time_tot;
  write_result();
  
  shutdown_wstacking(0, NULL, 0, 0);

  return 0;
}
