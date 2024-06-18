#include <stdio.h>
#include "allvars.h"
#include "proto.h"


void write_result( void )
{
  
  if (rank == 0)
    {
      
      int Ntasksmpi;
      
      printf("%40s time: %g sec\n", "Setup", timing_wt.setup);
      printf("%40s time : %g sec\n", "Process", timing_wt.gridding);      
      printf("%40s time : %g sec\n", "Kernel", timing_wt.kernel);     
      printf("%40s time : %g sec\n", "Array Composition", timing_wt.compose);           
      printf("%40s time : %g sec\n", "Reduce", timing_wt.reduce);
      
     #if defined(USE_FFTW)
     #if !defined(CUFFTMP)
      printf("%40s time : %g sec\n", "FFTW", timing_wt.fftw);
     #else
      printf("%40s time : %g sec\n", "cufftMP", timing_wt.cufftmp);
     #endif //CUFFTMP
      printf("%40s time : %g sec\n", "Phase", timing_wt.phase);
     #endif //USE_FFTW

     #if defined(WRITE_IMAGE)
      printf("%40s time : %g sec\n", "Image writing", timing_wt.write);
     #endif
      
      printf("%40s time : %g sec\n", "TOT", timing_wt.total);
  
      file.pFile = fopen (out.timingfile, "w");


     #if defined(USE_FFTW)
     #if !defined(CUFFTMP)
      fprintf(file.pFile, "%g %g %g %g %g %g %g\n",
	      timing_wt.setup, timing_wt.kernel, timing_wt.compose,
	      timing_wt.reduce, timing_wt.fftw, timing_wt.phase, timing_wt.total);
     #else
      fprintf(file.pFile, "%g %g %g %g %g %g %g\n",
	      timing_wt.setup, timing_wt.kernel, timing_wt.compose,
	      timing_wt.reduce, timing_wt.cufftmp, timing_wt.phase, timing_wt.total);
     #endif //CUFFTMP
      fclose(file.pFile);
     #endif //USE_FFTW
    }

  return;
}
  
