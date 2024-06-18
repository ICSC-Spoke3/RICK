#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

int main( int argc, char **argv )
{

  if ( argc < 3 ) { printf("I'm expecting 3 args\n"); return 1; }
  
  unsigned int n = atoi(*(argv+1));
  
  FILE *file1    = fopen( *(argv+2), "r" );
  FILE *file2    = fopen( *(argv+3), "r" );

  double *array1 = (double*)calloc( n, sizeof(double));
  double *array2 = (double*)calloc( n, sizeof(double));
  
  fread( array1, sizeof(double), n, file1 );
  fread( array2, sizeof(double), n, file2 );

  fclose( file1 );
  fclose( file2 );

  typedef struct { double min, max, avg, stddev; } info_t;

  info_t relative = {1e10, 0, 0, 0};
  info_t absolute = {1e10, 0, 0, 0};

  double abs_max_track[2];
  double rel_max_track[2];

  unsigned int first_zero = 0;
  unsigned int second_zero = 0;


  
  for( unsigned int i = 0; i < n; i++ )
    {
      double dev = (array1[i] - array2[i]);

      if( array1[i]==0 && array2[i]!= 0)
	{
	  first_zero++;
	  fprintf(stderr, "[A] %g\n", array2[i] );
	}
	
      if ( array1[i]!=0 && array2[i]== 0)
      {
	second_zero ++;
	fprintf(stderr, "[B] %g\n", array1[i] );
      }

      absolute.min     = ( fabs(dev) < fabs(absolute.min) ? dev : absolute.min );
      if( fabs(dev) > fabs(absolute.max) )
	{
	  absolute.max     = dev;
	  abs_max_track[0] = array1[i];
	  abs_max_track[1] = array2[i];
	}
	  
      absolute.avg    += dev;
      absolute.stddev += dev*dev;

      double val = ( array1[i] > 0 ? array1[i] : (array2[i] > 0 ? array2[i] : 0) );

      dev = ( val > 0 ? dev/val : 0 );
      
      relative.min     = ( fabs(dev) < fabs(relative.min) ? dev : relative.min );
      if( fabs(dev) > fabs(relative.max) )
	{
	  relative.max     = dev;
	  rel_max_track[0] = array1[i];
	  rel_max_track[1] = array2[i];
	}

      relative.avg    += dev;
      relative.stddev += dev*dev;
      
    }
  

  absolute.avg /= n;
  relative.avg /= n;

  absolute.stddev = sqrt(absolute.stddev / (n-1) - absolute.avg*absolute.avg);
  relative.stddev = sqrt(relative.stddev / (n-1) - relative.avg*relative.avg);

  printf("%u times (%g %%) first file has zero entries corresponding to non-zero in second file\n"
	 "%u times (%g %%) second file has zero entries corresponding to non-zero in first file\n\n",	 
	 first_zero, (double)first_zero/n*100,
	 second_zero, (double)second_zero/n*100 );
  
  printf("Report on *absolute* differences:\n"
	 "%22s : %8.6g\n"
	 "%23s : %8.6g\n"
	 "%22s : %8.6g\n"
	 "%22s : %8.6g (@ %8.6g %8.6g)\n\n",
	 "AVG difference", absolute.avg,
	 "STD DEV of difference", absolute.stddev,
	 "MIN difference", absolute.min,
	 "MAX difference", absolute.max, abs_max_track[0], abs_max_track[1] );
  
  printf("Report on *relative* differences:\n"
	 "%22s : %8.6g\n"
	 "%23s : %8.6g\n"
	 "%22s : %8.6g\n"
	 "%22s : %8.6g (@ %8.6g %8.6g)\n",	 
	 "AVG difference", relative.avg,
	 "STD DEV of difference", relative.stddev,
	 "MIN difference", relative.min,
	 "MAX difference", relative.max, rel_max_track[0], rel_max_track[1] );
  
  free( array2 );
  free( array1 );
  return 0;
}
