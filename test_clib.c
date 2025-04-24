#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
#include <omp.h>
#include "ricklib.h"
#ifdef FITSIO
#include "fitsio.h"
#endif

#define PI 3.14159265359

#define NFILES 100
#define FILENAMELENGTH 50

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))


/* Struct for domain decomposition */
typedef struct {
    int tile_id;
    int start_x; 
    int start_y;
    int size_x;
    int size_y; 
} subGrid;

int exchange_Npts ( myull *to_be_sent, myull *to_be_recv, int Me, int Ntasks, MPI_Comm COMM )
{
  // find the log2P+1
  //
  int Ptasks, Top;
  for( Ptasks = 0; (1 << Ptasks) < Ntasks; Ptasks++ )
    ;
  Top = (1 << Ptasks);

 #ifdef DEBUG
  if ( Me == 0 )
    printf ( "%d communication levels\n", Top );
 #endif
  
  for( int ngrp = 1; ngrp < Top; ngrp++ )
    {
      int target = Me ^ ngrp;

      if(target < Ntasks)
        {
	 #define SHAKE_HANDS 0

	 #ifdef DEBUG
	  if ( Me < target )
	    printf("Task %d and %d are setting up communication\n",
		   Me, target );
	 #endif
	  
	  MPI_Sendrecv( &to_be_sent[target], 1, MPI_COUNT_T, target, SHAKE_HANDS,
                        &to_be_recv[target], 1, MPI_COUNT_T, target, SHAKE_HANDS, COMM, MPI_STATUS_IGNORE );
	  
        }

    }
  
  
}

int exchange_double ( myull *to_be_sent, myull *to_be_recv, double *data_to_be_sent, double *data_to_be_recv, int Me, int Ntasks, MPI_Comm COMM )
{
  
  // find the log2P+1
  //
  int Ptasks, Top;
  for( Ptasks = 0; (1 << Ptasks) < Ntasks; Ptasks++ )
    ;
  Top = (1 << Ptasks);

 #ifdef DEBUG
  if ( Me == 0 )
    printf ( "%d communication levels\n", Top );
 #endif
  
  for( int ngrp = 1; ngrp < Top; ngrp++ )
    {
      int target = Me ^ ngrp;

      if(target < Ntasks)
        {
	  #define SEND_DATA 1

	 #ifdef DEBUG
	  if ( Me < target )
	    printf("Task %d and %d are setting up communication\n",
		   Me, target );
	 #endif
	  
	  
	  if ( (to_be_sent[target] > 0) ||
	       (to_be_recv[target] > 0) )
	    {
	     #ifdef DEBUG
	      if ( Me < target )
		printf("\tTask %d *-- %llu --> %d\n"
		       "\tTask %d <-- %llu --* %d\n",
		       Me, (unsigned long long)to_be_sent[target], target,
		       Me, (unsigned long long)to_be_recv[target], target );
	     #endif
	      
	      myull offset   = 0;
	      myull offset_r = 0;
	      for ( int t = 0; t < target; t++ )
		{
		  offset   += to_be_sent[t];
		  offset_r += to_be_recv[t];
		}

	     #ifdef DEBUG
	      printf("[OFFSET]: %d <--> %d, send offset %llu, recv offset %llu\n",
		     Me, target, offset, offset_r);
	     #endif
		
	      
	      MPI_Sendrecv( &data_to_be_sent[offset], to_be_sent[target]*sizeof(double), MPI_BYTE, target, SEND_DATA,
			    &data_to_be_recv[offset_r], to_be_recv[target]*sizeof(double), MPI_BYTE, target, SEND_DATA, COMM, MPI_STATUS_IGNORE );
	      
	      
	    }
        }

    }

  myull offset_mine      = 0;
  myull offset_recv_mine = 0;
  for (int i=0; i<Ntasks; i++)
    {
      offset_mine      += (i < Me ? to_be_sent[i] : 0);
      offset_recv_mine += to_be_recv[i];
    }

 #ifdef DEBUG
  printf("Task %d, offset_mine = %llu, offset_recv_mine = %llu, to_be_sent to me = %llu\n", Me, offset_mine, offset_recv_mine, to_be_sent[Me]);
 #endif

  memcpy(data_to_be_recv + offset_recv_mine, data_to_be_sent + offset_mine, to_be_sent[Me]*sizeof(double));
  
  
}

int exchange_float ( myull *to_be_sent, myull *to_be_recv, float *data_to_be_sent, float *data_to_be_recv, int Me, int Ntasks, MPI_Comm COMM )
{
  
  // find the log2P+1
  //
  int Ptasks, Top;
  for( Ptasks = 0; (1 << Ptasks) < Ntasks; Ptasks++ )
    ;
  Top = (1 << Ptasks);

 #ifdef DEBUG
  if ( Me == 0 )
    printf ( "%d communication levels\n", Top );
 #endif
  
  for( int ngrp = 1; ngrp < Top; ngrp++ )
    {
      int target = Me ^ ngrp;

      if(target < Ntasks)
        {
	  #define SEND_DATA 1

	 #ifdef DEBUG
	  if ( Me < target )
	    printf("Task %d and %d are setting up communication\n",
		   Me, target );
	 #endif
	  
	  if ( (to_be_sent[target] > 0) ||
	       (to_be_recv[target] > 0) )
	    {
	     #ifdef DEBUG
	      if ( Me < target )
		printf("\tTask %d *-- %llu --> %d\n"
		       "\tTask %d <-- %llu --* %d\n",
		       Me, (unsigned long long)to_be_sent[target], target,
		       Me, (unsigned long long)to_be_recv[target], target );
	     #endif
	      
	      myull offset   = 0;
	      myull offset_r = 0;
	      for ( int t = 0; t < target; t++ )
		{
		  offset   += to_be_sent[t];
		  offset_r += to_be_recv[t];
		}

	     #ifdef DEBUG
	      printf("[OFFSET]: %d <--> %d, send offset %llu, recv offset %llu\n",
		     Me, target, offset, offset_r);
	     #endif
		
	      
	      MPI_Sendrecv( &data_to_be_sent[offset], to_be_sent[target]*sizeof(float), MPI_BYTE, target, SEND_DATA,
			    &data_to_be_recv[offset_r], to_be_recv[target]*sizeof(float), MPI_BYTE, target, SEND_DATA, COMM, MPI_STATUS_IGNORE );
	      
	      
	    }
        }

    }

  myull offset_mine      = 0;
  myull offset_recv_mine = 0;
  for (int i=0; i<Ntasks; i++)
    {
      offset_mine      += (i < Me ? to_be_sent[i] : 0);
      offset_recv_mine += to_be_recv[i];
    }

 #ifdef DEBUG
  printf("Task %d, offset_mine = %llu, offset_recv_mine = %llu, to_be_sent to me = %llu\n", Me, offset_mine, offset_recv_mine, to_be_sent[Me]);
 #endif

  memcpy(data_to_be_recv + offset_recv_mine, data_to_be_sent + offset_mine, to_be_sent[Me]*sizeof(float));
  
  
}

void io_read(int rank, char *filename, char *datapath, MPI_File File, char *rfiles, char *data, MPI_Offset offset, unsigned long long Ndata)
{
  strcpy(filename, datapath);
  strcat(filename, rfiles);

  int ierr;
  ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &File);

  if (ierr != MPI_SUCCESS)
    {
      if (rank == 0)
	fprintf(stderr, "Error: Could not open file '%s' for reading.\n", filename);
    }

  MPI_File_read_at(File, offset, data, Ndata, MPI_BYTE, MPI_STATUS_IGNORE);
      
  MPI_File_close(&File);
      
  

  MPI_Barrier(MPI_COMM_WORLD);
}


void compute_gaussian_1d_decomp(int N_x, int N_y,
                                  int N_P, int rank,
                                  int *start_x, int *start_y,
                                  int *size_x, int *size_y)
{
    // X axis is not decomposed
    *start_x = 0;
    *size_x = N_x;

    // Y-axis Gaussian decomposition
    double center_y = (double)(N_y - 1) / 2.0;
    double sigma_y = (N_y / (double)N_P) * SIGMA_FACTOR_Y;

    double *weights_y = (double *)malloc(N_y * sizeof(double));
    double total_weight_y = 0.0;

    for (int iy = 0; iy < N_y; iy++) {
        double dy = iy - center_y;
        weights_y[iy] = exp(-(dy * dy) / (2.0 * sigma_y * sigma_y));
        total_weight_y += weights_y[iy];
    }

    int *y_boundaries = (int *)malloc((N_P + 1) * sizeof(int));
    y_boundaries[0] = 0;

    double cumulative_y = 0.0;
    int cur_tile = 1;

    for (int iy = 0; iy < N_y; iy++) {
        cumulative_y += weights_y[iy];
        if (cur_tile < N_P &&
            cumulative_y >= (total_weight_y * cur_tile) / N_P)
        {
            y_boundaries[cur_tile++] = iy + 1;
        }
    }
    y_boundaries[N_P] = N_y;

    *start_y = y_boundaries[rank];
    *size_y  = y_boundaries[rank + 1] - y_boundaries[rank];

    free(weights_y);
    free(y_boundaries);

    if (rank == 0) {
        printf("1D Gaussian decomposition on Y (σ_y = %.2f × tile_y)\n", SIGMA_FACTOR_Y);
    }
}

void collect_decomposition(int rank, int size,
                             int start_x, int start_y,
                             int size_x, int size_y, int size_z,
                             subGrid **decomp_table_out)
{
    subGrid local_info = {rank, start_x, start_y, size_x, size_y};
    subGrid *all_info = (subGrid *)malloc(size * sizeof(subGrid));
    
    // Define MPI datatype for subGrid
    MPI_Datatype mpi_subGrid_type;
    int block_lengths[5] = {1, 1, 1, 1, 1};
    MPI_Aint displacements[5];
    MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_INT};

    subGrid temp;
    MPI_Aint base_address;
    MPI_Get_address(&temp, &base_address);
    MPI_Get_address(&temp.tile_id, &displacements[0]);
    MPI_Get_address(&temp.start_x, &displacements[1]);
    MPI_Get_address(&temp.start_y, &displacements[2]);
    MPI_Get_address(&temp.size_x, &displacements[3]);
    MPI_Get_address(&temp.size_y, &displacements[4]);

    for (int i = 0; i < 5; i++) {
        displacements[i] -= base_address;
    }

    MPI_Type_create_struct(5, block_lengths, displacements, types, &mpi_subGrid_type);
    MPI_Type_commit(&mpi_subGrid_type);

    // Use custom type in Allgather
    MPI_Allgather(&local_info, 1, mpi_subGrid_type,
                  all_info, 1, mpi_subGrid_type,
                  MPI_COMM_WORLD);

    *decomp_table_out = all_info;

    MPI_Type_free(&mpi_subGrid_type);
}

/* FUNCTION FOR TIMINGS */
void write_timings(int rank, timing_t timing)
{
  double time_IO, time_check, time_bucket, time_comm, time_gridding, time_fft, time_phase, time_total;
  
  MPI_Reduce(&timing.IO, &time_IO, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timing.gridding, &time_gridding, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timing.fft, &time_fft, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timing.phase, &time_phase, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timing.total, &time_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Reduce(&timing.check, &time_check, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timing.bucket, &time_bucket, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&timing.communication, &time_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
  if (rank == 0)
    {
      printf("%40s time: %g sec\n", "I/O (reading)", time_IO);
      printf("%40s time: %g sec\n", "Initialization", time_check);
      printf("%40s time: %g sec\n", "Bucket sort", time_bucket);
      printf("%40s time: %g sec\n", "Communication", time_comm);
      printf("%40s time: %g sec\n", "Gridding", time_gridding);
      printf("%40s time: %g sec\n", "FFT", time_fft);
      printf("%40s time: %g sec\n", "Phase correction", time_phase);
      printf("%40s time: %g sec\n", "Total", time_total);
    }
}



int main(int argc, char **argv)
{
  int rank;
  int size;

  int num_files_to_read = 6;
  
  // Define main filenames
  FILE *pFile;
  MPI_File pFile1;

  // MS paths
  char datapath[900];
  char datapath_multi[NFILES][900];


  /* Create an array of files to be read */
  char filename[1000];
  char **rfiles = (char**)malloc(num_files_to_read*sizeof(char*));
  for (int i=0; i<num_files_to_read; i++)
    rfiles[i]   = (char*)malloc(FILENAMELENGTH*sizeof(char));
      
  rfiles[0] = "ucoord.bin";
  rfiles[1] = "vcoord.bin";
  rfiles[2] = "wcoord.bin";
  rfiles[3] = "weights.bin";
  rfiles[4] = "visibilities_real.bin";
  rfiles[5] = "visibilities_img.bin";
  
  
  char metafile[FILENAMELENGTH] = "meta.txt";

#if defined(WRITE_DATA)
  char gridded_writedata1[FILENAMELENGTH] = "gridded_data_real.bin";
  char gridded_writedata2[FILENAMELENGTH] = "gridded_data_img.bin";
  char fftfile_writedata1[FILENAMELENGTH] = "ffted_data_real.bin";
  char fftfile_writedata2[FILENAMELENGTH] = "ffted_data_img.bin";
#endif

  myull Nmeasures;
  myull Nvis;
  myull Nweights;
  int freq_per_chan;
  int polarisations;
  int Ntimes;
  double dt;
  double thours;
  long baselines;
  double uvmin;
  double uvmax;
  double wmin;
  double wmax;
  double resolution;

  // Mesh related parameters: global size
  int grid_size_x = 4096;
  int grid_size_y = 4096;
  // Split Mesh size (auto-calculated)
  int local_grid_size_x;
  int local_grid_size_y;
  int xaxis;
  int yaxis;

  // Number of planes in the w direction
  int num_w_planes = 8;

  // Size of the convoutional kernel support
  int w_support = 7;

  double dx = 1.0 / (double)grid_size_x;
  double dw = 1.0 / (double)num_w_planes;

  double w_supporth = (double)((w_support - 1) / 2) * dx;

  myull naxis = 2;
  myull naxes[2] = {grid_size_x, grid_size_y};

  int num_threads;

#ifdef USE_MPI
#ifdef _DOPENMP // Use MPI and OpenMP
  int thread_level;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
#else
  MPI_Init(&argc, &argv);
#endif //_OPENMP
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

#ifdef _OPENMP
  num_threads = omp_get_max_threads();
#else
  num_threads = 1;
#endif //_OPENMP

  if (rank == 0)
  {
    printf("\n");
    printf("RRR   III   CCCC  K   K\n");
    printf("R  R   I   C      K  K \n");
    printf("RRR    I   C      KK   \n");
    printf("R  R   I   C      K  K \n");
    printf("R   R III   CCCC  K   K    Library\n");
    printf("\n");
    printf("Radio Imaging Code Kernels Library (v2.0.0)\n");
    printf("\n");

    printf("Running with %d MPI tasks\n", size);
  }
#else
  rank = 0;
  size = 1;

#ifdef _OPENMP
  num_threads = omp_get_max_threads();
#else
  num_threads = 1;
#endif //_OPENMP
#endif // USE_MPI

  int x_start;
  int y_start;

  if (rank == 0)
    printf("Compute Gaussian DD along y (v) axis...\n");
  
  /* COMPUTE GAUSSIAN SLAB DECOMPOSITION ALONG Y AXIS */
  compute_gaussian_1d_decomp(grid_size_x, grid_size_y, size, rank, &x_start, &y_start, &xaxis, &yaxis);

  int x_end = x_start + xaxis;
  int y_end = y_start + yaxis;
  
  /* COLLECT DD AMONG ALL MPI TASKS */
  subGrid *decomp_table = NULL;
  collect_decomposition(rank, size, x_start, y_start, xaxis, yaxis, num_w_planes, &decomp_table);

  if (rank == 0)
    printf("DD computed and collected\n");

  if (rank == 0)
    {
      for (int r = 0; r < size; r++)
	{
	  int sx = decomp_table[r].start_x;
	  int sy = decomp_table[r].start_y;
	  int dx = decomp_table[r].size_x;
	  int dy = decomp_table[r].size_y;
	  int id = decomp_table[r].tile_id;
	  
	 #ifdef DEBUG
	  printf("Rank %d sees: task %d (%d) → start=(%d,%d), end=(%d,%d), size=(%d × %d )\n", rank, r, id,  sx, sy, sx+dx-1, sy+dy -1, dx, dy);
	 #endif
	}
    }

  
  
  int ndatasets = 1;
  //strcpy(datapath_multi[0], "/beegfs/glacopo/IMAGING/ZW2_IFRQ_0444.binMS/");
  //strcpy(datapath_multi[0], "/data/ZW2_IFRQ_0444.binMS/");
  //strcpy(datapath_multi[0], "/home/giovanni/RICK_LIBRARY/RICK/data/newgauss2noconj_t201806301100_SBL180.binMS/");
  strcpy(datapath_multi[0], "/u/glacopo/RICK_PMT/newgauss2noconj_t201806301100_SBL180.binMS/");

  char metaname[1000];
  
  strcpy(datapath, datapath_multi[0]);
  // Read metadata
  strcpy(metaname, datapath);
  strcat(metaname, metafile);
  pFile = fopen(metaname, "r");
  fscanf(pFile, "%llu", &Nmeasures);
  fscanf(pFile, "%llu", &Nvis);
  fscanf(pFile, "%d", &freq_per_chan);
  fscanf(pFile, "%d", &polarisations);
  fscanf(pFile, "%u", &Ntimes);
  fscanf(pFile, "%lf", &dt);
  fscanf(pFile, "%lf", &thours);
  fscanf(pFile, "%ld", &baselines);
  fscanf(pFile, "%lf", &uvmin);
  fscanf(pFile, "%lf", &uvmax);
  fscanf(pFile, "%lf", &wmin);
  fscanf(pFile, "%lf", &wmax);
  fclose(pFile);

  Nvis = Nmeasures * freq_per_chan * polarisations;
  Nweights = Nmeasures * freq_per_chan * polarisations;

  myull nm_pe    = Nmeasures / size;
  myull rem      = Nmeasures % size;
  myull startrow = rank * nm_pe + (rank < rem ? rank : rem);
  nm_pe         = nm_pe + (rank < rem ? 1 : 0);
  
  Nmeasures = nm_pe;
  Nvis = Nmeasures * freq_per_chan * polarisations;
  Nweights = Nmeasures * freq_per_chan * polarisations;

  if (rank == 0)
  {
    printf("N. measurements %llu\n", Nmeasures);
    printf("N. visibilities %llu\n", Nvis);
  }

  double *vvt = (double*)malloc(Nmeasures*sizeof(double));
    
  /* DEFINE THE TIMINGS STRUCTURE */
  timing_t timing;

  if (rank == 0)
    printf("READING DATA WITH MPI-I/O AND DISTRIBUTING WITH POINT-TO-POINT COMMUNICATION\n");

  double total_start = WALLCLOCK_TIME;

  io_read(rank, filename, datapath, pFile1, rfiles[1], (char*)vvt, sizeof(double) * startrow, Nmeasures * sizeof(double));
  
  timing.IO += WALLCLOCK_TIME - total_start;

  /******************************************************************************/
  /******************************************************************************/
  /******************************************************************************/
  /******************************************************************************/
  /******************************************************************************/

  myull *Npts      = (myull*)calloc(size,sizeof(myull)); 
  myull *Npts_recv = (myull*)calloc(size,sizeof(myull));
  double *min      = (double*)calloc(size,sizeof(double));
  double *max      = (double*)calloc(size,sizeof(double));
  
  /* GHOST REGION */
  double epsilon_ghost = w_supporth;
  //double epsilon_ghost = 0.0;
    
  for (int r = 0; r < size; r++)
    {
      /* Check for boundaries in accounting for ghost regions */
      double sy_r = (double)decomp_table[r].start_y / grid_size_y;
      double dy_r = (double)decomp_table[r].size_y / grid_size_y;

      min[r] = (sy_r - epsilon_ghost >= 0. ? sy_r - epsilon_ghost : 0.);
      max[r] = (sy_r + dy_r + epsilon_ghost <= 1. ? sy_r +  dy_r + epsilon_ghost : 1.);

     #ifdef DEBUG
      if (rank == 0)
	printf("Rank 0 sees: Rank %d: %f ---> %f\n", r, min[r], max[r]);
     #endif
    }
  
  double check_start = WALLCLOCK_TIME;

  for (myull i=0; i<Nmeasures; i++)
    for (int r = 0; r < size; r++)
      Npts[r]             += ((vvt[i] >= min[r] && vvt[i] <= max[r]) ? 1 : 0);

  myull offset_bs = 0;
  myull *Noff     = (myull*)malloc(size*sizeof(myull));
  
  for(int r=0; r < size; r++)
    {
      Noff[r] = offset_bs;
      offset_bs += Npts[r];
    }

  myull *bucket_sort = (myull*)malloc(offset_bs*sizeof(myull));
  myull *ncount = (myull*)calloc(size, sizeof(myull));
  for (myull i=0; i<Nmeasures; i++)
    for (int r = 0; r < size; r++)
      {
	int check = (vvt[i] >= min[r] && vvt[i] <= max[r]);
	if (check)
	  {
	    bucket_sort[Noff[r] + ncount[r]] = i;
	    ncount[r]++;
	  }
      }
  
  free(ncount);
  
 #ifdef DEBUG
  printf("Task %d: Noff = %llu\n", rank, offset_bs);
 #endif
  
  exchange_Npts(Npts, Npts_recv, rank, size, MPI_COMM_WORLD);

  /* Initialization time */
  timing.check += WALLCLOCK_TIME - check_start;

  myull total_size = Npts[rank];
  
  for (int i=0; i<size; i++)
    total_size += Npts_recv[i];

 #ifdef DEBUG4
  printf("Rank %d: Npts_total = %llu, Nmeasures = %llu\n",
	 rank, total_size, Nmeasures);
 #endif
  
  /* Allocate arrays in reordered series */
  
  double *vv = (double*)malloc(total_size*sizeof(double));

  /* Perform the communication for vv */
  
  
  double *buffer_coord = (double*)malloc(offset_bs*sizeof(double));

  double start_bucket = WALLCLOCK_TIME;
  
  for (myull kk=0; kk<offset_bs; kk++)
    {
      buffer_coord[kk] = vvt[bucket_sort[kk]];
    }

  timing.bucket += WALLCLOCK_TIME - start_bucket;
  
  free(vvt);

  double start_comm = WALLCLOCK_TIME;
  
  exchange_double(Npts, Npts_recv, buffer_coord, vv, rank, size, MPI_COMM_WORLD);

  timing.communication += WALLCLOCK_TIME - start_comm;
  
  /* Set the buffer coord values to zero */
  memset(buffer_coord, 0.0, offset_bs * sizeof(double));

  /* Read and communicate uu */
  double *uut       = (double*)malloc(Nmeasures*sizeof(double));

  double start_read = WALLCLOCK_TIME;
  
  io_read(rank, filename, datapath, pFile1, rfiles[0], (char*)uut, sizeof(double) * startrow, Nmeasures * sizeof(double));

  timing.IO += WALLCLOCK_TIME - start_read;

  start_bucket = WALLCLOCK_TIME;
  
  for (myull kk=0; kk<offset_bs; kk++)
    {
      buffer_coord[kk] = uut[bucket_sort[kk]];
    }

  timing.bucket += WALLCLOCK_TIME - start_bucket;
  
  free(uut);

  double *uu = (double*)malloc(total_size*sizeof(double));

  start_comm = WALLCLOCK_TIME;
  
  exchange_double(Npts, Npts_recv, buffer_coord, uu, rank, size, MPI_COMM_WORLD);

  timing.communication += WALLCLOCK_TIME - start_comm;

  /* Set the buffer coord values to zero */
  memset(buffer_coord, 0.0, offset_bs * sizeof(double));

  /* Read and communicate ww */
  double *wwt       = (double*)malloc(Nmeasures*sizeof(double)); 

  start_read = WALLCLOCK_TIME;

  io_read(rank, filename, datapath, pFile1, rfiles[2], (char*)wwt, sizeof(double) * startrow, Nmeasures * sizeof(double));

  timing.IO += WALLCLOCK_TIME - start_read;

  start_bucket = WALLCLOCK_TIME;

  for (myull kk=0; kk<offset_bs; kk++)
    {
      buffer_coord[kk] = wwt[bucket_sort[kk]];
    }

  timing.bucket += WALLCLOCK_TIME - start_bucket;
  
  free(wwt);

  double *ww = (double*)malloc(total_size*sizeof(double));

  start_comm = WALLCLOCK_TIME;
  
  exchange_double(Npts, Npts_recv, buffer_coord, ww, rank, size, MPI_COMM_WORLD);

  timing.communication += WALLCLOCK_TIME - start_comm;
  
  free(buffer_coord);


  /* Read and communicate weights */
  float  *weightst  = (float*)malloc(Nweights*sizeof(float));

  start_read = WALLCLOCK_TIME;

  io_read(rank, filename, datapath, pFile1, rfiles[3], (char*)weightst, sizeof(float) * startrow * freq_per_chan * polarisations, Nweights * sizeof(float));

  timing.IO += WALLCLOCK_TIME - start_read;

  /* Define new arrays for Npts and Npts_recv */
  myull *vis_Npts      = (myull*)malloc(size*sizeof(myull));
  myull *vis_Npts_recv = (myull*)malloc(size*sizeof(myull));

  for (int r=0; r<size; r++)
    {
      vis_Npts[r]      = Npts[r] * freq_per_chan * polarisations;
      vis_Npts_recv[r] = Npts_recv[r] * freq_per_chan * polarisations;
    }
  
  /* Allocate a new buffer for weights */
  float *buffer_weights = (float*)malloc(offset_bs*freq_per_chan*polarisations*sizeof(float));

  start_bucket = WALLCLOCK_TIME;
  
  for (myull kk=0; kk<offset_bs; kk++)
    for (myull ww=0; ww<(freq_per_chan * polarisations); ww++)
      buffer_weights[kk * freq_per_chan * polarisations + ww] = weightst[bucket_sort[kk] * freq_per_chan * polarisations + ww];
    

  timing.bucket += WALLCLOCK_TIME - start_bucket;
  
  free(weightst);
  
  float *weights = (float*)malloc(total_size * freq_per_chan * polarisations * sizeof(float));

  start_comm = WALLCLOCK_TIME;
  
  exchange_float(vis_Npts, vis_Npts_recv, buffer_weights, weights, rank, size, MPI_COMM_WORLD);

  timing.communication += WALLCLOCK_TIME - start_comm;
  
  free(buffer_weights);

  
  /* Read and communicate visibilities */
  float  *vis_realt = (float*)malloc(Nvis*sizeof(float));
    
  start_read = WALLCLOCK_TIME;

  io_read(rank, filename, datapath, pFile1, rfiles[4], (char*)vis_realt, sizeof(float) * startrow * freq_per_chan * polarisations, Nvis * sizeof(float));

  timing.IO += WALLCLOCK_TIME - start_read;

  /* Allocate a new buffer for visibility data */
  float *buffer_vis = (float*)malloc(offset_bs*freq_per_chan*polarisations*sizeof(float));

  start_bucket = WALLCLOCK_TIME;
  
  for (myull kk=0; kk<offset_bs; kk++)
    for (myull ww=0; ww<(freq_per_chan * polarisations); ww++)
      buffer_vis[kk * freq_per_chan * polarisations + ww] = vis_realt[bucket_sort[kk] * freq_per_chan * polarisations + ww];


  timing.bucket += WALLCLOCK_TIME - start_bucket;
  
  free(vis_realt);
    
  float *vis_real = (float*)malloc(total_size * freq_per_chan * polarisations * sizeof(float));

  start_comm = WALLCLOCK_TIME;
  
  exchange_float(vis_Npts, vis_Npts_recv, buffer_vis, vis_real, rank, size, MPI_COMM_WORLD);

  timing.communication += WALLCLOCK_TIME - start_comm;
  
  /* Set the buffer coord values to zero */
  memset(buffer_vis, 0.0, Nvis * sizeof(float));

  float  *vis_imgt  = (float*)malloc(Nvis*sizeof(float));

  start_read = WALLCLOCK_TIME;

  io_read(rank, filename, datapath, pFile1, rfiles[5], (char*)vis_imgt, sizeof(float) * startrow * freq_per_chan * polarisations, Nvis * sizeof(float));

  timing.IO += WALLCLOCK_TIME - start_read;

  start_bucket = WALLCLOCK_TIME;
  
  for (myull kk=0; kk<offset_bs; kk++)
    for (myull ww=0; ww<(freq_per_chan * polarisations); ww++)
      buffer_vis[kk * freq_per_chan * polarisations + ww] = vis_imgt[bucket_sort[kk] * freq_per_chan * polarisations + ww];
    
  timing.bucket += WALLCLOCK_TIME - start_bucket;
  
  free(vis_imgt);

  float *vis_img = (float*)malloc(total_size * freq_per_chan * polarisations * sizeof(float));

  start_comm = WALLCLOCK_TIME;
  
  exchange_float(vis_Npts, vis_Npts_recv, buffer_vis, vis_img, rank, size, MPI_COMM_WORLD);

  timing.communication += WALLCLOCK_TIME - start_comm;
  
  free(buffer_vis);
  
  free(vis_Npts_recv);
  free(vis_Npts);
  free(bucket_sort);
  free(Npts_recv);
  free(Npts);
  free(min);
  free(max);
  
  myull size_of_grid = 2 * num_w_planes * xaxis * yaxis;
  //printf("Task %d, my size = %llu\n", rank, size_of_grid);
  
  
  double *grid;
  grid = (double *)calloc(size_of_grid, sizeof(double));
  //double *gridss;
  //gridss = (double *)calloc(size_of_grid, sizeof(double));

  double *image_real = (double *)calloc(xaxis * yaxis, sizeof(double));
  double *image_imag = (double *)calloc(xaxis * yaxis, sizeof(double));

  double gridding_start = WALLCLOCK_TIME;
  
  gridding(
      rank,
      size,
      total_size,
      uu,
      vv,
      ww,
      grid,
      MPI_COMM_WORLD,
      num_threads,
      grid_size_x,
      grid_size_y,
      y_start,
      yaxis,
      w_support,
      num_w_planes,
      polarisations,
      freq_per_chan,
#if defined(WRITE_DATA)
      gridded_writedata1,
      gridded_writedata2,
#endif
      vis_real,
      vis_img,
      weights,
      uvmin,
      uvmax,
      total_size);

  timing.gridding += WALLCLOCK_TIME - gridding_start;

  free(ww);
  free(vv);
  free(uu);


  double fft_start = WALLCLOCK_TIME;
  
  fftw_data(
      grid_size_x,
      grid_size_y,
      xaxis,
      y_start,
      yaxis,
      num_w_planes,
      num_threads,
      MPI_COMM_WORLD,
      size,
      rank,
#if defined(WRITE_DATA)
      fftfile_writedata1,
      fftfile_writedata2,
#endif
      grid);
  
  timing.fft += WALLCLOCK_TIME - fft_start;

  double phase_start = WALLCLOCK_TIME;
  
  phase_correction(
      grid,
      image_real,
      image_imag,
      num_w_planes,
      grid_size_x,
      grid_size_y,
      xaxis,
      yaxis,
      y_start,
      wmin,
      wmax,
      uvmin,
      uvmax,
      num_threads,
      size,
      rank,
      MPI_COMM_WORLD);

  timing.phase += WALLCLOCK_TIME - phase_start;

  free(image_imag);
  free(image_real);
  free(grid);
  
  if (rank == 0)
    printf("End of main\n");

  timing.total = WALLCLOCK_TIME - total_start;

  write_timings(rank, timing);
  
#ifdef USE_MPI
  MPI_Finalize();
#endif
}
