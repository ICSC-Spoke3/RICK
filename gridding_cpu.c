#include "allvars.h"
#include "proto.h"


/* 
 * Implements the gridding of data through CPU,
 * with MPI + OpenMP
 *
 */

#if !defined( NCCL_REDUCE ) && !defined( RCCL_REDUCE )

int reduce_ring (int);

//   .....................................................................
//
void gridding_data()
//
// actually performs the gridding of the data
//
  
{

  double shift = (double)(dx*yaxis);
    

  if( (size > 1) && (param.reduce_method == REDUCE_RING) )
    {
      memset( (char*)Me.win.ptr, 0, size_of_grid*sizeof(double)*1.1);                                                                               
      gridss = (double*)Me.win.ptr; //gridss must point to the right location [GL]
  
      memset( (char*)Me.fwin.ptr, 0, size_of_grid*sizeof(double)*1.1); //allocate the memory for the results [GL]
  
      if( Me.Rank[myHOST] == 0 )
	{
	  for( int tt = 1; tt < Me.Ntasks[myHOST]; tt++ )
	    memset( (char*)Me.swins[tt].ptr, 0, size_of_grid*sizeof(double)*1.1);
	}


      MPI_Barrier(MYMPI_COMM_WORLD);
      if( Me.Rank[HOSTS] >= 0 )
	requests = (MPI_Request *)calloc( Me.Ntasks[WORLD], sizeof(MPI_Request) );

      if( Me.Rank[myHOST] == 0 ) {
	*((int*)win_ctrl_hostmaster_ptr+CTRL_BARRIER_END) = 0;
	*((int*)win_ctrl_hostmaster_ptr+CTRL_BARRIER_START) = 0;
      }

      *((int*)Me.win_ctrl.ptr + CTRL_FINAL_STATUS) = FINAL_FREE;
      *((int*)Me.win_ctrl.ptr + CTRL_FINAL_CONTRIB) = 0;
      *((int*)Me.win_ctrl.ptr + CTRL_SHMEM_STATUS) = -1;
      MPI_Barrier(*(Me.COMM[myHOST]));
      
      blocks.Nblocks = Me.Ntasks[myHOST];
      blocks.Bstart  = (int_t*)calloc( blocks.Nblocks, sizeof(int_t));
      blocks.Bsize   = (int_t*)calloc( blocks.Nblocks, sizeof(int_t));
      int_t size_b   = size_of_grid / blocks.Nblocks;
      int_t rem      = size_of_grid % blocks.Nblocks;
      
      blocks.Bsize[0]  = size_b + (rem > 0);
      blocks.Bstart[0] = 0;
      for(int b = 1; b < blocks.Nblocks; b++ ) {
	blocks.Bstart[b] = blocks.Bstart[b-1]+blocks.Bsize[b-1];
	blocks.Bsize[b] = size_b + (b < rem);
      }
      
    }   // closes reduce_method == REDUCE_RING

  
  timing_wt.kernel     = 0.0;
  timing_wt.reduce     = 0.0;
  timing_wt.reduce_mpi = 0.0;
  timing_wt.reduce_sh  = 0.0;
  timing_wt.compose    = 0.0;

  // calculate the resolution in radians
  resolution = 1.0/MAX(fabs(metaData.uvmin),fabs(metaData.uvmax));
    
  // calculate the resolution in arcsec 
  double resolution_asec = (3600.0*180.0)/MAX(fabs(metaData.uvmin),fabs(metaData.uvmax))/PI;
  if ( rank == 0 )
    printf("RESOLUTION = %f rad, %f arcsec\n", resolution, resolution_asec);

  // find the largest value in histo_send[]
  //
  
  for (myuint isector = 0; isector < nsectors; isector++)
    {

      double start = CPU_TIME_wt;

      myuint Nsec            = histo_send[isector];
      myuint Nweightss       = Nsec*metaData.polarisations;
      myull Nvissec         = Nweightss*metaData.freq_per_chan;
      double_t *memory     = (double*) malloc ( (Nsec*3)*sizeof(double_t) +
						(Nvissec*2+Nweightss)*sizeof(float_t) );

      if ( memory == NULL )
	shutdown_wstacking(NOT_ENOUGH_MEM_STACKING, "Not enough memory for stacking", __FILE__, __LINE__);
  
      double_t *uus        = (double_t*) memory;
      double_t *vvs        = (double_t*) uus+Nsec;
      double_t *wws        = (double_t*) vvs+Nsec;
      float_t  *weightss   = (float_t*)((double_t*)wws+Nsec);
      float_t  *visreals   = (float_t*)weightss + Nweightss;
      float_t  *visimgs    = (float_t*)visreals + Nvissec;
  
      
      
      // select data for this sector
      myuint icount = 0;
      myuint ip = 0;
      myuint inu = 0;

      #warning "this loop should be threaded"
      #warning "the counter of this loop should not be int"
      for( int iphi = histo_send[isector]-1; iphi >=0 ; iphi--)
        {
	  
	  myuint ilocal = sectorarray[isector][iphi];

	  uus[icount] = data.uu[ilocal];
	  vvs[icount] = data.vv[ilocal]-isector*shift;
	  wws[icount] = data.ww[ilocal];
	  for (myuint ipol=0; ipol<metaData.polarisations; ipol++)
	    {
	      weightss[ip] = data.weights[ilocal*metaData.polarisations+ipol];
	      ip++;
	    }
	  for (myuint ifreq=0; ifreq<metaData.polarisations*metaData.freq_per_chan; ifreq++)
	    {
	      visreals[inu] = data.visreal[ilocal*metaData.polarisations*metaData.freq_per_chan+ifreq];
	      visimgs[inu]  = data.visimg[ilocal*metaData.polarisations*metaData.freq_per_chan+ifreq];
	      inu++;
	    }
	  icount++;
	}
      
      double uumin = 1e20;
      double vvmin = 1e20;
      double uumax = -1e20;
      double vvmax = -1e20;

     #pragma omp parallel reduction( min: uumin, vvmin) reduction( max: uumax, vvmax) num_threads(param.num_threads)
      {
	double my_uumin = 1e20;
	double my_vvmin = 1e20;
	double my_uumax = -1e20;
	double my_vvmax = -1e20;

       #pragma omp for 
	for (myuint ipart=0; ipart<Nsec; ipart++)
	  {
	    my_uumin = MIN(my_uumin, uus[ipart]);
	    my_uumax = MAX(my_uumax, uus[ipart]);
	    my_vvmin = MIN(my_vvmin, vvs[ipart]);
	    my_vvmax = MAX(my_vvmax, vvs[ipart]);	    
	  }

	uumin = MIN( uumin, my_uumin );
	uumax = MAX( uumax, my_uumax );
	vvmin = MIN( vvmin, my_vvmin );
	vvmax = MAX( vvmax, my_vvmax );
      }

      //printf("UU, VV, min, max = %f %f %f %f\n", uumin, uumax, vvmin, vvmax);
      

      timing_wt.compose += CPU_TIME_wt - start;
      
      // Make convolution on the grid

     #ifdef VERBOSE
      printf("Processing sector %ld\n",isector);
     #endif

      double *stacking_target_array;
      if ( size > 1 )
	stacking_target_array = gridss;
      else
	stacking_target_array = grid;

      start = CPU_TIME_wt;
	    
     //We have to call different GPUs per MPI task!!! [GL]
      wstack(param.num_w_planes,
	     Nsec,
	     metaData.freq_per_chan,
	     metaData.polarisations,
	     uus,
	     vvs,
	     wws,
	     visreals,
	     visimgs,
	     weightss,
	     dx,
	     dw,
	     param.w_support,
	     xaxis,
	     yaxis,
	     stacking_target_array,
	     param.num_threads,
	     rank);


      timing_wt.kernel += CPU_TIME_wt - start;
      
     #ifdef VERBOSE
      printf("Processed sector %ld\n",isector);
     #endif

      if( size > 1 )
	{
	  // Write grid in the corresponding remote slab
	  
	  int target_rank = (int)(isector % size);

	  start = CPU_TIME_wt;


	  //Force to use MPI_Reduce when -fopenmp is not active
	 #ifdef _OPENMP
	  if( param.reduce_method == REDUCE_MPI )
	   
	    MPI_Reduce(gridss, grid, size_of_grid, MPI_DOUBLE, MPI_SUM, target_rank, MYMPI_COMM_WORLD);
	  
	  else if ( param.reduce_method == REDUCE_RING )
	    {
	      
	      int ret = reduce_ring(target_rank);
	      //grid    = (double*)Me.fwin.ptr; //Let grid point to the right memory location [GL]
	      
	      if ( ret != 0 )
		{
		  char message[1000];
		  sprintf( message, "Some problem occurred in the ring reduce "
			   "while processing sector %d", isector);
		  free( memory );
		  shutdown_wstacking( ERR_REDUCE, message, __FILE__, __LINE__);
		}
	      
	    }
	 #else
	  MPI_Reduce(gridss, grid, size_of_grid, MPI_DOUBLE, MPI_SUM, target_rank, MYMPI_COMM_WORLD);
	 #endif
	  
	  timing_wt.reduce += CPU_TIME_wt - start;

	  // Go to next sector
	  memset ( gridss, 0, 2*param.num_w_planes*xaxis*yaxis * sizeof(double) );	  
	}	

      free(memory);
    }

  if ( size > 1 )
    {
      double start = CPU_TIME_wt;
      if ( param.reduce_method == REDUCE_RING)
	if( (Me.Rank[HOSTS] >= 0) && (Me.Nhosts > 1 )) {
	  MPI_Waitall( Me.Ntasks[WORLD], requests, MPI_STATUSES_IGNORE);
	 free(requests);}
      
      timing_wt.reduce += CPU_TIME_wt - start;
      MPI_Barrier(MYMPI_COMM_WORLD);
    }

  return;

}


#endif    // closes initial if defined(NCCL_REDUCE)
