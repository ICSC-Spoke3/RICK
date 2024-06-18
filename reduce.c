#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif

#include <stdio.h>
#include <math.h>
#include "allvars.h"
#include "proto.h"
//#include "mypapi.h"


#if defined(DEBUG)
double check_host_value   = 0;
double check_global_value = 0;
#endif

struct { double rtime, ttotal, treduce, tspin, tspin_in, tmovmemory, tsum;} timing_red = {0};
struct { double tmpi, tmpi_reduce, tmpi_reduce_wait, tmpi_setup;} timing_redmpi = {0};


int_t summations = 0;
int_t memmoves   = 0;


int shmem_reduce_ring  ( int, int, int_t, map_t *, double * restrict, blocks_t *);

int reduce_ring (int target_rank)
{
  /* -------------------------------------------------
   *
   *  USE THE SHARED MEMORY WINDOWS TO REDUCE DATA 
   * ------------------------------------------------- */
  /*
  numa_init( rank, size, &MYMPI_COMM_WORLD, &Me );
  numa_allocate_shared_windows( &Me, size_of_grid*sizeof(double), sizeof(double));

  
  memset( (char*)Me.win.ptr, 0, size_of_grid*sizeof(double)*1.1);

  if( Me.Rank[myHOST] == 0 )
    {
      for( int tt = 1; tt < Me.Ntasks[myHOST]; tt++ )
        memset( (char*)Me.swins[tt].ptr, 0, size_of_grid*sizeof(double)*1.1);
    }

  numa_expose(&Me, verbose_level);
  
  MPI_Barrier(MPI_COMM_WORLD);
  
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
  int_t size_b  = size_of_grid / blocks.Nblocks;
  int_t rem   = size_of_grid % blocks.Nblocks;

  blocks.Bsize[0]  = size_b + (rem > 0);
  blocks.Bstart[0] = 0;
  for(int b = 1; b < blocks.Nblocks; b++ ) {
    blocks.Bstart[b] = blocks.Bstart[b-1]+blocks.Bsize[b-1];
    blocks.Bsize[b] = size_b + (b < rem); }
  */
  
  timing_red.rtime  = CPU_TIME_rt;
  timing_red.ttotal = CPU_TIME_pr;

 #pragma omp parallel num_threads(2)
  {

   #ifdef _OPENMP
    int thid         = omp_get_thread_num();
   #else
    int thid         = 0;
   #endif
    
    int Ntasks_local = Me.Ntasks[Me.SHMEMl];
    
    if( thid == 1 )
      {		    
	// check that the data in Me.win
	// can be overwritten by new data
	// -> this condition is true when
	// win_ctrl has the value "DATA_FREE"
		    		    
	if( Ntasks_local > 1 )
	  {
	    int value = target_rank * Ntasks_local;			
	    for ( int jj = 0; jj < Me.Ntasks[Me.SHMEMl]; jj++ )
	      *((int*)Me.win_ctrl.ptr+CTRL_BLOCKS+jj) = value;

	    atomic_store((_Atomic int*)Me.win_ctrl.ptr+CTRL_FINAL_CONTRIB, 0);
	    //atomic_thread_fence(memory_order_release);
	    atomic_store((_Atomic int*)Me.win_ctrl.ptr+CTRL_SHMEM_STATUS, value);

	    
	    //CPU_TIME_STAMP( Me.Rank[myHOST], "A0");
	    // calls the reduce
	    double start = CPU_TIME_tr;			
	    int ret = shmem_reduce_ring( target_rank, target_rank, size_of_grid, &Me, (double*)Me.win.ptr, &blocks );	
	    timing_red.treduce += CPU_TIME_tr - start;

	    reduce_shmem_time = timing_red.treduce;
	    if( ret != 0 )
	      {
		printf("Task %d : shared-memory reduce for sector %d has returned "
		       "an error code %d : better stop here\n",
		       rank, target_rank, ret );
		free( blocks.Bsize );
		free( blocks.Bstart );
		numa_shutdown(rank, 0, &MYMPI_COMM_WORLD, &Me);
		MPI_Finalize();
	      }
			
	  }
	else
	  {
	    ACQUIRE_CTRL((_Atomic int*)Me.win_ctrl.ptr+CTRL_FINAL_STATUS, FINAL_FREE, timing_red.tspin, != );
	    // mimic the production of new data
	    //memcpy(Me.win.ptr, gridss, sizeof(gridss));

	    atomic_store(((_Atomic int*)Me.win_ctrl.ptr+CTRL_FINAL_CONTRIB), Ntasks_local);
	  }

	int Im_target                   = (rank == target_rank);
	int Im_NOT_target_but_Im_master = (Me.Nhosts>1) &&
	  (Me.Ranks_to_host[target_rank]!=Me.myhost) && (Me.Rank[myHOST]==0);
		    
	if( Im_target || Im_NOT_target_but_Im_master )
	  {			
	    ACQUIRE_CTRL((_Atomic int*)Me.win_ctrl.ptr+CTRL_FINAL_CONTRIB, Ntasks_local, timing_red.tspin, !=);
	    atomic_store(((_Atomic int*)Me.win_ctrl.ptr+CTRL_FINAL_STATUS), target_rank);
	  }

	atomic_fetch_add( (_Atomic int*)win_ctrl_hostmaster_ptr+CTRL_BARRIER_END, (int)1 );
	switch( Me.Rank[Me.SHMEMl] ) {
	case 0: { ACQUIRE_CTRL((_Atomic int*)win_ctrl_hostmaster_ptr+CTRL_BARRIER_END, Ntasks_local, timing_red.tspin, !=);
	    atomic_store( (_Atomic int*)win_ctrl_hostmaster_ptr+CTRL_BARRIER_END, (int)0 ); } break;
	default : ACQUIRE_CTRL((_Atomic int*)win_ctrl_hostmaster_ptr+CTRL_BARRIER_END, 0, timing_red.tspin, !=); break;
	}

      }

    else 
      {
		    
	/* 
	 *
	 *  REDUCE AMONG HOSTS
	 */

	if ( (Me.Nhosts > 1) && (Me.Rank[myHOST] == 0) )
	  {			
	    double start = CPU_TIME_tr;
			
	    int target_task       = Me.Ranks_to_host[target_rank];
	    int Im_hosting_target = Me.Ranks_to_host[target_rank] == Me.myhost;
	    int target            = 0;
			
	    if( Im_hosting_target )
	      while( (target < Me.Ntasks[Me.SHMEMl]) &&
		     (Me.Ranks_to_myhost[target] != target_rank) )
		target++;

			
	    _Atomic int *ctrl_ptr = (_Atomic int*)Me.scwins[target].ptr+CTRL_FINAL_STATUS;
					    
	    double *send_buffer = ( Im_hosting_target ? MPI_IN_PLACE : (double*)Me.win.ptr+size_of_grid );
			
	    double *recv_buffer = ( Im_hosting_target ? (double*)Me.sfwins[target].ptr : NULL );

                       
	    timing_redmpi.tmpi_setup += CPU_TIME_tr - start;

	    double tstart = CPU_TIME_tr;
			
	    ACQUIRE_CTRL( ctrl_ptr, target_rank, timing_red.tspin, !=);
			
	    timing_redmpi.tmpi_reduce_wait += CPU_TIME_tr - tstart;

	    tstart = CPU_TIME_tr;
	    MPI_Ireduce(send_buffer, recv_buffer, size_of_grid, MPI_DOUBLE, MPI_SUM, target_task, COMM[HOSTS], &requests[target_rank]);			
	    timing_redmpi.tmpi_reduce += CPU_TIME_tr - tstart;
	    	    
	    MPI_Wait( &requests[target_rank], MPI_STATUS_IGNORE );
	    atomic_store(ctrl_ptr, FINAL_FREE);

	    //printf("Im after MPI_Ireduce and my global rank %d and local rank %d\n", rank, Me.Rank[HOSTS]);	
	    timing_redmpi.tmpi += CPU_TIME_tr - start;
	    reduce_mpi_time = timing_redmpi.tmpi;
	  }

	atomic_thread_fence(memory_order_release);
		    
      } // closes thread 0
		

  }
  timing_red.rtime  = CPU_TIME_rt - timing_red.rtime;
  timing_red.ttotal = CPU_TIME_pr - timing_red.ttotal;
	  


  return 0;
}

int shmem_reduce_ring( int sector, int target_rank, int_t size_of_grid, map_t *Me, double * restrict data, blocks_t *blocks )
 {
   int local_rank            = Me->Rank[Me->SHMEMl];
   int target_rank_on_myhost = 0;
   int Im_hosting_target     = 0;
   
   if( Me->Ranks_to_host[ target_rank ] == Me->myhost )
     // exchange rank 0 with target rank
     // in this way the following log2 alogorithm,
     // which reduces to rank 0, will work for
     // every target rank
     {

       Im_hosting_target = 1;
       target_rank_on_myhost = 0;
       while( (target_rank_on_myhost < Me->Ntasks[Me->SHMEMl]) &&
	      (Me->Ranks_to_myhost[target_rank_on_myhost] != target_rank) )
	 target_rank_on_myhost++;

       if( target_rank_on_myhost == Me->Ntasks[Me->SHMEMl] )
	 return -1;
     }

   // Here we start the reduction
   //

   dprintf(1, 0, 0, "@ SEC %d t %d (%d), %d\n",
	   sector, local_rank, rank, *(int*)Me->win_ctrl.ptr);

   // main reduction loop
   //
   int SHMEMl  = Me->SHMEMl;
   int Nt      = Me->Ntasks[SHMEMl];
   int end     = Me->Ntasks[SHMEMl]-1;
   int target  = (Nt+(local_rank-1)) % Nt;
   int myblock = local_rank;
   int ctrl    = sector*Nt;

   //CPU_TIME_STAMP( local_rank, "R0");
   ACQUIRE_CTRL( ((_Atomic int*)Me->scwins[target].ptr)+CTRL_SHMEM_STATUS, ctrl, timing_red.tspin_in, != );        // is my target ready?
   
   for(int t = 0; t < end; t++)
     {
 	                                                                      // prepare pointers for the summation loop
       int_t  dsize = blocks->Bsize[myblock];
       double * restrict my_source = (double*)Me->swins[target].ptr + blocks->Bstart[myblock];
       double * restrict my_target = data + blocks->Bstart[myblock];
       my_source = __builtin_assume_aligned( my_source, 8);
       my_target = __builtin_assume_aligned( my_target, 8);

       dprintf(1, 0, 0, "+ SEC %d host %d l %d t %d <-> %d block %d from %llu to %llu\n",
	       sector, Me->myhost, t, local_rank, target, myblock, 
	       blocks->Bstart[myblock], blocks->Bstart[myblock]+dsize );

       
	                                                                      // check whether the data of the source rank
	                                                                      // are ready to be used (control tag must have
	                                                                      // the value of the current sector )
       //CPU_TIME_STAMP( local_rank, "R1");
       ACQUIRE_CTRL( ((_Atomic int*)Me->scwins[target].ptr)+CTRL_BLOCKS+myblock, ctrl, timing_red.tspin_in, !=);        // is myblock@Me ready?
       //CPU_TIME_STAMP( local_rank, "R2");

	                                                                      // performs the summation loop
	                                                                      //
      #if defined(USE_PAPI)
       if( sector == 0 ) {
	 PAPI_START_CNTR;
	 summations += dsize; }
      #else
       summations += dsize;
      #endif

       double  tstart = CPU_TIME_tr;
       double *my_end = my_source+dsize;
       switch( dsize < BUNCH_FOR_VECTORIZATION )
	 {
	 case 0: {
	   int      dsize_4  = (dsize/4)*4;
	   double * my_end_4 = my_source+dsize_4;
	   for( ; my_source < my_end_4; my_source+=4, my_target+=4 )
	     {
	       __builtin_prefetch( my_target+8, 0, 1);
	       __builtin_prefetch( my_source+8, 0, 1);
	       *my_target += *my_source;
	       *(my_target+1) += *(my_source+1);
	       *(my_target+2) += *(my_source+2);
	       *(my_target+3) += *(my_source+3);
	     } }
	 case 1: { for( ; my_source < my_end; my_source++, my_target++)
	       *my_target += *my_source; } break;
	 }
       
       timing_red.tsum += CPU_TIME_tr - tstart;
      #if defined(USE_PAPI)
       if( sector == 0 )
	 PAPI_STOP_CNTR;
      #endif

       
       ctrl++;
       atomic_store( ((_Atomic int*)Me->win_ctrl.ptr+CTRL_BLOCKS+myblock), ctrl );
       //CPU_TIME_STAMP( local_rank, "R3");
       dprintf(1, 0, 0, "- SEC %d host %d l %d t %d ... writing tag %d on block %d = %d\n",
	       sector, Me->myhost, t, local_rank, ctrl, myblock, 
	       *((int*)Me->win_ctrl.ptr+CTRL_BLOCKS+myblock) );
       
       myblock = (Nt+(myblock-1)) % Nt;
       atomic_thread_fence(memory_order_release);
     }

   myblock = (myblock+1)%Nt;
   int_t offset = blocks->Bstart[myblock];
   int_t dsize  = blocks->Bsize[myblock];

   dprintf(1,0,0, "c SEC %d host %d t %d (%d) ==> t %d, block %d %llu from %llu\n",
	   sector, Me->myhost, local_rank, rank, target_rank_on_myhost, myblock, dsize, offset );

   double tstart2 = CPU_TIME_tr;
   double * restrict my_source = data+offset;
   double *          my_end    = my_source+dsize;
   double * restrict my_final;

   switch( Im_hosting_target ) {
   case 0: my_final = (double*)Me->swins[0].ptr+size_of_grid+offset; break;
   case 1: my_final = (double*)Me->sfwins[target_rank_on_myhost].ptr+offset; }

   my_source = __builtin_assume_aligned( my_source, 8);
   my_final  = __builtin_assume_aligned( my_final, 8);

   atomic_thread_fence(memory_order_acquire);
   ACQUIRE_CTRL((_Atomic int*)Me->scwins[target_rank_on_myhost].ptr+CTRL_FINAL_STATUS, FINAL_FREE, timing_red.tspin_in, != );

   switch( dsize < BUNCH_FOR_VECTORIZATION ) {
   case 0: { double *end_4 = my_source + (dsize/4)*4;
       for( ; my_source < end_4; my_source+=4, my_final+=4) {
	 *my_final = *my_source; *(my_final+1) = *(my_source+1);
	 *(my_final+2) = *(my_source+2); *(my_final+3) = *(my_source+3); } }
   case 1: { for ( ; my_source < my_end; my_source++, my_final++ ) *my_final = *my_source; } break;
   }
   
   atomic_fetch_add((_Atomic int*)Me->scwins[target_rank_on_myhost].ptr+CTRL_FINAL_CONTRIB, (int)1);   
   timing_red.tmovmemory += CPU_TIME_tr - tstart2;

   memmoves += dsize;
   
   //atomic_thread_fence(memory_order_release);
   return 0;
 }
