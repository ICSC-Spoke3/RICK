#include "allvars_rccl.hip.hpp"
#include "proto.h"
#include <hip/hip_runtime.h>
#include <rccl.h>

/* 
 * Implements the gridding of data via GPU
 * by using NCCL library
 *
 */


#if defined( RCCL_REDUCE )

/*
#define NCCLCHECK(cmd) do {                         
ncclResult_t r = cmd;                             
if (r!= ncclSuccess) {                            
  printf("Failed, NCCL error %s:%d '%s'\n",       
	 __FILE__,__LINE__,ncclGetErrorString(r));   
  exit(EXIT_FAILURE);                             
 }                                                 
} while(0)
*/  


static uint64_t getHostHash(const char* string) {
  // Based on DJB2a, result = result * 33 ^ char                                                                                                 
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }  
}




void gridding_data(){

  double shift = (double)(dx*yaxis);

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
 
  
    //Initialize nccl

  //double *gridss_gpu, *grid_gpu;
  int local_rank = 0;

  uint64_t hostHashs[size];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[rank] = getHostHash(hostname);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
  for (int p=0; p<size; p++) {
     if (p == rank) break;
     if (hostHashs[p] == hostHashs[rank]) local_rank++;
  }
  
  ncclUniqueId id;
  ncclComm_t comm;
  hipError_t nnn;
  hipStream_t stream_reduce, stream_stacking;

  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  int h = hipSetDevice(local_rank);

  int n = hipMalloc(&grid_gpu, 2*param.num_w_planes*xaxis*yaxis * sizeof(double));
  n = hipMalloc(&gridss_gpu, 2*param.num_w_planes*xaxis*yaxis * sizeof(double));
  n = hipStreamCreate(&stream_reduce);
  n = hipStreamCreate(&stream_stacking);
  
  ncclCommInitRank(&comm, size, id, rank);

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
      for(int iphi = histo_send[isector]-1; iphi>=0; iphi--)
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

      timing_wt.compose += CPU_TIME_wt - start;

      //printf("UU, VV, min, max = %f %f %f %f\n", uumin, uumax, vvmin, vvmax);
      
      // Make convolution on the grid

     #ifdef VERBOSE
      printf("Processing sector %ld\n",isector);
     #endif
          
      start = CPU_TIME_wt;
	    
     //We have to call different GPUs per MPI task!!! [GL]
     #ifdef HIPCC
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
	     gridss_gpu,
	     param.num_threads,
	     rank,
	     stream_stacking);
      #else
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
	     gridss,
	     param.num_threads,
	     rank);
      #endif
      //Allocate memory on devices non-blocking for the host                                                                                   
      ///////////////////////////////////////////////////////


      timing_wt.kernel += CPU_TIME_wt - start;
      
     #ifdef VERBOSE
      printf("Processed sector %ld\n",isector);
     #endif
      

      if( size > 1 )
	{
     
	  // Write grid in the corresponding remote slab
     
	  // int target_rank = (int)isector;    it implied that size >= nsectors
	  int target_rank = (int)(isector % size);

	  start = CPU_TIME_wt;

	  ncclReduce(gridss_gpu, grid_gpu, size_of_grid, ncclDouble, ncclSum, target_rank, comm, stream_reduce);
	  n = hipStreamSynchronize(stream_reduce);
      
	  timing_wt.reduce += CPU_TIME_wt - start;

	  // Go to next sector
	  nnn = hipMemset( gridss_gpu, 0.0, 2*param.num_w_planes*xaxis*yaxis * sizeof(double) );
	  if (nnn != hipSuccess) {printf("!!! w-stacking.cu hipMemset ERROR %d !!!\n", nnn);}
	}

      free(memory);
    }


  //Copy data back from device to host (to be deleted in next steps)
  
   n = hipMemcpyAsync(grid, grid_gpu, 2*param.num_w_planes*xaxis*yaxis*sizeof(double), hipMemcpyDeviceToHost, stream_reduce);
  
  MPI_Barrier(MPI_COMM_WORLD);

  n = hipStreamSynchronize(stream_reduce);
  n = hipFree(grid_gpu);
  n = hipFree(gridss_gpu);
  n = hipStreamDestroy(stream_reduce);
  n = hipStreamDestroy(stream_stacking);
  
  ncclCommDestroy(comm);

  return;
  
}

#endif
