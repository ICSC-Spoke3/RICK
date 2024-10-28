
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <stdatomic.h>
/* #include <omp.h>  to be included after checking the MPI version works */

#define PI 3.14159265359

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#define NOT_ENOUGH_MEM_STACKING   3

void initialize_array(
    int nsectors,
    int nmeasures,
    double w_supporth,
    double* vv,
    int yaxis,
    double dx,
    int **histo_send,
    int ***sectorarray
)
{
  printf("Beginning of _initialize_array_ function\n");
  *histo_send = calloc(nsectors+1, sizeof(int));

  printf("w_supporth : %f \n", w_supporth);
  printf("nsectors : %d\n", nsectors);
  printf("yaxis : %d\n", yaxis);
  printf("nmeasures : %d\n", nmeasures);
  printf("vv[5] : %f\n", vv[5]);

  for (int iphi = 0; iphi < nmeasures; iphi++)
    {
      double vvh = vv[iphi];
      //printf("vvh = %f\n", vvh);              //less or equal to 0.6
      int binphi = (int)(vvh*nsectors); //has values expect 0 and nsectors-1.
      //printf("binphi = %d\n", binphi);
      //So we use updist and downdist condition
	
      // check if the point influences also neighboring slabs
      double updist   = (double)((binphi+1)*yaxis)*dx - vvh;
      //printf("updist = %f\n", updist);
      double downdist = vvh - (double)(binphi*yaxis)*dx;
      //printf("downdist = %f\n", downdist);
      //
      (histo_send)[binphi]++;
      if(updist < w_supporth && updist >= 0.0)
	      (*histo_send)[binphi+1]++;
	
      if(downdist < w_supporth && binphi > 0 && downdist >= 0.0)
	      (*histo_send)[binphi-1]++;
    }

  printf("Just before first malloc()...\n");
  //
  // !! error is down here !!
  //x

  *sectorarray = malloc((nsectors+1)*sizeof(int*));
  if (*sectorarray == NULL) {
    fprintf(stderr, "Error allocating memory for sectorarray\n");
  exit(EXIT_FAILURE);
  }

  printf("Just before second malloc()...\n");
  int  *counter = calloc((nsectors+1), sizeof(int));
  if (counter == NULL) {
    fprintf(stderr, "Error allocating memory for counter\n");
    exit(EXIT_FAILURE);
  }

  
  for(int sec=0; sec<(nsectors+1); sec++)
    {
      printf("(histo_send)[%d] = %u\n", sec, (histo_send)[sec]);
      (*sectorarray)[sec] = (int*)calloc((*histo_send)[sec], sizeof(int));
    }


  printf("Just before _for_ loop\n");
  for (int iphi = 0; iphi < nmeasures; iphi++)
    {
      double vvh      = vv[iphi];
      int    binphi   = (int)(vvh*nsectors);
      double updist   = (double)((binphi+1)*yaxis)*dx - vvh;
      double downdist = vvh - (double)(binphi*yaxis)*dx;
      (*sectorarray)[binphi][counter[binphi]] = iphi;
      counter[binphi]++;
	
      if(updist < w_supporth && updist >= 0.0) {
	(*sectorarray)[binphi+1][counter[binphi+1]] = iphi; counter[binphi+1]++; };
      if(downdist < w_supporth && binphi > 0 && downdist >= 0.0) {
	(*sectorarray)[binphi-1][counter[binphi-1]] = iphi; counter[binphi-1]++; };
    }

  free( counter );
    
 #ifdef VERBOSE
  for (int iii=0; iii<nsectors+1; iii++)
    printf("HISTO %d %d %ld\n",rank, iii, histo_send[iii]);
 #endif
}




void wstack(
	    int num_w_planes,
	    int num_points,
	    int freq_per_chan,
	    int polarizations,
	    double* uu,
	    double* vv,
	    double* ww,
	    float* vis_real,
	    float* vis_img,
	    float* weight,
	    double dx,
	    double dw,
	    int w_support,
	    int grid_size_x,
	    int grid_size_y,
	    double* grid,
	    int num_threads,
	    int rank
    )
{
  int i;
  //int index;
  unsigned long long visindex;
  
  // initialize the convolution kernel
  // For simplicity, we use for the moment only the Gaussian kernel:
  int KernelLen = (w_support-1)/2;
  int increaseprecision = 5; // this number must be odd: increaseprecison*w_support must be odd (w_support must be odd)
  double std = 1.0;
  double std22 = 1.0/(2.0*std*std);
  double norm = std22/PI;
  double * convkernel = (double*)malloc(increaseprecision*w_support*sizeof(*convkernel));
  
  int n = increaseprecision*KernelLen, mid = n / 2;
  for (int i = 0; i != mid + 1; i++) {
      double term = (double)i/(double)increaseprecision;
      convkernel[mid + i] = sqrt(norm) * exp(-(term*term)*std22);
  }

  for (int i = 0; i != mid; i++) convkernel[i] = convkernel[n - 1 - i];


#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif

#if defined(ACCOMP) && (GPU_STACKING)
  omp_set_default_device(rank % omp_get_num_devices());
  myull Nvis = num_points*freq_per_chan*polarizations;
#pragma omp target teams distribute parallel for private(visindex) map(to:uu[0:num_points], vv[0:num_points], ww[0:num_points], vis_real[0:Nvis], vis_img[0:Nvis], weight[0:Nvis/freq_per_chan]) map(tofrom:grid[0:2*num_w_planes*grid_size_x*grid_size_y])
#else
#pragma omp parallel for private(visindex)
#endif

  printf("Before _for_ loop into _wstack_ function\n");
  for (i = 0; i < num_points; i++)
    {
#ifdef _OPENMP
      //int tid;
      //tid = omp_get_thread_num();
      //printf("%d\n",tid);
#endif

      visindex = i*freq_per_chan*polarizations;

      double sum = 0.0;
      int j, k;
      //if (i%1000 == 0)printf("%ld\n",i);

      /* Convert UV coordinates to grid coordinates. */
      double pos_u = uu[i] / dx;
      double pos_v = vv[i] / dx;
      double ww_i  = ww[i] / dw;
	
      int grid_w = (int)ww_i;
      int grid_u = (int)pos_u;
      int grid_v = (int)pos_v;

      // check the boundaries
      int jmin = (grid_u > KernelLen - 1) ? grid_u - KernelLen : 0;
      int jmax = (grid_u < grid_size_x - KernelLen) ? grid_u + KernelLen : grid_size_x - 1;
      int kmin = (grid_v > KernelLen - 1) ? grid_v - KernelLen : 0;
      int kmax = (grid_v < grid_size_y - KernelLen) ? grid_v + KernelLen : grid_size_y - 1;
      //printf("%d, %ld, %ld, %d, %ld, %ld\n",grid_u,jmin,jmax,grid_v,kmin,kmax);


      // Convolve this point onto the grid.
      for (k = kmin; k <= kmax; k++)
        {

	  double v_dist = (double)k+0.5 - pos_v;
	  //double v_dist = (double)k - pos_v;

	  for (j = jmin; j <= jmax; j++)
            {
	      double u_dist = (double)j+0.5 - pos_u;
	      //double u_dist = (double)j - pos_u;
	      int iKer = 2 * (j + k*grid_size_x + grid_w*grid_size_x*grid_size_y);
	      int jKer = (int)(increaseprecision * (fabs(u_dist+(double)KernelLen)));
	      int kKer = (int)(increaseprecision * (fabs(v_dist+(double)KernelLen)));

	      double conv_weight = convkernel[jKer]*convkernel[kKer];
	      // Loops over frequencies and polarizations
	      double add_term_real = 0.0;
	      double add_term_img = 0.0;
	      unsigned long long ifine = visindex;
	      // DAV: the following two loops are performend by each thread separately: no problems of race conditions
	      for (int ifreq=0; ifreq<freq_per_chan; ifreq++)
		{
		  int iweight = visindex/freq_per_chan;
		  for (int ipol=0; ipol<polarizations; ipol++)
		    {
                      if (!isnan(vis_real[ifine]))
			{
			  //printf("%f %ld\n",weight[iweight],iweight);
			  add_term_real += weight[iweight] * vis_real[ifine] * conv_weight;
			  add_term_img += weight[iweight] * vis_img[ifine] * conv_weight;
			  //if(vis_img[ifine]>1e10 || vis_img[ifine]<-1e10)printf("%f %f %f %f %ld %ld\n",vis_real[ifine],vis_img[ifine],weight[iweight],conv_weight,ifine,num_points*freq_per_chan*polarizations);
			}
		      ifine++;
		      iweight++;
		    }
	        }
	      // DAV: this is the critical call in terms of correctness of the results and of performance
#pragma omp atomic
	      grid[iKer] += add_term_real;
#pragma omp atomic
	      grid[iKer+1] += add_term_img;
            }
        }
	
    }
#if defined(ACCOMP) && (GPU_STACKING)
#pragma omp target exit data map(delete:uu[0:num_points], vv[0:num_points], ww[0:num_points], vis_real[0:Nvis], vis_img[0:Nvis], weight[0:Nvis/freq_per_chan], grid[0:2*num_w_planes*grid_size_x*grid_size_y])
#endif
}


void free_array( int *histo_send, int ***sectorarrays, int nsectors )

{ 
  for ( int i = nsectors-1; i > 0; i-- )
    free( (*sectorarrays)[i] );

  free( *sectorarrays );

  free( histo_send );
  
  return;	  
}







void gridding_data(
    double_t dx,
    double_t dw,
    int num_threads,
    int size,
    int rank,
    int xaxis,
    int yaxis,
    int size_of_grid,
    int num_w_planes,
    int w_support,
    double uvmin,
    double uvmax,
    int polarisations,
    int freq_per_chan,
    double* uu,
    double* vv,
    double* ww,
    double* grid,
    double* gridss,
    float* visreal,
    float* visimg,
    int* histo_send,
    float* weights,
    int ***sectorarray,
    MPI_Comm MYMYMPI_COMM
)
//
// actually performs the gridding of the data
//
  
{

  double shift = (double)(dx*yaxis);
    
  /*
  if( (size > 1) && (reduce_method == REDUCE_RING) )
    {
      memset( (char*)Me.win.ptr, 0, size_of_grid*sizeof(double)*1.1);                                                                               
      gridss = (double*)Me.win.ptr; //gridss must point to the right location [GL]
  
      memset( (char*)Me.fwin.ptr, 0, size_of_grid*sizeof(double)*1.1); //allocate the memory for the results [GL]
  
      if( Me.Rank[myHOST] == 0 )
	{
	  for( int tt = 1; tt < Me.Ntasks[myHOST]; tt++ )
	    memset( (char*)Me.swins[tt].ptr, 0, size_of_grid*sizeof(double)*1.1);
	}


      MPI_Barrier(MYMYMPI_COMM);
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
    */
  
  //timing_wt.kernel     = 0.0;
  //timing_wt.reduce     = 0.0;
  //timing_wt.reduce_mpi = 0.0;
  //timing_wt.reduce_sh  = 0.0;
  //timing_wt.compose    = 0.0;

  // calculate the resolution in radians
  double resolution = 1.0/MAX(fabs(uvmin),fabs(uvmax));
    
  // calculate the resolution in arcsec 
  double resolution_asec = (3600.0*180.0)/MAX(fabs(uvmin),fabs(uvmax))/PI;
  if ( rank == 0 )
    printf("RESOLUTION = %f rad, %f arcsec\n", resolution, resolution_asec);

  // find the largest value in histo_send[]
  //
  
  for (int isector = 0; isector < size; isector++)
    {
      //double start = CPU_TIME_wt;

      int Nsec            = histo_send[isector];
      int Nweightss       = Nsec*polarisations;
      unsigned long long Nvissec         = Nweightss*freq_per_chan;
      double_t *memory     = (double*) malloc ( (Nsec*3)*sizeof(double_t) +
						(Nvissec*2+Nweightss)*sizeof(float_t) );

      //if ( memory == NULL )
	//shutdown_wstacking(NOT_ENOUGH_MEM_STACKING, "Not enough memory for stacking", __FILE__, __LINE__);
  
      double_t *uus        = (double_t*) memory;
      double_t *vvs        = (double_t*) uus+Nsec;
      double_t *wws        = (double_t*) vvs+Nsec;
      float_t  *weightss   = (float_t*)((double_t*)wws+Nsec);
      float_t  *visreals   = (float_t*)weightss + Nweightss;
      float_t  *visimgs    = (float_t*)visreals + Nvissec;
  
      // select data for this sector
      int icount = 0;
      int ip = 0;
      int inu = 0;

      #warning "this loop should be threaded"
      #warning "the counter of this loop should not be int"
      for( int iphi = histo_send[isector]-1; iphi >=0 ; iphi--)
        {
	  
	  int ilocal = (*sectorarray)[isector][iphi];

	  uus[icount] = uu[ilocal];
	  vvs[icount] = vv[ilocal]-isector*shift;
	  wws[icount] = ww[ilocal];
	  for (int ipol=0; ipol<polarisations; ipol++)
	    {
	      weightss[ip] = weights[ilocal*polarisations+ipol];
	      ip++;
	    }
	  for (int ifreq=0; ifreq<polarisations*freq_per_chan; ifreq++)
	    {
	      visreals[inu] = visreal[ilocal*polarisations*freq_per_chan+ifreq];
	      visimgs[inu]  = visimg[ilocal*polarisations*freq_per_chan+ifreq];
	      inu++;
	    }
	  icount++;
	}
      
      double uumin = 1e20;
      double vvmin = 1e20;
      double uumax = -1e20;
      double vvmax = -1e20;

     #pragma omp parallel reduction( min: uumin, vvmin) reduction( max: uumax, vvmax) num_threads(num_threads)
      {
	double my_uumin = 1e20;
	double my_vvmin = 1e20;
	double my_uumax = -1e20;
	double my_vvmax = -1e20;

       #pragma omp for 
	for (int ipart=0; ipart<Nsec; ipart++)
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
      

      //timing_wt.compose += CPU_TIME_wt - start;
      
      // Make convolution on the grid

     #ifdef VERBOSE
      printf("Processing sector %ld\n",isector);
     #endif

      double *stacking_target_array;
      if ( size > 1 )
	stacking_target_array = gridss;
      else
	stacking_target_array = grid;

      //start = CPU_TIME_wt;

      printf("Calling _wstack_ function\n");
	    
     //We have to call different GPUs per MPI task!!! [GL]
      wstack(num_w_planes,
	     Nsec,
	     freq_per_chan,
	     polarisations,
	     uus,
	     vvs,
	     wws,
	     visreals,
	     visimgs,
	     weightss,
	     dx,
	     dw,
	     w_support,
	     xaxis,
	     yaxis,
	     stacking_target_array,
	     num_threads,
	     rank);


      //timing_wt.kernel += CPU_TIME_wt - start;
      
     #ifdef VERBOSE
      printf("Processed sector %ld\n",isector);
     #endif

      if( size > 1 )
	{
	  // Write grid in the corresponding remote slab
	  
	  int target_rank = (int)(isector % size);

	  //start = CPU_TIME_wt;


	  //Force to use MPI_Reduce when -fopenmp is not active
	 #ifdef _OPENMP
	  if( reduce_method == REDUCE_MPI )
	   
	    MPI_Reduce(gridss, grid, size_of_grid, MPI_DOUBLE, MPI_SUM, target_rank, MYMYMPI_COMM);
	  
	  else if ( reduce_method == REDUCE_RING )
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
	  MPI_Reduce(gridss, grid, size_of_grid, MPI_DOUBLE, MPI_SUM, target_rank, MYMYMPI_COMM);
	 #endif
	  
	  //timing_wt.reduce += CPU_TIME_wt - start;

	  // Go to next sector
	  memset ( gridss, 0, 2*num_w_planes*xaxis*yaxis * sizeof(double) );	  
	}	

      free(memory);
    }

  if ( size > 1 )
    {
      //double start = CPU_TIME_wt;
//      if ( reduce_method == REDUCE_RING)
//	if( (Me.Rank[HOSTS] >= 0) && (Me.Nhosts > 1 )) {
//  	  MPI_Waitall( Me.Ntasks[WORLD], requests, MPI_STATUSES_IGNORE);
//	 free(requests);}
      
      //timing_wt.reduce += CPU_TIME_wt - start;
      MPI_Barrier(MYMYMPI_COMM);
    }

  return;

}





void gridding(
    int rank,
    int size,
    int nmeasures,
    double* uu,
    double* vv,
    double* ww,
    double* grid,
    double* gridss,
    MPI_Comm MYMPI_COMM,
    int num_threads,
    int grid_size_x,
    int grid_size_y,
    int w_support,
    int num_w_planes,
    int polarisations,
    int freq_per_chan,
    float* visreal,
    float* visimg,
    float* weights,
    double uvmin,
    double uvmax
)
{

  if(rank == 0)
    printf("RICK GRIDDING DATA\n");

  //double start = CPU_TIME_wt;

  int xaxis = grid_size_x;
  int yaxis = grid_size_y/size;
  int size_of_grid = 2*num_w_planes*xaxis*yaxis;
  
  double dx = 1.0/(double)grid_size_x;
  //printf("w_support : %d\n", w_support);
  double dw = 1.0/(double)num_w_planes;
  double w_supporth = (double)((w_support-1)/2)*dx;
  //printf("w_supporth outside : %f\n", w_supporth);

  int *histo_send = NULL;
  int **sectorarray = NULL;

/*
  test_malloc(
    nmeasures,
    &histo_send,
    &sectorarray);
*/

  printf("Calling _initialize_array_ function\n");
  
  // Create histograms and linked lists
  
  // Initialize linked list
  initialize_array(
    size,
    nmeasures,
    w_supporth,
    vv,
    yaxis,
    dx,
    &histo_send,
    &sectorarray);

  //timing_wt.init += CPU_TIME_wt - start;

  printf("Calling _gridding_data_ function\n");
  
  //Sector and Gridding data
  gridding_data(
    dx,
    dw,
    num_threads,
    size,
    rank,
    xaxis,
    yaxis,
    size_of_grid,
    num_w_planes,
    w_support,
    uvmin,
    uvmax,
    polarisations,
    freq_per_chan,
    uu,
    vv,
    ww,
    grid,
    gridss,
    visreal,
    visimg,
    histo_send,
    weights,
    &sectorarray,
    MYMPI_COMM
  );
  
  //timing_wt.gridding += CPU_TIME_wt - start;
  
  free_array( &histo_send, &sectorarray, size );
  
  MPI_Barrier(MYMPI_COMM);
  
  return;
}
