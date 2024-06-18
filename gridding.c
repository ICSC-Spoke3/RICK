#include "allvars.h"
#include "proto.h"




void free_array       ( myuint *, myuint **, int );
void initialize_array ( void );
void gridding_data    ( void );



void gridding()
{

  if(rank == 0)
    printf("GRIDDING DATA\n");

  double start = CPU_TIME_wt;
  
 #ifdef NORMALIZE_UVW

  if (rank==0)
    printf("NORMALIZING DATA\n");

  typedef struct {
    double u;
    double v;
    double w; } cmp_t;

  
  cmp_t getmin = { 1e20, 1e20, 1e20 };
  cmp_t getmax = { 0 };

 #pragma omp parallel num_threads(param.num_threads)
  {
    cmp_t mygetmin = { 1e20, 1e20, 1e20 };
    cmp_t mygetmax = { 0 };

   #pragma omp for
    for (myuint inorm=0; inorm<metaData.Nmeasures; inorm++)
      {
	mygetmin.u = MIN(mygetmin.u, data.uu[inorm]);
	mygetmin.v = MIN(mygetmin.v, data.vv[inorm]);
	mygetmin.w = MIN(mygetmin.w, data.ww[inorm]);
	
	mygetmax.u = MAX(mygetmax.u, data.uu[inorm]);
	mygetmax.v = MAX(mygetmax.v, data.vv[inorm]);
	mygetmax.w = MAX(mygetmax.w, data.ww[inorm]);
      }
    
   #pragma omp critical (getmin_u)
    getmin.u = MIN( mygetmin.u, getmin.u );
   #pragma omp critical (getmin_v)
    getmin.v = MIN( mygetmin.v, getmin.v );
   #pragma omp critical (getmin_w)
    getmin.w = MIN( mygetmin.w, getmin.w );
    
   #pragma omp critical (getmax_u)
    getmax.u = MAX( mygetmax.u, getmax.u );
   #pragma omp critical (getmax_v)
    getmax.v = MAX( mygetmax.v, getmax.v );
   #pragma omp critical (getmax_w)
    getmax.w = MAX( mygetmax.w, getmax.w );
  }

  MPI_Allreduce(MPI_IN_PLACE, &getmin, 3, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &getmax, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  double offset = 0.001;
  double ming = MAX(fabs(getmin.u), fabs(getmin.v));
  double maxg = MAX(fabs(getmax.u), fabs(getmax.v));
  maxg = MAX(maxg,ming);
  maxg = maxg + offset*maxg;

 #pragma omp parallel for num_threads(param.num_threads)
  for (myuint inorm=0; inorm < metaData.Nmeasures; inorm++)
    {
      data.uu[inorm] = (data.uu[inorm]+maxg) / (2.0*maxg);
      data.vv[inorm] = (data.vv[inorm]+maxg) / (2.0*maxg);
      data.ww[inorm] = (data.ww[inorm]-getmin.w) / (getmax.w-getmin.w);
    }
 #endif
  
  // Create histograms and linked lists
  
  // Initialize linked list
  initialize_array();

  timing_wt.init += CPU_TIME_wt - start;
  
  //Sector and Gridding data
  gridding_data();
  
  timing_wt.gridding += CPU_TIME_wt - start;
  
  free_array( histo_send, sectorarray, nsectors );
  
  MPI_Barrier(MYMPI_COMM_WORLD);
  
  return;
}


/* ----------------------------------------------------------- *
   |                                                           |
   | internal routines called from gridding                    |
   | - initialize memory                                       |
   | - gridding                                                |
   |   - call of stack routine                                 |
   |   - reduce                                                |
   |                                                           |
   ----------------------------------------------------------- * */


//   .....................................................................
//
void free_array( myuint *histo_send, myuint **sectorarrays, int nsectors )
//
// releases memory allocated for gridding and reduce ops
//
  
{ 
  
  for ( myuint i = nsectors-1; i > 0; i-- )
    free( sectorarrays[i] );

  free( sectorarrays );

  free( histo_send );
  
  return;	  
}



//   .....................................................................
//
void initialize_array()
//
// allocate the memory and initialize
// some values
//
  
{

  histo_send = (myuint*) calloc(nsectors+1, sizeof(myuint));

  for (myuint iphi = 0; iphi < metaData.Nmeasures; iphi++)
    {
      double vvh = data.vv[iphi];              //less or equal to 0.6
      int binphi = (int)(vvh*nsectors); //has values expect 0 and nsectors-1.
      //So we use updist and downdist condition
	
      // check if the point influences also neighboring slabs
      double updist   = (double)((binphi+1)*yaxis)*dx - vvh;
      double downdist = vvh - (double)(binphi*yaxis)*dx;
      //
      histo_send[binphi]++;
      if(updist < w_supporth && updist >= 0.0)
	histo_send[binphi+1]++;
	
      if(downdist < w_supporth && binphi > 0 && downdist >= 0.0)
	histo_send[binphi-1]++;
    }

  sectorarray = (myuint**)malloc ((nsectors+1) * sizeof(myuint*));
  myuint  *counter     = (myuint*) calloc (nsectors+1, sizeof(myuint));
  for(myuint sec=0; sec<(nsectors+1); sec++)
    {
      sectorarray[sec] = (myuint*)malloc(histo_send[sec]*sizeof(myuint));
    }
    
    
  for (myuint iphi = 0; iphi < metaData.Nmeasures; iphi++)
    {
      double vvh      = data.vv[iphi];
      int    binphi   = (int)(vvh*nsectors);
      double updist   = (double)((binphi+1)*yaxis)*dx - vvh;
      double downdist = vvh - (double)(binphi*yaxis)*dx;
      sectorarray[binphi][counter[binphi]] = iphi;
      counter[binphi]++;
	
      if(updist < w_supporth && updist >= 0.0) {
	sectorarray[binphi+1][counter[binphi+1]] = iphi; counter[binphi+1]++; };
      if(downdist < w_supporth && binphi > 0 && downdist >= 0.0) {
	sectorarray[binphi-1][counter[binphi-1]] = iphi; counter[binphi-1]++; };
    }
     

  free( counter );
    
 #ifdef VERBOSE
  for (int iii=0; iii<nsectors+1; iii++)
    printf("HISTO %d %d %ld\n",rank, iii, histo_send[iii]);
 #endif
}







//   .....................................................................
//
void write_gridded_data()
{

 #ifdef WRITE_DATA

  // Write gridded results
  MPI_Win slabwin;
  MPI_Win_create(gridss, size_of_grid*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &slabwin);
  MPI_Win_fence(0,slabwin);

  
  if (rank == 0)
    {
      printf("WRITING GRIDDED DATA\n");
      file.pFilereal = fopen (out.gridded_writedata1,"wb");
      file.pFileimg = fopen (out.gridded_writedata2,"wb");

      for (int isector=0; isector<nsectors; isector++)
	{
	 #ifdef RING //Let the MPI_Get copy from the right location (Results must be checked!) [GL]
	  MPI_Get(gridss,size_of_grid,MPI_DOUBLE,isector,0,size_of_grid,MPI_DOUBLE,Me.win.win);
	 #else
	  MPI_Win_lock(MPI_LOCK_SHARED,isector,0,slabwin);
	  MPI_Get(gridss,size_of_grid,MPI_DOUBLE,isector,0,size_of_grid,MPI_DOUBLE,slabwin);
	  MPI_Win_unlock(isector,slabwin);
	 #endif
	  for (myuint i=0; i<size_of_grid/2; i++)
	    {
	      gridss_real[i] = gridss[2*i];
	      gridss_img[i] = gridss[2*i+1];
	    }
	  if (param.num_w_planes > 1)
	    {
	      for (int iw=0; iw<param.num_w_planes; iw++)
		for (int iv=0; iv<yaxis; iv++)
		  for (int iu=0; iu<xaxis; iu++)
		    {
		      myuint global_index = (iu + (iv+isector*yaxis)*xaxis + iw*param.grid_size_x*param.grid_size_y)*sizeof(double);
		      myuint index = iu + iv*xaxis + iw*xaxis*yaxis;
		      fseek(file.pFilereal, global_index, SEEK_SET);
		      fwrite(&gridss_real[index], 1, sizeof(double), file.pFilereal);
		    }
	      for (int iw=0; iw<param.num_w_planes; iw++)
		for (int iv=0; iv<yaxis; iv++)
		  for (int iu=0; iu<xaxis; iu++)
		    {
		      myuint global_index = (iu + (iv+isector*yaxis)*xaxis + iw*param.grid_size_x*param.grid_size_y)*sizeof(double);
		      myuint index = iu + iv*xaxis + iw*xaxis*yaxis;
		      fseek(file.pFileimg, global_index, SEEK_SET);
		      fwrite(&gridss_img[index], 1, sizeof(double), file.pFileimg);
		      //double v_norm = sqrt(gridss[index]*gridss[index]+gridss[index+1]*gridss[index+1]);
		      //fprintf (file.pFile, "%d %d %d %f %f %f\n", iu,isector*yaxis+iv,iw,gridss[index],gridss[index+1],v_norm);
		    }
	      
	    }
	  else
	    {
	      for (int iw=0; iw<param.num_w_planes; iw++)
		{
		  myuint global_index = (xaxis*isector*yaxis + iw*param.grid_size_x*param.grid_size_y)*sizeof(double);
		  myuint index = iw*xaxis*yaxis;
		  fseek(file.pFilereal, global_index, SEEK_SET);
		  fwrite(&gridss_real[index], xaxis*yaxis, sizeof(double), file.pFilereal);
		  fseek(file.pFileimg, global_index, SEEK_SET);
		  fwrite(&gridss_img[index], xaxis*yaxis, sizeof(double), file.pFileimg);
		}
	    }
	}
      fclose(file.pFilereal);
      fclose(file.pFileimg);
    }
  
  MPI_Win_fence(0,slabwin);
  MPI_Win_free(&slabwin);
  MPI_Barrier(MPI_COMM_WORLD);
 #endif //WRITE_DATA 
}
