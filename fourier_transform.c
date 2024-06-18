#include "allvars_nccl.h"
#include "proto.h"

#if defined(CUFFTMP)
void cuda_fft( int, int, int, int, int, double*, double*, int, MPI_Comm );
#endif


// ------------------------------------
#if defined(USE_FFTW) && !defined(CUFFTMP)      //  PERFORM FFT on CPU with FFTW
						// ------------------------------------ 

void fftw_data ( void )
{

  // FFT transform the data (using distributed FFTW)
  if(rank == 0)printf("PERFORMING FFT\n");

  double start = CPU_TIME_wt;
	
  fftw_plan plan;
  fftw_complex *fftwgrid;
  ptrdiff_t alloc_local, local_n0, local_0_start;
  double norm = 1.0/(double)(param.grid_size_x*param.grid_size_y);

  //Use the hybrid MPI-OpenMP FFTW
 #ifdef HYBRID_FFTW
  fftw_plan_with_nthreads(param.num_threads);
 #endif
  // map the 1D array of complex visibilities to a 2D array required by FFTW (complex[*][2])
  // x is the direction of contiguous data and maps to the second parameter
  // y is the parallelized direction and corresponds to the first parameter (--> n0)
  // and perform the FFT per w plane
  alloc_local = fftw_mpi_local_size_2d(param.grid_size_y, param.grid_size_x, MPI_COMM_WORLD,&local_n0, &local_0_start);
  fftwgrid = fftw_alloc_complex(alloc_local);
  plan = fftw_mpi_plan_dft_2d(param.grid_size_y, param.grid_size_x, fftwgrid, fftwgrid, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

  myuint fftwindex = 0;
  myuint fftwindex2D = 0;
  for (int iw=0; iw<param.num_w_planes; iw++)
    {
      //printf("FFTing plan %d\n",iw);
      //select the w-plane to transform

     #ifdef HYBRID_FFTW
     #pragma omp parallel for collapse(2) num_threads(param.num_threads)
     #endif
      for (int iv=0; iv<yaxis; iv++)
	{
	  for (int iu=0; iu<xaxis; iu++)
	    {
	      fftwindex2D = iu + iv*xaxis;
	      fftwindex = 2*(fftwindex2D + iw*xaxis*yaxis);
	      fftwgrid[fftwindex2D][0] = grid[fftwindex];
	      fftwgrid[fftwindex2D][1] = grid[fftwindex+1];
	    }
	}

      // do the transform for each w-plane        
      fftw_execute(plan);

      // save the transformed w-plane
	    
     #ifdef HYBRID_FFTW
     #pragma omp parallel for collapse(2) num_threads(param.num_threads)
     #endif
      for (int iv=0; iv<yaxis; iv++)
	{
	  for (int iu=0; iu<xaxis; iu++)
	    {
	      fftwindex2D = iu + iv*xaxis;
	      fftwindex = 2*(fftwindex2D + iw*xaxis*yaxis);
	      gridss[fftwindex] = norm*fftwgrid[fftwindex2D][0];
	      gridss[fftwindex+1] = norm*fftwgrid[fftwindex2D][1];
	    }
	}

    }

 #ifdef HYBRID_FFTW
  fftw_cleanup_threads();
 #endif
  fftw_destroy_plan(plan);
  fftw_free(fftwgrid);

		
  MPI_Barrier(MPI_COMM_WORLD);
        

  timing_wt.fftw += CPU_TIME_wt - start;

  return;
}
                                         // ------------------------------------
#else                                    //  PERFORM FFT ON GPU USING CUFFTMP
					 // ------------------------------------

void fftw_data ( void )
{

  // FFT transform the data (using distributed FFTW)
  if(rank == 0)printf("PERFORMING FFT\n");

  // FFT transform the data using cuFFT                                                                                                    
  if(rank==0)printf("PERFORMING CUDA FFT\n");

  double start = CPU_TIME_wt;

  
  cuda_fft(
	   param.num_w_planes,
	   param.grid_size_x,
	   param.grid_size_y,
	   xaxis,
	   yaxis,
	   grid_gpu,
	   gridss_gpu,
	   rank,
	   MPI_COMM_WORLD);

  timing_wt.cufftmp += CPU_TIME_wt - start;

  return;
}


#endif                                  // END OF FFT SELECTION

   

void write_fftw_data(){

 #ifdef USE_FFTW
 #ifdef WRITE_DATA
  // Write results let's skip this part for the moment

  MPI_Win writewin;
  MPI_Win_create(gridss, size_of_grid*sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &writewin);
  MPI_Win_fence(0,writewin);
  if (rank == 0)
    {
      printf("WRITING FFT TRANSFORMED DATA\n");
      file.pFilereal = fopen (out.fftfile_writedata1,"wb");
      file.pFileimg = fopen (out.fftfile_writedata2,"wb");
      for (int isector=0; isector<nsectors; isector++)
	{
	  MPI_Win_lock(MPI_LOCK_SHARED,isector,0,writewin);
	  MPI_Get(gridss_w,size_of_grid,MPI_DOUBLE,isector,0,size_of_grid,MPI_DOUBLE,writewin);
	  MPI_Win_unlock(isector,writewin);
	  for (myuint i=0; i<size_of_grid/2; i++)
	    {
	      gridss_real[i] = gridss_w[2*i];
	      gridss_img[i] = gridss_w[2*i+1];
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
		    }
	    } 
	  else 
	    {
	      fwrite(gridss_real, size_of_grid/2, sizeof(double), file.pFilereal);
	      fwrite(gridss_img, size_of_grid/2, sizeof(double), file.pFileimg);
	    }

	}
      /*
	for (int iw=0; iw<param.num_w_planes; iw++)
	for (int iv=0; iv<grid_size_y; iv++)
	for (int iu=0; iu<grid_size_x; iu++)
	{
	int isector = 0;
	myuint index = 2*(iu + iv*grid_size_x + iw*grid_size_x*grid_size_y);
	double v_norm = sqrt(gridtot[index]*gridtot[index]+gridtot[index+1]*gridtot[index+1]);
	fprintf (file.pFile, "%d %d %d %f %f %f\n", iu,iv,iw,gridtot[index],gridtot[index+1],v_norm);
	}
      */

      fclose(file.pFilereal);
      fclose(file.pFileimg);
    }
  MPI_Win_fence(0,writewin);
  MPI_Win_free(&writewin);
  MPI_Barrier(MPI_COMM_WORLD);
 #endif //WRITE_DATA


  // Phase correction  

  double start = CPU_TIME_wt;
	       
  if(rank == 0)printf("PHASE CORRECTION\n");
  double* image_real = (double*) calloc(xaxis*yaxis,sizeof(double));
  double* image_imag = (double*) calloc(xaxis*yaxis,sizeof(double));

#ifdef CUFFTMP
  phase_correction(gridss_gpu,image_real,image_imag,xaxis,yaxis,param.num_w_planes,param.grid_size_x,param.grid_size_y,resolution,metaData.wmin,metaData.wmax,param.num_threads,rank);
#else
  phase_correction(gridss,image_real,image_imag,xaxis,yaxis,param.num_w_planes,param.grid_size_x,param.grid_size_y,resolution,metaData.wmin,metaData.wmax,param.num_threads,rank);
#endif
  
  timing_wt.phase += CPU_TIME_wt - start;
  
#ifdef WRITE_IMAGE

  double start_image = CPU_TIME_wt;
  
  if(rank == 0)
    {

     #ifdef FITSIO
      printf("REMOVING RESIDUAL FITS FILE\n");
      remove(testfitsreal);
      remove(testfitsimag);


      printf("FITS CREATION\n");
      status = 0;

      fits_create_file(&fptrimg, testfitsimag, &status);
      fits_create_img(fptrimg, DOUBLE_IMG, naxis, naxes, &status);
      fits_close_file(fptrimg, &status);

      status = 0;

      fits_create_file(&fptreal, testfitsreal, &status);
      fits_create_img(fptreal, DOUBLE_IMG, naxis, naxes, &status);
      fits_close_file(fptreal, &status);
     #endif
      
      file.pFilereal = fopen (out.fftfile2,"wb");
      file.pFileimg = fopen (out.fftfile3,"wb");
      fclose(file.pFilereal);
      fclose(file.pFileimg);
    }

  MPI_Barrier(MPI_COMM_WORLD);

  if(rank == 0)printf("WRITING IMAGE\n");

#ifdef FITSIO
  myuint * fpixel = (myuint *) malloc(sizeof(myuint)*naxis);
  myuint * lpixel = (myuint *) malloc(sizeof(myuint)*naxis);
#endif

#ifdef FITSIO

  fpixel[0] = 1;
  fpixel[1] = rank*yaxis+1;
  lpixel[0] = xaxis;
  lpixel[1] = (rank+1)*yaxis;

  status = 0;
  fits_open_image(&fptreal, testfitsreal, READWRITE, &status);
  fits_write_subset(fptreal, TDOUBLE, fpixel, lpixel, image_real, &status);
  fits_close_file(fptreal, &status);

  status = 0;
  fits_open_image(&fptrimg, testfitsimag, READWRITE, &status);
  fits_write_subset(fptrimg, TDOUBLE, fpixel, lpixel, image_imag, &status);
  fits_close_file(fptrimg, &status);

#endif //FITSIO

  file.pFilereal = fopen (out.fftfile2,"wb");
  file.pFileimg = fopen (out.fftfile3,"wb");

  long global_index = rank*(xaxis*yaxis)*sizeof(long);

  fseek(file.pFilereal, global_index, SEEK_SET);
  fwrite(image_real, xaxis*yaxis, sizeof(double), file.pFilereal);
  fseek(file.pFileimg, global_index, SEEK_SET);
  fwrite(image_imag, xaxis*yaxis, sizeof(double), file.pFileimg); 

  fclose(file.pFilereal);
  fclose(file.pFileimg);
  
  MPI_Barrier(MPI_COMM_WORLD);

  timing_wt.write += CPU_TIME_wt - start_image;

#endif //WRITE_IMAGE

#endif  //FFTW
}
