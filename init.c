
#include "allvars.h"
#include "proto.h"

void init(int index)
{

  double begin = CPU_TIME_wt;
  
  // DAV: the corresponding KernelLen is calculated within the wstack function. It can be anyway hardcoded for optimization
  dx = 1.0/(double)param.grid_size_x;
  dw = 1.0/(double)param.num_w_planes;
  w_supporth = (double)((param.w_support-1)/2)*dx;
                            
  // MESH SIZE
  int local_grid_size_x;
  int local_grid_size_y;
   
  // set the local size of the image
  nsectors          = size;
  local_grid_size_x = param.grid_size_x;  
  local_grid_size_y = param.grid_size_y/nsectors;

  // LOCAL grid size
   xaxis = local_grid_size_x;
   yaxis = local_grid_size_y;


   //Initialize the NUMA region
   if( param.reduce_method == REDUCE_RING )
     {
       numa_init( rank, size, &MYMPI_COMM_WORLD, &Me );
       numa_expose(&Me,verbose_level);
     }
   
   // INPUT FILES (only the first ndatasets entries are used)
   strcpy(datapath,param.datapath_multi[index]);
   sprintf(num_buf, "%d", index);
   
   //Changing the output file names
   op_filename();
             
   // Read metadata
   fileName(datapath, in.metafile);
   readMetaData(filename);

   // Local Calculation
   metaData_calculation();

   // Allocate Data Buffer
   allocate_memory();
  
   // Reading Data
   readData();
 
   MPI_Barrier(MPI_COMM_WORLD);
   
   timing_wt.setup = CPU_TIME_wt - begin;

   return;
}

void op_filename() {

  if(rank == 0)
    {   
   	strcpy(buf, num_buf);
   	strcat(buf, outparam.outfile);
   	strcpy(out.outfile, buf);
   
   	strcpy(buf, num_buf);
   	strcat(buf, outparam.outfile1);
   	strcpy(out.outfile1, buf); 

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.outfile2);
   	strcpy(out.outfile2, buf);

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.outfile3);
   	strcpy(out.outfile3, buf);

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.fftfile);
   	strcpy(out.fftfile, buf);
#ifdef WRITE_DATA
        strcpy(buf, num_buf);
        strcat(buf, outparam.gridded_writedata1);
        strcpy(out.gridded_writedata1, buf);

        strcpy(buf, num_buf);
        strcat(buf, outparam.gridded_writedata2);
        strcpy(out.gridded_writedata2, buf);

        strcpy(buf, num_buf);
        strcat(buf, outparam.fftfile_writedata1);
        strcpy(out.fftfile_writedata1, buf);

        strcpy(buf, num_buf);
        strcat(buf, outparam.fftfile_writedata2);
        strcpy(out.fftfile_writedata2, buf);
#endif
   	strcpy(buf, num_buf);
   	strcat(buf, outparam.fftfile2);
   	strcpy(out.fftfile2, buf);

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.fftfile3);
   	strcpy(out.fftfile3, buf);
    
   	strcpy(buf, num_buf);
   	strcat(buf, outparam.logfile);
   	strcpy(out.logfile, buf);

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.extension);
   	strcpy(out.extension, buf);

   	strcpy(buf, num_buf);
   	strcat(buf, outparam.timingfile);
   	strcpy(out.timingfile, buf);
    }

  /* Communicating the relevent parameters to the other process */
  MPI_Bcast(&out, sizeof(struct op), MPI_BYTE, 0, MPI_COMM_WORLD);

}

void read_parameter_file(char *fname)
{
  int error = 0;
  
  if(rank == 0)
    {
      
      if( (file.pFile = fopen (fname,"r")) != NULL )
   	{
	  char buf1[30], buf2[100], buf3[30], num[30];
	  int i = 1;
	  while(fscanf(file.pFile, "%s" "%s", buf1, buf2) != EOF)
	    {
	      if(strcmp(buf1, "num_threads") == 0)
		{
		  param.num_threads = atoi(buf2);
		}
	      if(strcmp(buf1, "Datapath1") == 0)
		{
		  strcpy(param.datapath_multi[0], buf2);
		  i++;
		}
	      if(strcmp(buf1, "ndatasets") == 0)
		{
		  param.ndatasets = atoi(buf2);
		}
	      if(strcmp(buf1, "w_support") == 0)
		{
		  param.w_support = atoi(buf2);
		}
	      if(strcmp(buf1, "reduce_method") == 0)
		{
		  param.reduce_method = atoi(buf2);
		}
	      if(strcmp(buf1, "grid_size_x") == 0)
		{
		  param.grid_size_x = atoi(buf2);
		}
	      if(strcmp(buf1, "grid_size_y") == 0)
		{
		  param.grid_size_y = atoi(buf2);
		}
	      if(strcmp(buf1, "num_w_planes") == 0)
		{
		  param.num_w_planes = atoi(buf2);
		}
	      if(strcmp(buf1, "ufile") == 0)
		{
		  strcpy(in.ufile, buf2);
		}
	      if(strcmp(buf1, "vfile") == 0)
		{
		  strcpy(in.vfile, buf2);
		}
	      if(strcmp(buf1, "wfile") == 0)
		{
		  strcpy(in.wfile, buf2);
		}
	      if(strcmp(buf1, "weightsfile") == 0)
		{
		  strcpy(in.weightsfile, buf2);
		}
	      if(strcmp(buf1, "visrealfile") == 0)
		{
		  strcpy(in.visrealfile, buf2);
		}
	      if(strcmp(buf1, "visimgfile") == 0)
		{
		  strcpy(in.visimgfile, buf2);
		}
	      if(strcmp(buf1, "metafile") == 0)
		{
		  strcpy(in.metafile, buf2);
		}
	      if(strcmp(buf1, "outfile") == 0)
		{
		  strcpy(outparam.outfile, buf2);
		}
	      if(strcmp(buf1, "outfile1") == 0)
		{
		  strcpy(outparam.outfile1, buf2);
		}
	      if(strcmp(buf1, "outfile2") == 0)
		{
		  strcpy(outparam.outfile2, buf2);
		}
	      if(strcmp(buf1, "outfile3") == 0)
		{
		  strcpy(outparam.outfile3, buf2);
		}
	      if(strcmp(buf1, "fftfile") == 0)
		{
		  strcpy(outparam.fftfile, buf2);
		}
#ifdef WRITE_DATA
	      if(strcmp(buf1, "gridded_writedata1") == 0)
                {
                  strcpy(outparam.gridded_writedata1, buf2);
                }
	      if(strcmp(buf1, "gridded_writedata2") == 0)
                {
                  strcpy(outparam.gridded_writedata2, buf2);
                }
	      if(strcmp(buf1, "fftfile_writedata1") == 0)
                {
                  strcpy(outparam.fftfile_writedata1, buf2);
                }
	      if(strcmp(buf1, "fftfile_writedata2") == 0)
                {
                  strcpy(outparam.fftfile_writedata2, buf2);
                }
#endif
	      if(strcmp(buf1, "fftfile2") == 0)
		{
		  strcpy(outparam.fftfile2, buf2);
		}
	      if(strcmp(buf1, "fftfile3") == 0)
		{
		  strcpy(outparam.fftfile3, buf2);
		}
	      if(strcmp(buf1, "logfile") == 0)
		{
		  strcpy(outparam.logfile, buf2);
		}
	      if(strcmp(buf1, "extension") == 0)
		{
		  strcpy(outparam.extension, buf2);
		}
	      if(strcmp(buf1, "timingfile") == 0)
		{
		  strcpy(outparam.timingfile, buf2);
		}
	      if(param.ndatasets > 1)
		{
                   
		  sprintf(num, "%d", i);
		  strcat(strcpy(buf3,"Datapath"),num);
		  if(strcmp(buf1,buf3) == 0)
		    {
		      strcpy(param.datapath_multi[i-1], buf2);
		      i++;
		    } 
		}
	    }
	  fclose(file.pFile);
      
	}
      else
	error = 1;
    }

  /* Communicating the relevent parameters to the other process */

  MPI_Bcast(&error, 1, MPI_INT, 0, MYMPI_COMM_WORLD);

  if( error )
    shutdown_wstacking(ERR_IN_PARAMFILE, "I/O error while reading paramfile\n", __FILE__, __LINE__);
  
  MPI_Bcast(&in,       sizeof(struct ip), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&outparam, sizeof(struct op), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&param,    sizeof(struct parameter), MPI_BYTE, 0, MPI_COMM_WORLD);

}


void fileName(char datapath[900], char file[30]) {
     strcpy(filename,datapath);
     strcat(filename,file);
}


void readMetaData(char fileLocal[1000])
{
  if(rank == 0) 
    {
      if( (file.pFile = fopen (fileLocal,"r")) != NULL )
        {
	  int ret = 0;
	  ret += fscanf(file.pFile, "%u", &metaData.Nmeasures);
	  ret += fscanf(file.pFile, "%lu", &metaData.Nvis);
	  ret += fscanf(file.pFile, "%u", &metaData.freq_per_chan);
	  ret += fscanf(file.pFile, "%u", &metaData.polarisations);
	  ret += fscanf(file.pFile, "%u", &metaData.Ntimes);
	  ret += fscanf(file.pFile, "%lf", &metaData.dt);
	  ret += fscanf(file.pFile, "%lf", &metaData.thours);
	  ret += fscanf(file.pFile, "%u", &metaData.baselines);
	  ret += fscanf(file.pFile, "%lf", &metaData.uvmin);
	  ret += fscanf(file.pFile, "%lf", &metaData.uvmax);
	  ret += fscanf(file.pFile, "%lf", &metaData.wmin);
	  ret += fscanf(file.pFile, "%lf", &metaData.wmax);
	  fclose(file.pFile);
        } 
      else
        {
	  printf("error opening meta file");
	  exit(1);
        }
    }
      
  /* Communicating the relevent parameters to the other process */

  MPI_Bcast(&metaData, sizeof(struct meta), MPI_BYTE, 0, MPI_COMM_WORLD);
     
  return;      
}

void metaData_calculation() {
   
     int nsub = 1000;
     if ( rank == 0 ) printf("Subtracting last %d measurements\n",nsub);
     metaData.Nmeasures = metaData.Nmeasures-nsub;
     metaData.Nvis = metaData.Nmeasures*metaData.freq_per_chan*metaData.polarisations;
      // calculate the coordinates of the center
     double uvshift = metaData.uvmin/(metaData.uvmax-metaData.uvmin);

     if (rank == 0)
     {
          printf("N. measurements %u\n",metaData.Nmeasures);
	  printf("Channels %u\n", metaData.freq_per_chan);
	  printf("Correlations %u\n", metaData.polarisations);
          printf("N. visibilities %lu\n",metaData.Nvis);
     }

     // Set temporary local size of points
     myuint nm_pe = (myuint)(metaData.Nmeasures/size);
     myuint remaining = metaData.Nmeasures%size;

     startrow = rank*nm_pe;
     if (rank == size-1)nm_pe = nm_pe+remaining;

     //myuint Nmeasures_tot = metaData.Nmeasures;
     metaData.Nmeasures = nm_pe;
     //unsigned long Nvis_tot = metaData.Nvis;
     metaData.Nvis = metaData.Nmeasures*metaData.freq_per_chan*metaData.polarisations;
     metaData.Nweights = metaData.Nmeasures*metaData.polarisations;

     #ifdef VERBOSE
          printf("N. measurements on %d %ld\n",rank,metaData.Nmeasures);
          printf("N. visibilities on %d %ld\n",rank,metaData.Nvis);
     #endif

}

void allocate_memory() {


     // DAV: all these arrays can be allocatate statically for the sake of optimization. However be careful that if MPI is used
     // all the sizes are rescaled by the number of MPI tasks
     //  Allocate arrays
     
     data.uu = (double*) calloc(metaData.Nmeasures,sizeof(double));
     data.vv = (double*) calloc(metaData.Nmeasures,sizeof(double));
     data.ww = (double*) calloc(metaData.Nmeasures,sizeof(double));
     data.weights = (float*) calloc(metaData.Nweights,sizeof(float));
     data.visreal = (float*) calloc(metaData.Nvis,sizeof(float));
     data.visimg = (float*) calloc(metaData.Nvis,sizeof(float));


     // Create sector grid
     
     size_of_grid = 2*param.num_w_planes*xaxis*yaxis;

     int size_of_grid_pointers = (3 + 2*(param.reduce_method != REDUCE_RING)) * size_of_grid;     
     grid_pointers = (double*)calloc( size_of_grid_pointers, sizeof(double));

     if ( param.reduce_method != REDUCE_RING )
       {
	 gridss   = grid_pointers;
	 grid     = gridss + size_of_grid;
	 gridss_w = gridss + size_of_grid;
       }
     else
       {
	 gridss_w = grid_pointers;
	 numa_allocate_shared_windows( &Me, size_of_grid*sizeof(double)*1.1, sizeof(double)*1.1 );
	 gridss = (double*)Me.win.ptr;  // gridss must point to the right location [GL]
	 grid   = (double*)Me.fwin.ptr; // let grid point to the right memory location [GL]
       }
          
     gridss_real = gridss_w + size_of_grid;
     gridss_img  = gridss_real + size_of_grid / 2;
          
}

void readData()
{
  int ret;
  
  if(rank == 0) 
    printf("READING DATA\n");
  
  
  fileName(datapath, in.ufile);
  if( (file.pFile = fopen (filename,"rb")) != NULL )
    {
      fseek (file.pFile,startrow*sizeof(double),SEEK_SET);
      ret = fread(data.uu,metaData.Nmeasures*sizeof(double),1,file.pFile);
      fclose(file.pFile);
    }
  
  else
    {
      printf("error opening ucoord file");
      exit(1);
    }
  
  fileName(datapath, in.vfile);
  if( (file.pFile = fopen (filename,"rb")) != NULL )
    {
      fseek (file.pFile,startrow*sizeof(double),SEEK_SET);
      ret = fread(data.vv,metaData.Nmeasures*sizeof(double),1,file.pFile);
      fclose(file.pFile);
    }
  else
    {
      printf("error opening vcoord file");
      exit(1);
    }
  
  fileName(datapath, in.wfile);
  if( (file.pFile = fopen (filename,"rb")) != NULL )
    {
      fseek (file.pFile,startrow*sizeof(double),SEEK_SET);
      ret = fread(data.ww,metaData.Nmeasures*sizeof(double),1,file.pFile);
      fclose(file.pFile);
    }
  else
    {
      printf("error opening wcoord file");
      exit(1);
    }
  
  fileName(datapath, in.weightsfile);
  if( (file.pFile = fopen (filename,"rb")) != NULL )
    {
      fseek (file.pFile,startrow*metaData.polarisations*sizeof(float),SEEK_SET);
      ret = fread(data.weights,(metaData.Nweights)*sizeof(float),1,file.pFile);
      fclose(file.pFile);
    }
  else
    {
      printf("error opening weights file");
      exit(1);
    }
  
  fileName(datapath, in.visrealfile);
  if( (file.pFile = fopen (filename,"rb")) != NULL )
    {
      fseek (file.pFile,startrow*metaData.freq_per_chan*metaData.polarisations*sizeof(float),SEEK_SET);
      ret = fread(data.visreal,metaData.Nvis*sizeof(float),1,file.pFile);
      fclose(file.pFile);
    }
  else
    {
      printf("error opening visibilities_real file");
      exit(1);
    }
  
  fileName(datapath, in.visimgfile);
  if( (file.pFile = fopen (filename,"rb")) != NULL )
    {
      fseek (file.pFile,startrow*metaData.freq_per_chan*metaData.polarisations*sizeof(float),SEEK_SET);
      ret = fread(data.visimg,metaData.Nvis*sizeof(float),1,file.pFile);
      fclose(file.pFile);
    }
  else
    {
      printf("error opening visibilities_img file");
      exit(1);
    }
  
  
  MPI_Barrier(MPI_COMM_WORLD);
  
}
