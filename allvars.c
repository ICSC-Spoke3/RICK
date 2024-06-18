

#include "allvars.h"


struct io file;

struct ip in;

struct op out, outparam;

struct meta      metaData;
struct parameter param;
struct fileData  data;

char   filename[LONGNAME_LEN], buf[NAME_LEN], num_buf[NAME_LEN];
char   datapath[LONGNAME_LEN];
int    xaxis, yaxis;
int    rank;
int    size;
myuint   nsectors;
myuint   startrow;
double resolution, dx, dw, w_supporth;

myuint **sectorarray   = NULL;
myuint  *histo_send    = NULL;
int    verbose_level = 0; 

timing_t timing_wt;
double   reduce_mpi_time;
double   reduce_shmem_time;


myuint     size_of_grid;
double   *grid_pointers = NULL, *grid, *gridss, *gridss_real, *gridss_img, *gridss_w, *grid_gpu, *gridss_gpu;

MPI_Comm      MYMPI_COMM_WORLD; 
MPI_Win       slabwin;
