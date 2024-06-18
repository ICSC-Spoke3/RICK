

#define NUMA    0     // my NUMA node communicator, includes all the sibling tasks that share memory
#define ISLAND  1     // something between the host and the NUMA nodes, if present
#define myHOST  2     // my host communicator, includes all the sibling tasks running on the same hosts
#define HOSTS   3     // the communicator that includes only the masters of the hosts
#define WORLD   4     // everybody is in (i.e. this is MPI_COMM_WORLD)
#define HLEVELS 5

extern char *LEVEL_NAMES[HLEVELS];

typedef struct
{
  MPI_Win   win;
  MPI_Aint  size;   
  void     *ptr __attribute__((aligned(128)));
  int       disp;
} win_t;

typedef struct
{
  int  mycpu;                          // the core (hwthread) on which i'm running
  int  mysocket;                       // the socket on which i'm running
  int  nthreads;                       // how many (omp) thread do i have
  int  myhost;                         // the host on which i'm running
  int  Nhosts;
  int  Ntasks[HLEVELS];  
  int *Ranks_to_host;                  // keeps track to what host the original global_ranks belong to
  int *Ranks_to_myhost;                // keeps track of the local_rank of the original global_rank
  int  Rank[HLEVELS];
  int  MAXl;                           // the maximum level of the hierarchy
  int  SHMEMl;                         // the maximum hierarchy level that is in shared memory
  MPI_Comm *COMM[HLEVELS];
  // -----------------------
  // not yet used
  // int  mynode;                      // the numa node on which i'm running
  // int  ntasks_in_my_node;
  win_t  win_ctrl;                     // my shared memory window used for flow control
  win_t  win;                          // my shared-memory window used for temporary data
  win_t  fwin;                         // my shared-memory window used for final data
  win_t *scwins;                       // the control shared-memory windows of the tasks in my host
  win_t *swins;                        // the temporary data shared-memory windows of the tasks in my host
  win_t *sfwins;                       // the final data shared-memory windows of the tasks in my host
} map_t;



#define NSLEEP( T ) {struct timespec tsleep={0, (T)}; nanosleep(&tsleep, NULL); }


#define ACQUIRE_CTRL( ADDR, V, TIME, CMP ) {                            \
    double tstart = CPU_TIME_tr;                                        \
    atomic_thread_fence(memory_order_acquire);                          \
    int read = atomic_load((ADDR));                                     \
    while( read CMP (V) ) { NSLEEP(50);                                 \
      read = atomic_load((ADDR)); }                                     \
    (TIME) += CPU_TIME_tr - tstart; }

              
 #define ACQUIRE_CTRL_DBG( ADDR, V, CMP, TAG ) {                  \
    atomic_thread_fence(memory_order_acquire);                          \
    int read = atomic_load((ADDR));                                     \
    printf("%s %s Task %d read is %d\n", TAG, #ADDR, rank, read);      \
    unsigned int counter = 0;                                           \
    while( read CMP (V) ) { NSLEEP(50);                                 \
    read = atomic_load((ADDR));                                         \
    if( (++counter) % 10000 == 0 )                                      \
                        printf("%s %p Task %d read %u is %d\n",         \
                        TAG, ADDR, rank, counter, read);}              \
    }



#define DATA_FREE   -1
#define FINAL_FREE  -1

#define CTRL_DATA          0
#define CTRL_FINAL_STATUS  0
#define CTRL_FINAL_CONTRIB 1
#define CTRL_SHMEM_STATUS  2
#define CTRL_BLOCKS        3
#define CTRL_END           (CTRL_BLOCKS+1)

#define CTRL_BARRIER_START 0
#define CTRL_BARRIER_END   1

#define BUNCH_FOR_VECTORIZATION 128




extern map_t       Me;
extern MPI_Comm    COMM[HLEVELS];


extern MPI_Aint    win_ctrl_hostmaster_size;
extern MPI_Win     win_ctrl_hostmaster;
extern int         win_ctrl_hostmaster_disp;
extern void       *win_ctrl_hostmaster_ptr;  

extern MPI_Aint    win_hostmaster_size;
extern MPI_Win     win_hostmaster;
extern int         win_hostmaster_disp;
extern void       *win_hostmaster_ptr;  

extern win_t *win_ctrl;


typedef long long unsigned int int_t;
typedef struct { int Nblocks; int_t *Bstart; int_t *Bsize; } blocks_t;
extern MPI_Request *requests;
extern int thid;
extern int Ntasks_local;
extern blocks_t blocks;
extern double **swins;
extern int    **cwins;
extern int max_level;
extern double *end_4, *end_reduce;
extern int dsize_4, iter; 

int numa_init( int, int, MPI_Comm *, map_t *);
int numa_allocate_shared_windows( map_t *, MPI_Aint, MPI_Aint );
int numa_shutdown( int, int, MPI_Comm *, map_t *);
void numa_expose( map_t *, int );
int check_ctrl_win( win_t *, win_t *, int);
