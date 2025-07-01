HEFFTE = /leonardo_scratch/fast/CNHPC_1498509/heffte_nvc/local

INC   =  -I$(HEFFTE)/include

LIB   =  -L$(HEFFTE)/lib64 -lheffte -lm

EXEC = rick_gpu

CC   = mpicc

OPT += -DGAUSS
OPT += -DUSE_MPI
OPT += -DPHASE_ON
OPT += -DSTOKESI
OPT += -DNVIDIA
#OPT += -DWRITE_DATA
#OPT += -DDEBUG

FLAGS   = -O3 -g -mp=gpu -target=gpu -gpu=cc80 -Minfo=mp -march=native -mavx -mavx2 
#FLAGS   = -O3 -fopenmp --offload-arch=native -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=/data/Linux_x86_64/25.1/cuda/12.6 -g

# Source files #

OBJECTS = test_clib.o gridding_library.o fft_library.o phase_correction_library.o mpi_functions_redefined.o

DEPS = ricklib.h

%.o: %.c $(DEPS)
	$(CC) $(FLAGS) $(OPT) $(INC) $(LIB) -c -o $@ $< 

w-stacking: $(OBJECTS) $(DEPS) Makefile
	$(CC) $(FLAGS) $(OPT) $(INC) $(OBJECTS) -o $(EXEC) $(LIB) 
.PHONY:
	clean		

clean:
	rm *.o $(EXEC)
