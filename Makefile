HEFFTE = /beegfs/glacopo/heffte_gen10/local

INC   =  -I$(HEFFTE)/include

LIB   =  -L$(HEFFTE)/lib64 -lheffte -lm

EXEC = rick

CC   = mpicc

OPT += -DGAUSS
OPT += -DUSE_MPI
OPT += -DPHASE_ON
OPT += -DSTOKESI
#OPT += WRITE_DATA

FLAGS   = -O3 -march=native -mavx -mavx512f 

# Source files #

OBJECTS = test_clib.o gridding_library.o fft_library.o phase_correction_library.o

DEPS = ricklib.h

%.o: %.c $(DEPS)
	$(CC) $(FLAGS) $(OPT) $(INC) $(LIB) -c -o $@ $< 

w-stacking: $(OBJECTS) $(DEPS) Makefile
	$(CC) $(FLAGS) $(OPT) $(INC) $(LIB) $(OBJECTS) -o $(EXEC)
.PHONY:
	clean		

clean:
	rm *.o rick
