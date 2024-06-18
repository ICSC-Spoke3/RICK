# comment/uncomment the various options depending hoe you want to build the program
# Set default values for compiler options if no systype options are given or found

EXEC = w-stacking
EXEC_EXT :=

MPICC     = mpicc
MPICXX    = mpiCC
OPTIMIZE  = -fopenmp -O3 -march=native 
MPICHLIB  = 
SWITCHES =

ifdef SYSTYPE
SYSTYPE := $(SYSTYPE)
include Build/Makefile.$(SYSTYPE)
else
include Build/Makefile.systype
include Build/Makefile.$(SYSTYPE)
endif

LINKER=$(MPICC)

FFTW_MPI_INC = 
FFTW_MPI_LIB = 

CFLAGS += -I./

FFTWLIBS =


# ========================================================
# CODE OPTIONS FOR THE COMPILATION
# (refer to the RICK wiki here
# https://www.ict.inaf.it/gitlab/claudio.gheller/hpc_imaging/-/wikis/home)
# ========================================================


# PARALLEL FFT IMPLEMENTATION


# use FFTW (it can be switched on ONLY if MPI is active)
OPT += -DUSE_FFTW

# use omp-ized version of fftw routines
#OPT += -DHYBRID_FFTW

# switch on the OpenMP parallelization
#OPT += -DUSE_OMP

# ========================================================

# ACTIVATE PIECES OF THE CODE TO BE PERFORMED


# perform w-stacking phase correction
OPT += -DPHASE_ON

# Normalize uvw in case it is not done in the binMS
#OPT += -DNORMALIZE_UVW

# ========================================================

# SELECT THE GRIDDING KERNEL: GAUSS, GAUSS_HI_PRECISION, KAISERBESSEL


#OPT += -DGAUSS_HI_PRECISION

#OPT += -DGAUSS

OPT += -DKAISERBESSEL

# ========================================================

# GPU OFFLOADING OPTIONS


#OPT += -DNVIDIA

# use CUDA for GPUs
#OPT += -DCUDACC

# use GPU acceleration via OMP 
#OPT += -DACCOMP

# perform stacking on GPUs
#OPT += -DGPU_STACKING

# use NVIDIA GPU to perform the reduce
#OPT += -DNCCL_REDUCE

# use GPU to perform FFT
#OPT += -DCUFFTMP

# FULL NVIDIA GPU SUPPORT - Recommended for full NVIDIA GPU code execution
#OPT += -DFULL_NVIDIA
ifeq (FULL_NVIDIA,$(findstring FULL_NVIDIA,$(OPT)))
OPT += -DCUDACC -DNCCL_REDUCE -DCUFFTMP
endif



# use HIP for GPUs
#OPT += -DHIPCC

# support for AMD GPUs
#OPT += -D__HIP_PLATFORM_AMD__

# use AMD GPU to perform the reduce
#OPT += -DRCCL_REDUCE

# FULL AMD GPU SUPPORT - Recommended for full AMD GPU code execution
#OPT += -DFULL_AMD
ifeq (FULL_AMD,$(findstring FULL_AMD,$(OPT)))
OPT += -DHIPCC -DRCCL_REDUCE -D__HIP_PLATFORM_AMD__
endif

# =======================================================

# OUTPUT HANDLING


# Support CFITSIO !!! Remember to add the path to the CFITSIO library to LD_LIBRARY_PATH
#OPT += -DFITSIO

# write the full 3D cube of gridded visibilities and its FFT transform
#OPT += -DWRITE_DATA

# write the final image
OPT += -DWRITE_IMAGE

# =======================================================

# DEVELOPING OPTIONS


#OPT += -DVERBOSE

#perform the debugging in the ring implementation
#OPT += -DDEBUG

# =======================================================
# END OF OPTIONS
# =======================================================

ifeq (USE_OMP,$(findstring USE_OMP,$(OPT)))
FLAGS=$(OPTIMIZE)
else
FLAGS=$(OPT_PURE_MPI)
endif

ifeq (FITSIO,$(findstring FITSIO,$(OPT)))
        LIBS += -L$(FITSIO_LIB) -lcfitsio
endif	

DEPS = w-stacking.h  main.c allvars.h


# -------------------------------------------------------
#
#  here we define which OBJ files have to be compiled by who;
#  in fact, depending on the GPU-acceleration being on or off,
#  and on having AMD/NVidia GPUs, things may be different
#
# ------------------------------------------------------


# ----- define which files will be compiled by MPICC
#
# these are the OBJS that will be compiled by C compiler if no acceleration (neither with CUDA nor with OpenMP) is provided
CC_OBJ_NOACC = allvars.o main.o init.o gridding.o gridding_cpu.o fourier_transform.o result.o numa.o reduce.o w-stacking.o phase_correction.o

# these are the OBJs that will be compiled by the normal MPICC compiler if GPU acceleration is switched on
CC_OBJ_ACC = allvars.o main.o init.o gridding.o gridding_cpu.o fourier_transform.o result.o numa.o reduce.o


# ----- define which files will be compiled by NVCC for Nvidia
#
DEPS_ACC_CUDA = w-stacking.h w-stacking.cu phase_correction.cu
OBJ_ACC_CUDA = phase_correction.o w-stacking.o

# ----- define which files will be compiled by HIPCC for AMD
#
DEPS_ACC_HIP = w-stacking.hip.hpp w-stacking.hip.cpp phase_correction.hip.cpp
OBJ_ACC_HIP = phase_correction.hip.o w-stacking.hip.o

# ----- define which files will be compiled by NVC with OMP offloading for wither Nvidia or AMD
#
DEPS_ACC_OMP = w-stacking.h phase_correction.c w-stacking.c
OBJ_ACC_OMP = phase_correction.o w-stacking.o



# ----- define what files will be compiled by NVC with OMP offloading when the stacking reduce is
#       offloaded on GPU
ifeq (CUDACC,$(findstring CUDACC,$(OPT)))
DEPS_NCCL_REDUCE = gridding_nccl.cu
OBJ_NCCL_REDUCE  = gridding_nccl.o

else ifeq (HIPCC,$(findstring HIPCC,$(OPT)))
DEPS_RCCL_REDUCE = gridding_rccl.hip.cpp allvars_rccl.hip.hpp
OBJ_RCCL_REDUCE  = gridding_rccl.hip.o
else
DEPS_NCCL_REDUCE = gridding_nccl.cpp
OBJ_NCCL_REDUCE  = gridding_nccl.o
endif

# ----- define what files will be compiled by NVCC for Nvidia cufftMP implementation of FFT
#
ifeq (CUDACC,$(findstring CUDACC,$(OPT)))
DEPS_ACC_CUFFTMP = cuda_fft.cu 
OBJ_ACC_CUFFTMP  = cuda_fft.o
else
DEPS_ACC_CUFFTMP = cuda_fft.cpp 
OBJ_ACC_CUFFTMP  = cuda_fft.o
endif


# -----------------------------------------------------
#
# end of OBJ definition
# ----------------------------------------------------


ifeq (ACCOMP,$(findstring ACCOMP, $(OPT)))
OBJ = $(CC_OBJ_ACC)
else ifeq (CUDACC,$(findstring CUDACC, $(OPT)))
OBJ = $(CC_OBJ_ACC)
else ifeq (HIPCC,$(findstring HIPCC, $(OPT)))
OBJ = $(CC_OBJ_ACC)
else 
OBJ = $(CC_OBJ_NOACC)
endif

ifeq (USE_FFTW,$(findstring USE_FFTW,$(OPT)))
CFLAGS += $(FFTW_MPI_INC)
ifeq (HYBRID_FFTW,$(findstring HYBRID_FFTW,$(OPT)))
FFTWLIBS = $(FFTW_MPI_LIB) -lfftw3_omp -lfftw3_mpi -lfftw3 -lm
else
FFTWLIBS = $(FFTW_MPI_LIB) -lfftw3_mpi -lfftw3 -lm
endif
endif

# define rules for sources that contains GPU code
#

ifneq (CUDACC,$(findstring CUDACC,$(OPT)))
w-stacking.c: w-stacking.cu
	cp w-stacking.cu w-stacking.c

phase_correction.c: phase_correction.cu
	cp phase_correction.cu phase_correction.c

cuda_fft.cpp: cuda_fft.cu
	cp cuda_fft.cu cuda_fft.cpp

gridding_nccl.cpp: gridding_nccl.cu
	cp gridding_nccl.cu gridding_nccl.cpp

gridding_rccl.cpp: gridding_rccl.cu
	cp gridding_rccl.cu gridding_rccl.cpp

else ifneq (HIPCC,$(findstring HIPCC,$(OPT)))
w-stacking.c: w-stacking.cu
	cp w-stacking.cu w-stacking.c

phase_correction.c: phase_correction.cu
	cp phase_correction.cu phase_correction.c

cuda_fft.cpp: cuda_fft.cu
	cp cuda_fft.cu cuda_fft.cpp

gridding_nccl.cpp: gridding_nccl.cu
	cp gridding_nccl.cu gridding_nccl.cpp

gridding_rccl.cpp: gridding_rccl.cu
	cp gridding_rccl.cu gridding_rccl.cpp
else
w-stacking.c: w-stacking.cu
	rm -f w-stacking.c
	touch w-stacking.c
phase_correction.c: phase_correction.cu
	rm -f phase_correction.c
	touch phase_correction.c
cuda_fft.cpp: cuda_fft.cu
	rm -f cuda_fft.cpp
	touch cuda_fft.cpp
gridding_nccl.cpp: gridding_nccl.cu
	rm -f gridding_nccl.cpp
	touch gridding_nccl.cpp
gridding_rccl.cpp: gridding_rccl.cu
	rm -f gridding_rccl.cpp
	touch gridding_rccl.cpp
endif


#####################################################################################

ifeq (USE_FFTW,$(findstring USE_FFTW,$(OPT)))
EXEC_EXT := $(EXEC_EXT)_fftw
endif

ifeq (CUDACC,$(findstring CUDACC,$(OPT)))
EXEC_EXT := $(EXEC_EXT)_acc-cuda
LINKER=$(MPIC++)
FLAGS=$(OPTIMIZE) 
LIBS=$(NVLIB)
$(OBJ_ACC_CUDA): $(DEPS_ACC_CUDA)
	$(NVCC) $(OPT) $(OPT_NVCC) $(CFLAGS) -c w-stacking.cu phase_correction.cu $(LIBS)
OBJ += $(OBJ_ACC_CUDA)
endif

ifeq (HIPCC,$(findstring HIPCC,$(OPT)))
EXEC_EXT := $(EXEC_EXT)_acc-hip
LINKER=$(MPIC++)
FLAGS=$(OPTIMIZE) 
LIBS=$(AMDLIB)
$(OBJ_ACC_HIP): $(DEPS_ACC_HIP)
	$(HIPCC) $(OPT) $(OPT_HIPCC) $(CFLAGS) -c w-stacking.hip.cpp phase_correction.hip.cpp $(LIBS)
OBJ += $(OBJ_ACC_HIP)
endif

ifeq (ACCOMP,$(findstring ACCOMP,$(OPT)))

# >>>>> AMD GPUs
ifeq (__HIP_PLATFORM_AMD__,$(findstring __HIP_PLATFORM_AMD__,$(OPT)))

EXEC_EXT := $(EXEC_EXT)_acc-omp
LINKER=$(MPICC)
FLAGS=$(OPTIMIZE_AMD) $(CFLAGS)
LIBS=$(AMDLIB) 
$(OBJ_ACC_OMP): $(DEPS_ACC_OMP)
	$(MPICC) $(FLAGS) $(OPT) -c $^ $(CFLAGS) 
OBJ += $(OBJ_ACC_OMP)

# >>>> NVIDIA GPUs
else

EXEC_EXT := $(EXEC_EXT)_acc-omp
LINKER=$(NVC)
FLAGS=$(NVFLAGS) $(CFLAGS)
LIBS=$(NVLIB)
$(OBJ_ACC_OMP): $(DEPS_ACC_OMP)
	$(NVC) $(FLAGS) $(OPT) -c $^ $(LIBS)
OBJ += $(OBJ_ACC_OMP)

endif

endif


ifeq (NCCL_REDUCE,$(findstring NCCL_REDUCE,$(OPT)))

ifeq (CUDACC,$(findstring CUDACC,$(OPT)))
EXEC_EXT := $(EXEC_EXT)_acc-reduce
LINKER=$(MPIC++)
FLAGS=$(OPTIMIZE)
LIBS=$(NVLIB_3)
$(OBJ_NCCL_REDUCE): $(DEPS_NCCL_REDUCE)
	$(NVCC) $(OPT_NVCC) $(OPT) -c $^ $(LIBS)
OBJ += $(OBJ_NCCL_REDUCE)

else ifeq (HIPCC,$(findstring HIPCC,$(OPT)))
EXEC_EXT := $(EXEC_EXT)_acc-reduce
LINKER=$(MPIC++)
FLAGS=$(OPTIMIZE) $(CFLAGS)
LIBS=$(AMDLIB_3) 
$(OBJ_NCCL_REDUCE): $(DEPS_NCCL_REDUCE)
	$(MPIC++) $(FLAGS) $(OPT) -c $^ $(CFLAGS) $(LIBS)
OBJ += $(OBJ_NCCL_REDUCE)

else

EXEC_EXT := $(EXEC_EXT)_acc-reduce
LINKER=$(NVC++)
FLAGS=$(NVFLAGS) $(CFLAGS)
LIBS=$(NVLIB) $(NVLIB_3)
$(OBJ_NCCL_REDUCE): $(DEPS_NCCL_REDUCE)
	$(NVC++) $(FLAGS) $(OPT) -c $^ $(LIBS)
OBJ += $(OBJ_NCCL_REDUCE)
endif
endif

ifeq (RCCL_REDUCE,$(findstring RCCL_REDUCE,$(OPT)))
EXEC_EXT := $(EXEC_EXT)_acc-reduce
LINKER=$(MPIC++)
FLAGS=$(OPTIMIZE_AMD) $(CFLAGS)
LIBS=$(AMDLIB_3) 
$(OBJ_RCCL_REDUCE): $(DEPS_RCCL_REDUCE)
	$(HIPCC) $(FLAGS) $(OPT) -c $^ $(CFLAGS) $(LIBS)
OBJ += $(OBJ_RCCL_REDUCE)
endif

ifeq (CUFFTMP,$(findstring CUFFTMP,$(OPT)))

ifeq (CUDACC,$(findstring CUDACC,$(OPT)))
EXEC_EXT := $(EXEC_EXT)_acc-fft
LINKER=$(MPIC++)
FLAGS=$(OPTIMIZE)
ifeq (NCCL_REDUCE,$(findstring NCCL_REDUCE,$(OPT)))
LIBS=$(NVLIB_2) $(NVLIB_3)
else
LIBS=$(NVLIB_2)
endif
$(OBJ_ACC_CUFFTMP): $(DEPS_ACC_CUFFTMP)
	$(NVCC) $(OPT_NVCC) $(OPT) -c $^ $(LIBS)
OBJ += $(OBJ_ACC_CUFFTMP)

else

EXEC_EXT := $(EXEC_EXT)_acc-fft
LINKER=$(NVC++)
FLAGS=$(NVFLAGS) $(CFLAGS)
LIBS=$(NVLIB_2)
$(OBJ_ACC_CUFFTMP): $(DEPS_ACC_CUFFTMP)
	$(NVC++) $(FLAGS) $(OPT) -c $^ $(LIBS)
OBJ += $(OBJ_ACC_CUFFTMP)
endif

endif

###################################################################################

w-stacking: $(OBJ) $(DEPS) Makefile
	$(LINKER) $(FLAGS) $(OPT) $(OBJ) -o $(EXEC)$(EXEC_EXT) -lmpi $(FFTWLIBS) $(LIBS)

#$(OBJ): $(DEPS) Makefile

ifeq (USE_OMP,$(findstring USE_OMP,$(OPT)))
%.o: %.c $(DEPS)
	$(MPICC) $(OPTIMIZE) $(OPT) -c -o $@ $< $(CFLAGS)
else
%.o: %.c $(DEPS)
	$(MPICC) $(OPT_PURE_MPI) $(OPT) -c -o $@ $< $(CFLAGS)
endif

clean:
	rm -f *.o
	rm -f w-stacking.c
	rm -f phase_correction.c
	rm -f cuda_fft.cpp
	rm -f gridding_nccl.cpp
	rm -f gridding_rccl.cpp

#cleanall:
#	rm -f $(EXEC)$(EXT)
#	rm -f *.o
#	rm -f w-stacking.c
#	rm -f phase_correction.c
