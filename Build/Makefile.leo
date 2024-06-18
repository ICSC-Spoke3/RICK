CC       =  gcc
CXX      =  g++

MPICC    =  mpicc
MPIC++   =  mpic++

OPTIMIZE = -O4 -fopenmp -march=native -mavx -mavx2 
OPT_PURE_MPI = -O4 -march=native -mavx -mavx2

OMP_GPU = -mp=multicore,gpu -gpu=cuda11.8 -gpu=cc80

###CUDA_INC = -I/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.5-pdmwq3k5perrhdqyrv2hspv4poqrb2dr/Linux_x86_64/23.5/cuda/11.8/include
###CUDA_LIB = -L/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.5-pdmwq3k5perrhdqyrv2hspv4poqrb2dr/Linux_x86_64/23.5/cuda/11.8/lib64 -L/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.5-pdmwq3k5perrhdqyrv2hspv4poqrb2dr/Linux_x86_64/23.5/cuda/11.8/targets/x86_64-linux/lib/stubs


CUDA_INC = -I/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.11-tgvw3c2exrfgdvn5qdw3rybzd3dbkkim/Linux_x86_64/23.11/cuda/12.3/include
CUDA_LIB = -L/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.11-tgvw3c2exrfgdvn5qdw3rybzd3dbkkim/Linux_x86_64/23.11/cuda/12.3/lib -L/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.11-tgvw3c2exrfgdvn5qdw3rybzd3dbkkim/Linux_x86_64/23.11/cuda/12.3/targets/x86_64-linux/lib/stubs



FFTW_INCL=
FFTW_LIB=


##########################################################
#NVIDIA CUFFTMP

###CUFFTMP_LIB  = -L/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.5-pdmwq3k5perrhdqyrv2hspv4poqrb2dr/Linux_x86_64/23.5/math_libs/11.8/lib64
###CUFFTMP_INC = -I/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.5-pdmwq3k5perrhdqyrv2hspv4poqrb2dr/Linux_x86_64/23.5/math_libs/11.8/include/cufftmp

CUFFTMP_INC = -I/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.11-tgvw3c2exrfgdvn5qdw3rybzd3dbkkim/Linux_x86_64/23.11/math_libs/12.3/include/cufftmp
CUFFTMP_LIB = -L/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.11-tgvw3c2exrfgdvn5qdw3rybzd3dbkkim/Linux_x86_64/23.11/math_libs/12.3/lib64


##########################################################

###NVSHMEM_INC = -I/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.5-pdmwq3k5perrhdqyrv2hspv4poqrb2dr/Linux_x86_64/23.5/comm_libs/11.8/nvshmem_cufftmp_compat/include/
###NVSHMEM_LIB = -L/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.5-pdmwq3k5perrhdqyrv2hspv4poqrb2dr/Linux_x86_64/23.5/comm_libs/11.8/nvshmem_cufftmp_compat/lib/

NVSHMEM_INC = -I/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.11-tgvw3c2exrfgdvn5qdw3rybzd3dbkkim/Linux_x86_64/23.11/comm_libs/12.3/nvshmem/include
NVSHMEM_LIB = -L/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.11-tgvw3c2exrfgdvn5qdw3rybzd3dbkkim/Linux_x86_64/23.11/comm_libs/12.3/nvshmem/lib


##########################################################
#NVIDIA NCCL REDUCE

###NCCL_INC = -I/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.5-pdmwq3k5perrhdqyrv2hspv4poqrb2dr/Linux_x86_64/23.5/comm_libs/11.8/nccl/include
###NCCL_LIB = -L/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.5-pdmwq3k5perrhdqyrv2hspv4poqrb2dr/Linux_x86_64/23.5/comm_libs/11.8/nccl/lib


NCCL_INC = -I/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.11-tgvw3c2exrfgdvn5qdw3rybzd3dbkkim/Linux_x86_64/23.11/comm_libs/12.3/nccl/include
NCCL_LIB = -L/leonardo/prod/spack/03/install/0.19/linux-rhel8-icelake/gcc-11.3.0/nvhpc-23.11-tgvw3c2exrfgdvn5qdw3rybzd3dbkkim/Linux_x86_64/23.11/comm_libs/12.3/nccl/lib


##########################################################

NVC = nvc 
NVC++ = nvc++
NVFLAGS = -O4 -fast -march=native $(OMP_GPU) -mavx -mavx2  
NVLIB = $(CUDA_INC) $(CUDA_LIB) -lcuda -lcudart


NVLIB_2 = $(CUDA_INC) $(CUDA_LIB) $(MPI_INC) $(MPI_LIB) $(CUFFTMP_INC) $(CUFFTMP_LIB) $(NVSHMEM_INC) $(NVSHMEM_LIB) -lcuda -lnvidia-ml -lnvshmem_host -lnvshmem_device -lcudart -lcufftMp 

NVLIB_3 = $(CUDA_INC) $(CUDA_LIB) $(MPI_INC) $(MPI_LIB) $(NCCL_INC) $(NCCL_LIB) -lcudart -lnccl


NVCC = nvcc
OPT_NVCC   = -std=c++17 --generate-code arch=compute_80,code=sm_80 

CFLAGS +=

MPICHLIB =



##########################################################
#AMD GPUs (DEFAULT = LUMI)

CLANG   = clang
CLANG++ = clang++
 
OPTIMIZE_AMD = -O3 -Ofast -fopenmp -march=native -mavx -mavx2 -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx90a

RCCL_INCL= -I/opt/rocm-5.2.3/rccl/include
RCCL_LIB= -L/opt/rocm-5.2.3/rccl/lib

HIP_INCL= -I/opt/rocm-5.2.3/hip/include
HIP_LIB= -L/opt/rocm-5.2.3/hip/lib

AMDLIB = $(HIP_INCL) $(HIP_LIB) $(RCCL_INCL) $(RCCL_LIB) -lamdhip64 -lrccl
###########################################################

