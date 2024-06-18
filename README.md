# Radio Imaging Code Kernels

Radio Imaging Code Kernels (RICK) is a software that is able to exploit parallelism and accelerators for radio astronomy imaging. **This software is currently under development**.<br>

RICK is written in C/C++ and can perform the following routines:
- gridding
- Fast Fourier Transform (FFT)
- w-correction

It exploits the Message Passing Interface (MPI) and OpenMP for parallelism, and is able to run on both NVIDIA and AMD GPUs using CUDA, HIP, and OpenMP for GPU offloading.

## Why and where to use RICK?

Given that it is currently under development, RICK can be used to test its performances for your dataset. It has been tested for several radio interferometers:
- Low Frequency Array (LOFAR)
- Jansky Very Large Array (JVLA)
- Atacama Large Millimeter Array (ALMA)

If you tested RICK on an interferometer not present on this list, please contact us with a brief comment or report on your experience.

<br>

## What can RICK *not* do?

Currently in RICK we have not yet implemented the following steps:

- there is not the possibility to separate the Stokes parameters
- the flux scale is not physical
- weighting

All these points will be added in future versions of RICK.

<br>

## Preparing the dataset

To be "digested" by RICK, the input Measurement Set must be written in binary. To do this, a Python script named `create_binMS.py` can be found in the `/scripts` directory. The supported format for MS input is the Measurement Set V2.0 standard. Given that the script needs to read some columns of the input data, we reccommend to use a Singularity image to deal with the `casacore` dependencies. You can find one [here](https://lofar-webdav.grid.sara.nl/software/shub_mirror/tikk3r/lofar-grid-hpccloud/lofar_sksp_v3.5_x86-64_generic_noavx512_ddf.sif?action=show). The script has to be modified with the input Measurement Set.

## How to compile the code

The latest version of the code is in the *merge* branch (soon this will be deprecated, and the up-to-date code will always be in the main branch). To compile it, you need first to specify in the Makefile the required configuration for the RICK execution.<br>
Here there is a brief explaination of each macro contained in the Makefile.
- **Parallel FFT implementation**
	- USE_FFTW: enables the usage of the FFTW library with pure MPI.
	- HYBRID_FFTW: enables the usage of the hybrid FFT implementation with MPI+OpenMP using the FFTW library.
	- USE_OMP: to switch on OpenMP parallelization.
- **Activate pieces of the code to be performed**
	- PHASE_ON: performs w-stacking phase correction.
	- NORMALIZE_UVW: normalize $u$, $v$, and $w$ in case it is not done in the binMS.
- **Select the gridding kernel**
	- GAUSS: select a Gaussian gridding kernel.
	- GAUSS_HI_PRECISION: select an high-precision Gaussian gridding kernel.
	- KAISER_BESSEL: select a Kaiser-Bessel gridding kernel.
- **GPU offloading options**
	- NVIDIA: NVIDIA macro, required for NVIDIA calls (*currently not required*).
	- CUDACC: enables CUDA for GPU offloading.
	- ACCOMP: enables OpenMP for GPU offloading.
	- GPU_STACKING: enables the w-stacking step on GPUs. This can be required because, at the moment, RICK cannot do on GPUs both the reduce and the w-stacking (*currently not active*).
	- NCCL_REDUCE: use the NCCL implementation of the reduce on NVIDIA GPUs.
	- RCCL_REDUCE: use the accelerated implementation of the reduce on AMD GPU.
	- CUFFTMP: enables the distributed FFT on GPUs.
	- FULL_NVIDIA: enables the full-GPU version of the code. This feature is recommended to run the code full on NVIDIA GPUs, automatically activates the required macros.
- **Output handling**
	- FITSIO: enables the usage of CFITSIO library to write outputs in FITS format.
	- PARALLELIO: enables the parallelization for final images writing (will be deprecated).
	- WRITE_DATA: write the full 3D cube of gridded visibilities and its FFT transform (impacts the performance, use only for debugging or tests).
	- WRITE_IMAGE: write the final images as output.
- **Developing options** (can heavily impact the performances, so use them carefully!)
	- DEBUG.
	- VERBOSE: enable verbose.

**Be careful because there are some macros that are obviously not compatible with others. More detailed instructions will be given to choose the best combination of macros for your usage or architecture.**<br>

Once selected the desired macros, you need to select the architecture on which you want to execute the code. In the directory `Build/` there are several configuration files that adapt to your machine. To select your architecture you need to define the following environment variable
```
> export $SYSTYPE=architecture_name
```

The latest version of RICK has been tested on the following architectures:
- Marconi 100 (CINECA) (`M100`)
- Leonardo (CINECA) (`leo`)
- MacOSX (`Macosx`)

If your machine is not on the list, you can define `SYSTYPE=local` or modify it according to your needs and create one ad-hoc for your configuration. <br>

When you use GPU offloading with OpenMP, please **do not compile the CPU part with NVC**.
This can be easily fixed by setting the environment variable:
```
> export OMPI_CC=gcc
```

In the case in which the default compiler is NVC. The Makefile is suited to understand which are the parts to be compiled with NVC for the OpenMP offloading. The final linker in this case will be however the NVC/NVC++.

The problem does not raise on AMD platforms, because you use clang/clang++ for both CPUs and GPUs.

To use the **cuFFTMp** with **nvhpc 23.5** you need to add the following paths to the environment variable `LD_LIBRARY_PATH`:
```
> export LD_LIBRARY_PATH="$NVHPC_HOME/Linux_x86_64/23.5/comm_libs/11.8/nvshmem_cufftmp_compat/lib/:$LD_LIBRARY_PATH"

> export LD_LIBRARY_PATH="$NVHPC_HOME/Linux_x86_64/23.5/math_libs/11.8/lib64/:$LD_LIBRARY_PATH"
```

To use the NCCL reduce on NVIDIA GPUs you need to add the following paths to the environment variable `LD_LIBRARY_PATH`:
```
export LD_LIBRARY_PATH="$NVHPC_HOME/Linux_x86_64/23.5/comm_libs/11.8/nccl/lib/:$LD_LIBRARY_PATH"
```

All these libraries need to be linked if you want to run the code fully in GPUs. We recommend to enable the sole FULL_NVIDIA macro to do that.

At this point you can compile the code with the command:
```
> make w-stacking
```

An executable will be created whose name depends on the different implementations that have been required for the compilation. The extensions of the executable will be changed depending on the different acceleration options.

## How to execute the code

To run the code, the `data/paramfile.txt` is available. Feel free to change the parameters according to your configuration and your desired output. Here we briefly list some of them:
- Datapath: path of the input Measurement Set. Multiple MS can be specified inserting multiple paths.
- num_threads: number of OpenMP threads.
- reduce_method: select the reduce to be used for the execution. 0 corresponds to the MPI Reduce, 1 corresponds to the Reduce Ring.
- grid_size_x, grid_size_y: size of the output image.
- num_w_planes: number of w-planes for the w-stacking.
- fftfile2, fftfile3: name of the output `.bin` real and imaginary files.
- fftfile_writedata1, fftfile_writedata2: name of the FFT-ed real and imaginary cubes, produced only with the relative macro enabled. 
- logfile: name of the log file.

In the case in which the code has been compiled without either `-fopenmp` or `-D_OPENMP` options, the code is forced to use the standard MPI_Reduce implementation, since our reduce works only with OpenMP.

When CPU hybrid MPI+OpenMP parallelism is requested, you have to select the number of threads by setting **--cpus-per-task=** in your bash script, then add the following lines:
```
> export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
> export OMP_PLACES=cores
> export OMP_PROC_BIND=close
```

Then, to run the code in order to fully fill all the available cores in the node please add **--ntasks-per-socket=** in your bash script and then run:
```
> mpirun -np [n] --bind-to core --map-by ppr:${SLURM_NTASKS_PER_SOCKET}:socket:pe=${SLURM_CPUS_PER_TASK} -x OMP_NUM_THREADS [executable] data/paramfile.txt
```

Once you have compiled the code, run it simply with the command:
```
> mpirun -np [n] [executable] data/paramfile.txt
```

## Contacts

For feedbacks, suggestions, and requests [contact us](mailto:emanuele.derubeis2@unibo.it)!: emanuele.derubeis2@unibo.it
