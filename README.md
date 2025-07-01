# Radio Imaging Code Kernels

Radio Imaging Code Kernels (RICK) is a software that is able to exploit parallelism and accelerators for radio astronomy imaging. **This software is currently under development**.<br>

RICK is written in C/C++ and can perform the following routines:
- gridding
- Fast Fourier Transform (FFT)
- w-correction

It exploits the Message Passing Interface (MPI) and OpenMP for parallelism, and is able to run on both NVIDIA and AMD GPUs using OpenMP for GPU offloading.

## How to use RICK library

RICK library is also made for creating shared libraries to be called within any radio interferometric imaging software. <br>
RICKlib requires the following softwares:
- MPI
- HeFFTe (please install it with the correct support (FFTW3, CUFFT, ROCFFT))

On Setonix: do NOT compile the code with Cray clang, please use amdclang available under rocm/6.3.2 module.

Please install HeFFTe on your machine

### IN C:

```
> make
```

To run the code (SETONIX)
```
> srun -l -u -N --ntasks-per-node= -c 8 --gpus-per-task=1 ./rick_gpu
```

To run the code (LEONARDO) (Temporarily, we're gonna switch to srun again)
```
> mpirun -n ./rick_gpu
```

Without the -DAMD or -DNVIDIA options active, the code is going to be compiled for the host

<br>

## Contacts

For feedbacks, suggestions, and requests [contact us](mailto:emanuele.derubeis2@unibo.it, giovanni.lacopo@inaf.it)!: emanuele.derubeis2@unibo.it, giovanni.lacopo@inaf.it
