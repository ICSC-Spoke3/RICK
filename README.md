# Radio Imaging Code Kernels

Radio Imaging Code Kernels (RICK) is a software that is able to exploit parallelism and accelerators for radio astronomy imaging. **This software is currently under development**.<br>

RICK is written in C/C++ and can perform the following routines:
- gridding
- Fast Fourier Transform (FFT)
- w-correction

It exploits the Message Passing Interface (MPI) and OpenMP for parallelism, and is able to run on both NVIDIA and AMD GPUs using CUDA, HIP, and OpenMP for GPU offloading.

## How to use RICK library

RICK library is made for creating shared libraries to be called within any radio interferometric imaging software or, more simply, within a Python script. <br>
RICKlib requires the following softwares:
- MPI
- fftw3

### In Python:
To compile the individual libraries use the following commands:
```
> mpicc -L/path/to/mpi/lib/ -I/path/to/mpi/include/ -shared -o gridding.so -fPIC gridding_library.c
> mpicc -L/path/to/mpi/lib/ -L/path/to/fftw3/lib/  -I/path/to/mpi/include/ -I/path/to/fftw3/include/ -shared -o fft.so -fPIC fft_library.c
> mpicc -L/path/to/mpi/lib/ -I/path/to/mpi/include/ -shared -o phasecorr.so -fPIC phase_correction_library.c
```
These commands should create dynamic libraries with extension `.so` in your working directory.<br>

After this, you can run the Python script `lib_test.py` specifying the directory of the measurement set and the parameters for the image (size, w-planes, etc.)
```
> python3 lib_test.py
```

### IN C:

```
> mpicc -DGAUSS -DUSE_MPI -DPHASE_ON -DSTOKESI -DWRITE_DATA -lfftw3_mpi -lfftw3 -lm test_clib.c gridding_library.c fft_library.c phase_correction_library.c -o test.exe
```

To run the library
```
> mpirun -np N test.exe
```

<br>

## Contacts

For feedbacks, suggestions, and requests [contact us](mailto:emanuele.derubeis2@unibo.it)!: emanuele.derubeis2@unibo.it
