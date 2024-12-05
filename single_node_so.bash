#!/bin/bash


#modules to load: fftw, openmpi, nvhpc
rm *.cu

cp gridding_library.cpp gridding_library.cu

nvcc -shared -o librick_gridding.so -Xcompiler -fPIC -std=c++17 --generate-code arch=compute_80,code=sm_80 -DKAISERBESSEL -DRICK_GPU gridding_library.cu -I/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/cuda/12.3/include/ -L/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/cuda/12.3/lib64/ -L/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/cuda/12.3/targets/x86_64-linux/lib/stubs/ -lcuda -lcudart -lnvidia-ml

cp fft_library.cpp fft_library.cu

nvcc -shared -o librick_fft_gpu.so -DRICK_GPU -Xcompiler -fPIC fft_library.cu -I/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/math_libs/12.3/include/ -L/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/math_libs/12.3/lib64/ -lcufft

cp phase_correction_library.cpp phase_correction_library.cu

nvcc -DPHASE_ON -DRICK_GPU -shared -o librick_phasecorr.so -Xcompiler -fPIC -std=c++17 --generate-code arch=compute_80,code=sm_80 phase_correction_library.cu -I/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/cuda/12.3/include/ -L/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/cuda/12.3/lib64/ -L/leonardo/prod/spack/5.2/install/0.21/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.3-v63z4inohb4ywjeggzhlhiuvuoejr2le/Linux_x86_64/24.3/cuda/12.3/targets/x86_64-linux/lib/stubs/ -lcuda -lcudart -lnvidia-ml


mpic++ -fopenmp -o test_clib test_clib.cpp -L. -lrick_gridding -lrick_fft_gpu -lrick_phasecorr