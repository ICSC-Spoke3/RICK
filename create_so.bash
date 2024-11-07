mpicc -fopenmp -shared -o gridding.so -fPIC -DKAISERBESSEL gridding_library.c
mpicc -fopenmp -shared -o fft.so -fPIC fft_library.c -I/usr/local/include -L/usr/local/lib -lfftw3_omp -lfftw3_mpi -lfftw3
mpicc -fopenmp -shared -o phasecorr.so -fPIC phase_correction_library.c
