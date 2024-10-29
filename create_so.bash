mpicc -shared -o gridding.so -fPIC gridding_library_new.c
mpicc -I/usr/local/include -L/usr/local/lib -lfftw3 -lfftw3_mpi -lfftw3_omp -shared -o fft.so -fPIC fft_library.c
mpicc -shared -o phasecorr.so -fPIC phase_correction_library.c
