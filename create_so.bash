mpicc -shared -o gridding.so -fPIC -DKAISERBESSEL gridding_library.c
mpicc -shared -o fft.so -fPIC fft_library.c -I/usr/local/include -L/usr/local/lib -lfftw3 -lfftw3_mpi
mpicc -shared -o phasecorr.so -fPIC phase_correction_library.c
