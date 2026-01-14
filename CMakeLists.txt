cmake_minimum_required(VERSION 3.16)

# Project name and languages
# CXX is required for HeFFTe and correct OpenMP/C++ linking
project(RickMPI LANGUAGES C CXX)

# --- 1. Project Definitions ---
add_compile_definitions(
    KAISERBESSEL
    USE_MPI
    STOKESI
    PHASE_ON
)

option(ENABLE_FITSIO "Enable FITSIO support" OFF)

# --- 2. Find Dependencies ---

# A. MPI (Required)
find_package(MPI REQUIRED)

# B. OpenMP (Required)
find_package(OpenMP REQUIRED)

# C. FFTW3 + FFTW3_OMP (Required for fft_library.c)
# We search for the FFTW3 package and the OpenMP variant
find_package(PkgConfig)
pkg_check_modules(FFTW3 REQUIRED fftw3)
find_library(FFTW3_OMP_LIB NAMES fftw3_omp HINTS ${FFTW3_LIBRARY_DIRS})

# D. HeFFTe
find_path(HEFFTE_INCLUDE_DIR NAMES heffte.h
          HINTS ${HEFFTE_ROOT} $ENV{HEFFTE_ROOT}
          PATH_SUFFIXES include)

find_library(HEFFTE_LIBRARY NAMES heffte
             HINTS ${HEFFTE_ROOT} $ENV{HEFFTE_ROOT}
             PATH_SUFFIXES lib lib64)

if(NOT HEFFTE_INCLUDE_DIR OR NOT HEFFTE_LIBRARY)
    message(FATAL_ERROR "HeFFTe not found! Please set HEFFTE_ROOT.")
endif()

# E. CFITSIO (Optional)
if(ENABLE_FITSIO)
    find_package(CFITSIO REQUIRED)
    add_compile_definitions(FITSIO)
endif()

# --- 3. Source Files Configuration ---
set(SOURCES
    test_clib.c
    gridding_library.c
    phase_correction_library.c
    mpi_functions_redefined.c
    fft_library.c 
)

# --- 4. Target Definition ---
add_executable(rick ${SOURCES})

# --- 5. Compiler Flags & Optimizations ---
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|Intel")
    
    # Initialize a list with common optimization and warning flags
    # -O3: High-level optimization
    # -Wno-unused-result: Silence warnings for ignored return values (e.g., fscanf)
    set(RICK_FLAGS "-O3" "-Wno-unused-result")

    # Architecture-specific optimizations
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
        # Flags for Intel/AMD architectures
        # -march=native: Optimize for the host CPU instructions
        # -mavx -mavx2: Enable Advanced Vector Extensions
        list(APPEND RICK_FLAGS "-march=native" "-mtune=native" "-mavx" "-mavx2")
        message(STATUS "Architecture detected as x86_64. Enabling AVX/AVX2 flags.")
        
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        # Flags for ARM architectures (Apple Silicon, AWS Graviton, etc.)
        # AVX flags are removed as they are incompatible with ARM
        list(APPEND RICK_FLAGS "-march=native" "-mtune=native")
        message(STATUS "Architecture detected as ARM64. Using NEON/Native optimizations.")
        
    else()
        # Fallback for other architectures
        message(STATUS "Unknown architecture (${CMAKE_SYSTEM_PROCESSOR}). Using generic O3.")
    endif()

    # Apply the gathered flags to the 'rick' target
    # We also include OpenMP specific flags detected by find_package(OpenMP)
    target_compile_options(rick PRIVATE ${RICK_FLAGS} ${OpenMP_C_FLAGS})

endif()

# --- 6. Include Directories and Linking ---
target_include_directories(rick PRIVATE
    ${MPI_C_INCLUDE_DIRS}
    ${HEFFTE_INCLUDE_DIR}
    ${FFTW3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(rick PRIVATE
    MPI::MPI_C
    OpenMP::OpenMP_C
    ${HEFFTE_LIBRARY}
    ${FFTW3_LIBRARIES}
    ${FFTW3_OMP_LIB}   # <--- Crucial fix for 'fftw_plan_with_nthreads'
    m
)

if(ENABLE_FITSIO)
    target_link_libraries(rick PRIVATE CFITSIO::CFITSIO)
endif()

# --- 7. Installation ---
install(TARGETS rick DESTINATION bin)
