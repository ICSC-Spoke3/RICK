#!/bin/bash

# --- Rick Build Automation Script ---

# Exit immediately if a command exits with a non-zero status
set -e

# Define the build directory name
BUILD_DIR="build"

# 1. Clean up previous build artifacts
if [ -d "$BUILD_DIR" ]; then
    echo "--- Removing existing build directory: $BUILD_DIR ---"
    rm -rf "$BUILD_DIR"
fi

# 2. Create a fresh build directory
echo "--- Creating new build directory ---"
mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

# 3. Configure the project with CMake
# We set CMAKE_BUILD_TYPE to Release to enable the -O3 and -march=native flags
# defined in our CMakeLists.txt. 
# MPI and OpenMP detection is handled automatically.
echo "--- Running CMake configuration ---"
cmake -DCMAKE_BUILD_TYPE=Release ..

# 4. Compile the project
# 'nproc' detects the number of available CPU cores for parallel compilation
echo "--- Compiling with $(nproc) cores ---"
make -j$(nproc)

# 5. Provide feedback to the user
if [ -f "rick" ]; then
    echo "-------------------------------------------------------"
    echo "Build Successful! Executable 'rick' created."
    echo "Location: $(pwd)/rick"
    echo "To run it, use: mpirun -np <procs> ./build/rick [SIGMA]"
    echo "-------------------------------------------------------"
else
    echo "Build failed: Executable not found."
    exit 1
fi
