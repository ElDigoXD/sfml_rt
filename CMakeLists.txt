cmake_minimum_required(VERSION 3.27)
cmake_policy(SET CMP0135 NEW)

set(BUILD_SHARED_LIBS OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_COMPILER /usr/bin/clang++-16)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDAToolkit_ROOT /usr/local/cuda-12.3/bin)
set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/nvcc)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_ARCHITECTURES "native")
set(USE_NATIVE_ARCH)

set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "--generate-line-info -Xcompiler -fopenmp -Xcompiler -march=native -Xcompiler --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/12 --diag-suppress 549 --diag-suppress 177")

SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread -lfftw3 -lm -fopenmp=libiomp5") #
#include(FindFreetype)

project(sfml_rt LANGUAGES CXX CUDA)

#add_subdirectory(src)

add_subdirectory(src_cuda)