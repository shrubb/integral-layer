project(Integral)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")

find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)

if (CMAKE_VERSION VERSION_LESS 3.1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

add_library(integral-c SHARED integral-c.cpp integral-strided-c.cpp)
add_library(cityscapes-c SHARED "cityscapes-c.cpp")

#### CUDA implementation ###

find_package(Torch REQUIRED)
include_directories(${Torch_INSTALL_INCLUDE})

find_package(CUDA)

if (CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    cuda_add_library(integral-cuda SHARED integral-cuda.cu integral-zp-cuda.cu integral-strided-cuda.cu)
    cuda_add_library(integral-varscale-cuda SHARED integral-varscale-cuda.cu)
    cuda_add_library(integral-toborder-cuda SHARED integral-toborder-cuda.cu)
    cuda_add_library(flip-cuda SHARED flip.cu)
    
    cuda_add_cublas_to_target(integral-cuda)
    cuda_add_cublas_to_target(integral-toborder-cuda)

    target_link_libraries(integral-cuda TH)
endif()
