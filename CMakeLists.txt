cmake_minimum_required(VERSION 3.19...3.22)

set(CMAKE_CXX_STANDARD 20)
set(CUDAToolkit_Root $ENV{CUDA_HOME})
set(CMAKE_CUDA_COMPILER "${CUDAToolkit_Root}/bin/nvcc")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CUDECOMP_BUILD_FORTRAN "Build Fortran bindings" OFF)
option(CUDECOMP_ENABLE_NVSHMEM "Enable NVSHMEM" OFF)
option(CUDECOMP_BUILD_EXTRAS "Build benchmark, examples, and tests" OFF)

set(CUDECOMP_CUDA_CC_LIST "70;80;90" CACHE STRING "List of CUDA compute capabilities to build cuDecomp for.")

project(jaxdecomp LANGUAGES CXX CUDA)

# Add pybind11 and cuDecomp subdirectories
add_subdirectory(pybind11)

set(NVHPC_CUDA_VERSION 11.8)
# Build debug
set(CMAKE_BUILD_TYPE Debug)
add_subdirectory(third_party/cuDecomp)

find_package(NVHPC REQUIRED COMPONENTS MATH MPI NCCL)

string(REPLACE "/lib64" "/include" NVHPC_MATH_INCLUDE_DIR ${NVHPC_MATH_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_CUDA_INCLUDE_DIR ${NVHPC_CUDA_LIBRARY_DIR})


find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS ${NVHPC_NCCL_LIBRARY_DIR}
  )
  string(REPLACE "/lib" "/include" NCCL_INCLUDE_DIR ${NVHPC_NCCL_LIBRARY_DIR})


message(STATUS "Using NCCL library: ${NCCL_LIBRARY}")
message(STATUS "NCCL lib dir: ${NVHPC_NCCL_LIBRARY_DIR}")
message(STATUS "NCCL lib dir: ${NCCL_INCLUDE_DIR}")

# Add _jaxdecomp modulei
pybind11_add_module(_jaxdecomp 
                      src/halo.cu 
                      src/jaxdecomp.cc
                      src/GridDescriptorManager.cpp 
                      src/fft.cu 
)

set_target_properties(_jaxdecomp PROPERTIES CUDA_ARCHITECTURES "${CUDECOMP_CUDA_CC_LIST}")

target_include_directories(_jaxdecomp
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cuDecomp/include
  ${NVHPC_CUDA_INCLUDE_DIR}
  ${MPI_CXX_INCLUDE_DIRS}
  ${NVHPC_MATH_INCLUDE_DIR}
  ${NCCL_INCLUDE_DIR}
)

target_link_libraries(_jaxdecomp PRIVATE MPI::MPI_CXX)
target_link_libraries(_jaxdecomp PRIVATE NVHPC::CUFFT)
target_link_libraries(_jaxdecomp PRIVATE NVHPC::CUTENSOR)
target_link_libraries(_jaxdecomp PRIVATE NVHPC::CUDA)
target_link_libraries(_jaxdecomp PRIVATE ${NCCL_LIBRARY})
target_link_libraries(_jaxdecomp PRIVATE cudecomp)
set_target_properties(_jaxdecomp PROPERTIES LINKER_LANGUAGE CXX)
