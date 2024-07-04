cmake_minimum_required(VERSION 3.19...3.25)

find_program(NVHPC_CXX_BIN "nvc++" REQUIRED)
set(CMAKE_CXX_COMPILER ${NVHPC_CXX_BIN})

find_program(NVHPC_C_BIN "nvc" REQUIRED)
set(CMAKE_C_COMPILER ${NVHPC_C_BIN})

project(jaxdecomp LANGUAGES CXX CUDA)

# NVCC 12 does not support C++20
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

# Latest JAX v0.4.26 no longer supports cuda 11.8
find_package(CUDAToolkit REQUIRED VERSION 12)
set(NVHPC_CUDA_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})

message(STATUS "Using CUDA ${NVHPC_CUDA_VERSION}")
# Build Release by default
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build.")

add_subdirectory(third_party/cuDecomp)

option(CUDECOMP_BUILD_FORTRAN "Build Fortran bindings" OFF)
option(CUDECOMP_ENABLE_NVSHMEM "Enable NVSHMEM" OFF)
option(CUDECOMP_BUILD_EXTRAS "Build benchmark, examples, and tests" OFF)

set(CUDECOMP_CUDA_CC_LIST "70;80" CACHE STRING "List of CUDA compute capabilities to build cuDecomp for.")


# Add pybind11 and cuDecomp subdirectories
add_subdirectory(pybind11)


find_package(NVHPC REQUIRED COMPONENTS MATH MPI NCCL)

string(REPLACE "/lib64" "/include" NVHPC_MATH_INCLUDE_DIR ${NVHPC_MATH_LIBRARY_DIR})
string(REPLACE "/lib64" "/include" NVHPC_CUDA_INCLUDE_DIR ${NVHPC_CUDA_LIBRARY_DIR})


find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS ${NVHPC_NCCL_LIBRARY_DIR}
  )
string(REPLACE "/lib" "/include" NCCL_INCLUDE_DIR ${NVHPC_NCCL_LIBRARY_DIR})


message(STATUS "Using NCCL library: ${NCCL_LIBRARY}")
message(STATUS "NVHPC NCCL lib dir: ${NVHPC_NCCL_LIBRARY_DIR}")
message(STATUS "NCCL include dir: ${NCCL_INCLUDE_DIR}")

# Add _jaxdecomp modulei
pybind11_add_module(_jaxdecomp
                      src/halo.cu
                      src/jaxdecomp.cc
                      src/grid_descriptor_mgr.cc
                      src/fft.cu
                      src/transpose.cu
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
target_link_libraries(_jaxdecomp PRIVATE stdc++fs)
set_target_properties(_jaxdecomp PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(_jaxdecomp PROPERTIES INSTALL_RPATH "$ORIGIN/lib")

install(TARGETS _jaxdecomp LIBRARY DESTINATION . PUBLIC_HEADER DESTINATION .)
