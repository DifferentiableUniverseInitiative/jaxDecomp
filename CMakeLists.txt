cmake_minimum_required(VERSION 3.19...3.25)



project(jaxdecomp LANGUAGES CXX)

# NVCC 12 does not support C++20
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

option(JD_CUDECOMP_BACKEND "Use cuDecomp backend" OFF)
# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER AND JD_CUDECOMP_BACKEND)
  enable_language(CUDA)

  # Latest JAX v0.4.26 no longer supports cuda 11.8
  find_package(CUDAToolkit REQUIRED VERSION 12)
  set(NVHPC_CUDA_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR})

  message(STATUS "Using CUDA ${NVHPC_CUDA_VERSION}")

  add_subdirectory(third_party/cuDecomp)

  option(CUDECOMP_BUILD_FORTRAN "Build Fortran bindings" OFF)
  option(CUDECOMP_ENABLE_NVSHMEM "Enable NVSHMEM" OFF)
  option(CUDECOMP_BUILD_EXTRAS "Build benchmark, examples, and tests" OFF)

  # 70: Volta, 80: Ampere, 89: RTX 4060
  set(CUDECOMP_CUDA_CC_LIST "70;80;89" CACHE STRING "List of CUDA compute capabilities to build cuDecomp for.")

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
                        src/csrc/halo.cu
                        src/csrc/jaxdecomp.cc
                        src/csrc/grid_descriptor_mgr.cc
                        src/csrc/fft.cu
                        src/csrc/transpose.cu
  )

  set_target_properties(_jaxdecomp PROPERTIES CUDA_ARCHITECTURES "${CUDECOMP_CUDA_CC_LIST}")

  target_include_directories(_jaxdecomp
    PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/src/csrc/include
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
  target_compile_definitions(_jaxdecomp PRIVATE JD_CUDECOMP_BACKEND)
else()
  pybind11_add_module(_jaxdecomp src/csrc/jaxdecomp.cc)
  target_include_directories(_jaxdecomp PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/csrc/include)
  target_compile_definitions(_jaxdecomp PRIVATE JD_JAX_BACKEND)
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(_jaxdecomp PROPERTIES INSTALL_RPATH "$ORIGIN/../lib;${NVHPC_MATH_LIBRARY_DIR}")
install(TARGETS _jaxdecomp LIBRARY DESTINATION jaxdecomplib PUBLIC_HEADER DESTINATION jaxdecomplib)
