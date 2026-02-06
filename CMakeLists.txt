cmake_minimum_required(VERSION 3.19...3.25)

option(JD_CUDECOMP_BACKEND "Use cuDecomp backend" OFF)

if(JD_CUDECOMP_BACKEND)
  find_program(NVHPC_CXX_BIN "nvc++" REQUIRED)
  set(CMAKE_CXX_COMPILER ${NVHPC_CXX_BIN})
  add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)
  add_compile_definitions(__GXX_ABI_VERSION=1013)
  # set CXXFLAGS for NVHPC compiler
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -noswitcherror")
endif()

project(jaxdecomp LANGUAGES CXX)

set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# NVCC 12 does not support C++20
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

# Add the executable
find_package(
  Python 3.8 REQUIRED
  COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c"
          "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE XLA_FFI_INCLUDE_DIR)
message(STATUS "XLA FFI include directory: ${XLA_FFI_INCLUDE_DIR}")

# Detect the installed nanobind package and import it into CMake
find_package(nanobind CONFIG REQUIRED)


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

  # Add _jaxdecomp module
  nanobind_add_module(_jaxdecomp
                        STABLE_ABI
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
    ${XLA_FFI_INCLUDE_DIR}
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
  nanobind_add_module(_jaxdecomp STABLE_ABI src/csrc/jaxdecomp.cc)
  target_include_directories(_jaxdecomp PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/csrc/include ${XLA_FFI_INCLUDE_DIR})
  target_compile_definitions(_jaxdecomp PRIVATE JD_JAX_BACKEND)
endif()

set_target_properties(_jaxdecomp PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
install(TARGETS _jaxdecomp LIBRARY DESTINATION jaxdecomplib PUBLIC_HEADER DESTINATION jaxdecomplib)
