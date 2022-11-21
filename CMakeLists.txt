cmake_minimum_required(VERSION 3.4...3.18)

project(jaxdecomp LANGUAGES CXX CUDA)

find_package(NVHPC REQUIRED COMPONENTS MATH MPI)

add_subdirectory(pybind11)

# Adding a bunch of inputs
include_directories("${NVHPC_ROOT_DIR}/math_libs/include" "${NVHPC_ROOT_DIR}/comm_libs/nccl/include")
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)
include_directories("third_party/cuDecomp/build/include")
link_directories("third_party/cuDecomp/build/lib")


pybind11_add_module(_jaxdecomp src/fft.cu src/jaxdecomp.cc)
target_link_libraries(_jaxdecomp PRIVATE MPI::MPI_CXX NVHPC::MPI NVHPC::CUFFT "-lcudecomp")
