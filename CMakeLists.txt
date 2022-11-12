cmake_minimum_required(VERSION 3.4...3.18)
# TODO: automatize all of this
list(APPEND CMAKE_PREFIX_PATH 
    "/opt/nvidia/hpc_sdk/Linux_x86_64/22.9/cmake" 
    "/local/home/flanusse/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")
include_directories("/opt/nvidia/hpc_sdk/Linux_x86_64/22.9/math_libs/include/")

project(jaxdecomp)

find_package(NVHPC REQUIRED COMPONENTS MATH MPI CUDA)
find_package(pybind11 CONFIG REQUIRED)

enable_language(CUDA)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)
include_directories("third_party/cuDecomp/build/include")
link_directories("third_party/cuDecomp/build/lib")

pybind11_add_module(_jaxdecomp src/jaxdecomp.cc)
target_link_libraries(_jaxdecomp PUBLIC MPI::MPI_CXX "-lcudecomp -lopen-pal -lopen-rte" )
