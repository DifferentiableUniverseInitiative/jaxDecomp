#ifndef _JAX_DECOMP_H_
#define _JAX_DECOMP_H_
#ifdef JD_CUDECOMP_BACKEND
#include "checks.h"
#include <cudecomp.h>
#endif
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef JD_JAX_BACKEND
enum cudecompTransposeCommBackend_t {
  CUDECOMP_TRANSPOSE_COMM_MPI_P2P = 1,    ///< MPI backend using peer-to-peer algorithm (i.e.,MPI_Isend/MPI_Irecv)
  CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL = 2, ///< MPI backend using peer-to-peer algorithm with pipelining
  CUDECOMP_TRANSPOSE_COMM_MPI_A2A = 3,    ///< MPI backend using MPI_Alltoallv
  CUDECOMP_TRANSPOSE_COMM_NCCL = 4,       ///< NCCL backend
  CUDECOMP_TRANSPOSE_COMM_NCCL_PL = 5,    ///< NCCL backend with pipelining
  CUDECOMP_TRANSPOSE_COMM_NVSHMEM = 6,    ///< NVSHMEM backend
  CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL = 7  ///< NVSHMEM backend with pipelining
};

/**
 * @brief This enum lists the different available halo backend options.
 */
enum cudecompHaloCommBackend_t {
  CUDECOMP_HALO_COMM_MPI = 1,             ///< MPI backend
  CUDECOMP_HALO_COMM_MPI_BLOCKING = 2,    ///< MPI backend with blocking between each peer transfer
  CUDECOMP_HALO_COMM_NCCL = 3,            ///< NCCL backend
  CUDECOMP_HALO_COMM_NVSHMEM = 4,         ///< NVSHMEM backend
  CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING = 5 ///< NVSHMEM backend with blocking between each peer transfer
};
#endif

namespace jaxdecomp {

#ifdef JD_JAX_BACKEND

enum class TransposeType {
  TRANSPOSE_XY,
  TRANSPOSE_YZ,
  TRANSPOSE_ZY,
  TRANSPOSE_YX,
  TRANSPOSE_XZ,
  TRANSPOSE_ZX,
  UNKNOWN_TRANSPOSE
};

enum Decomposition { slab_XY = 0, slab_YZ = 1, pencil = 2, no_decomp = 3 };

#endif

/**
 * @brief A data structure defining configuration options for grid descriptor creation.
 * Slightly adapted version of cudecompGridDescConfig_t which can be automatically translated by pybind11
 */
typedef struct {
  // Grid information
  std::array<int32_t, 3> gdims; ///< dimensions of global data grid
  // std::array<int32_t, 3> gdims_dist; ///< dimensions of global data grid to use for distribution
  std::array<int32_t, 2> pdims; ///< dimensions of process grid

  // Transpose settings
  cudecompTransposeCommBackend_t transpose_comm_backend; ///< communication backend to use for transpose communication
                                                         ///< (default: CUDECOMP_TRANSPOSE_COMM_MPI_P2P)
  // bool transpose_axis_contiguous[3]; ///< flag (by axis) indicating if memory should be contiguous along pencil axis
  // /< (default: [false, false, false])

  // Halo settings
  cudecompHaloCommBackend_t
      halo_comm_backend; ///< communication backend to use for halo communication (default: CUDECOMP_HALO_COMM_MPI)

} decompGridDescConfig_t;

#ifdef JD_CUDECOMP_BACKEND

void cudecompGridDescConfigSet(cudecompGridDescConfig_t* config, const decompGridDescConfig_t* source) {
  // Initialize the config with the defaults
  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(config));
  for (int i = 0; i < 3; i++) config->gdims[i] = source->gdims[i];
  for (int i = 0; i < 2; i++) config->pdims[i] = source->pdims[i];
  for (int i = 0; i < 3; i++) config->transpose_axis_contiguous[i] = true;
  config->halo_comm_backend = source->halo_comm_backend;
  config->transpose_comm_backend = source->transpose_comm_backend;
};
#endif

/**
 * @brief A data structure containing geometry information about a pencil data buffer.
 * Slightly adapted version of cudecompPencilInfo_t which can be automatically translated by pybind11
 */
typedef struct {
  std::array<int32_t, 3> shape;        ///< pencil shape (in local order, including halo elements)
  std::array<int32_t, 3> lo;           ///< lower bound coordinates (in local order, excluding halo elements)
  std::array<int32_t, 3> hi;           ///< upper bound coordinates (in local order, excluding halo elements)
  std::array<int32_t, 3> order;        ///< data layout order (e.g. 2,1,0 means memory is ordered Z,Y,X)
  std::array<int32_t, 3> halo_extents; ///< halo extents by dimension (in global order)
  int64_t size;                        ///< number of elements in pencil (including halo elements)
} decompPencilInfo_t;

#ifdef JD_CUDECOMP_BACKEND
void decompPencilInfoSet(decompPencilInfo_t* info, const cudecompPencilInfo_t* source) {
  for (int i = 0; i < 3; i++) {
    info->hi[i] = source->hi[i];
    info->lo[i] = source->lo[i];
    info->halo_extents[i] = source->halo_extents[i];
    info->order[i] = source->order[i];
    info->shape[i] = source->shape[i];
  }
  info->size = source->size;
};
#endif

}; // namespace jaxdecomp

#endif
