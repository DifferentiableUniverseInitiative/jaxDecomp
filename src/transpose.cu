#include <cuda_runtime.h>
#include <cudecomp.h>
#include <iostream>
#include <regex>

#include "checks.h"
#include "transpose.h"

namespace jaxdecomp {

static inline cudecompDataType_t get_cudecomp_datatype(float) { return CUDECOMP_FLOAT; }
static inline cudecompDataType_t get_cudecomp_datatype(double) { return CUDECOMP_DOUBLE; }

template <typename real_t>
HRESULT Transpose<real_t>::get_transpose_descriptor(cudecompHandle_t handle, size_t& work_size,
                                                    transposeDescriptor& transpose_desc) {

  cudecompGridDescConfig_t& config = transpose_desc.config;

  // Create the grid description
  cudecompGridDesc_t grid_desc;
  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));

  // Get workspace sizes
  int64_t transpose_work_num_elements;
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, &transpose_work_num_elements));

  int64_t dtype_size;
  CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(get_cudecomp_datatype(real_t(0)), &dtype_size));

  work_size = dtype_size * transpose_work_num_elements;
  m_GridConfig = grid_desc;

  return S_OK;
}

// DEBUG ONLY ... I WARN YOU
template <typename real_t>
void Transpose<real_t>::inspect_device_array(void* data, bool transposed, cudaStream_t stream) {

  int rank(0);
  int size(0);
  CHECK_MPI_EXIT(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI_EXIT(MPI_Comm_size(MPI_COMM_WORLD, &size));
  // Copy input to host
  int px = !transposed ? 0 : rank % 2;
  int py = !transposed ? rank / 2 : 0;
  int pz = !transposed ? rank % 2 : rank / 2;
  const int nx = !transposed ? 4 : 2;
  const int ny = !transposed ? 2 : 4;
  const int nz = 2;
  const int local_size = nx * ny * nz;
  int* host = new int[local_size];

  cudaMemcpyAsync(host, data, sizeof(int) * local_size, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Flat print" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  for (int r = 0; r < size; r++) {

    if (rank == r) // to force printing in order so I have less headache
      for (int i = 0; i < nx * ny * nz; i++) {
        int global_indx = i + nx * px + ny * py + nz * pz;
        std::cout << "Rank[" << rank << "] Element [" << i << "] : global indexation: [" << global_indx
                  << "] : " << host[i] << std::endl;
      }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  cudaStreamSynchronize(stream);
  std::cout << "md array print" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);
  if (transposed) {
    for (int r = 0; r < size; r++) {
      for (int x = 0; x < nx; x++) {
        for (int z = 0; z < nz; z++) {
          for (int y = 0; y < ny; y++) {
            if (rank == r) {
              int indx = y + z * ny + x * ny * nx;
              std::cout << "Rank[" << rank << "] Element (" << y << "," << z << "," << x
                        << ") with global indexation: (" << y + py * ny << "," << z + pz * nz << "," << x + px * nx
                        << ") : " << (int)host[indx] << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
          }
        }
      }
    }

  } else {
    for (int r = 0; r < size; r++) {
      for (int z = 0; z < nz; z++) {
        for (int y = 0; y < ny; y++) {
          for (int x = 0; x < nx; x++) {
            if (rank == r) {
              int indx = x + y * nx + z * nx * ny;
              std::cout << "Rank[" << rank << "] Element (" << x << "," << y << "," << z
                        << ") with global indexation: (" << x + px * nx << "," << y + py * ny << "," << z + pz * nz
                        << ") : " << (int)host[indx] << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
          }
        }
      }
    }
  }
}

template <typename real_t>
HRESULT Transpose<real_t>::transpose(cudecompHandle_t handle, transposeDescriptor desc, cudaStream_t stream,
                                     void** buffers) {
  void* data_d = buffers[0];
  void* work_d = buffers[1];
  HRESULT hr = S_OK;
  switch (desc.transpose_type) {
  case TransposeType::TRANSPOSE_XY:
    CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, m_GridConfig, data_d, data_d, work_d,
                                              get_cudecomp_datatype(real_t(0)), nullptr, nullptr, stream));
    break;
  case TransposeType::TRANSPOSE_YZ:
    CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, m_GridConfig, data_d, data_d, work_d,
                                              get_cudecomp_datatype(real_t(0)), nullptr, nullptr, stream));
    break;
  case TransposeType::TRANSPOSE_ZY:
    CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, m_GridConfig, data_d, data_d, work_d,
                                              get_cudecomp_datatype(real_t(0)), nullptr, nullptr, stream));
    break;
  case TransposeType::TRANSPOSE_YX:
    CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, m_GridConfig, data_d, data_d, work_d,
                                              get_cudecomp_datatype(real_t(0)), nullptr, nullptr, stream));
    break;
  default: hr = E_INVALIDARG; break;
  }

  return hr;
}

template <typename real_t> HRESULT Transpose<real_t>::Release(cudecompHandle_t handle) {
  if (m_GridConfig) {
    CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, m_GridConfig));
    m_GridConfig = nullptr;
  }
  return S_OK;
}

template class Transpose<float>;
template class Transpose<double>;

} // namespace jaxdecomp
