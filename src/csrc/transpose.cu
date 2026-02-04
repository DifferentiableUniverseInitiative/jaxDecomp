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

template <typename real_t>
HRESULT Transpose<real_t>::transpose(cudecompHandle_t handle, transposeDescriptor desc, cudaStream_t stream,
                                     void** buffers) {
  void* data_d = buffers[0];
  void* work_d = buffers[1];
  HRESULT hr = S_OK;
  switch (desc.transpose_type) {
  case TransposeType::TRANSPOSE_XY:
    CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, m_GridConfig, data_d, data_d, work_d,
                                              get_cudecomp_datatype(real_t(0)), nullptr, nullptr, nullptr, nullptr,
                                              stream));
    break;
  case TransposeType::TRANSPOSE_YZ:
    CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, m_GridConfig, data_d, data_d, work_d,
                                              get_cudecomp_datatype(real_t(0)), nullptr, nullptr, nullptr, nullptr,
                                              stream));
    break;
  case TransposeType::TRANSPOSE_ZY:
    CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, m_GridConfig, data_d, data_d, work_d,
                                              get_cudecomp_datatype(real_t(0)), nullptr, nullptr, nullptr, nullptr,
                                              stream));
    break;
  case TransposeType::TRANSPOSE_YX:
    CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, m_GridConfig, data_d, data_d, work_d,
                                              get_cudecomp_datatype(real_t(0)), nullptr, nullptr, nullptr, nullptr,
                                              stream));
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
