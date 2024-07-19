#include <cuda_runtime.h>
#include <cudecomp.h>
#include <iostream>
#include <regex>

#include "checks.h"
#include "fft.h"
#include "halo.h"

namespace jaxdecomp {

static inline cudecompDataType_t get_cudecomp_datatype(float) { return CUDECOMP_FLOAT; }
static inline cudecompDataType_t get_cudecomp_datatype(double) { return CUDECOMP_DOUBLE; }

template <typename real_t>
HRESULT HaloExchange<real_t>::get_halo_descriptor(cudecompHandle_t handle, size_t& work_size,
                                                  haloDescriptor_t& halo_desc) {

  cudecompGridDescConfig_t& config = halo_desc.config;

  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &m_GridConfig, &config, nullptr));

  // Get pencil information for the specified axis
  CHECK_CUDECOMP_EXIT(
      cudecompGetPencilInfo(handle, m_GridConfig, &m_PencilInfo, halo_desc.axis, halo_desc.halo_extents.data()));

  // Get workspace size
  int64_t workspace_num_elements;
  CHECK_CUDECOMP_EXIT(cudecompGetHaloWorkspaceSize(handle, m_GridConfig, halo_desc.axis, m_PencilInfo.halo_extents,
                                                   &workspace_num_elements));

  // TODO(Wassim) Handle complex numbers
  int64_t dtype_size;
  if (halo_desc.double_precision)
    CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(CUDECOMP_DOUBLE, &dtype_size));
  else
    CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(CUDECOMP_FLOAT, &dtype_size));

  m_WorkSize = dtype_size * workspace_num_elements;
  work_size = m_WorkSize;

  return S_OK;
}

template <typename real_t>
HRESULT HaloExchange<real_t>::halo_exchange(cudecompHandle_t handle, haloDescriptor_t desc, cudaStream_t stream,
                                            void** buffers) {
  void* data_d = buffers[0];
  void* work_d = buffers[1];

  //  Perform halo exchange along the three dimensions
  for (int i = 0; i < 3; ++i) {
    switch (desc.axis) {
    case 0:
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, m_GridConfig, data_d, work_d, get_cudecomp_datatype(real_t(0)),
                                               m_PencilInfo.halo_extents, desc.halo_periods.data(), i, stream));
      break;
    case 1:
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosY(handle, m_GridConfig, data_d, work_d, get_cudecomp_datatype(real_t(0)),
                                               m_PencilInfo.halo_extents, desc.halo_periods.data(), i, stream));
      break;
    case 2:
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosZ(handle, m_GridConfig, data_d, work_d, get_cudecomp_datatype(real_t(0)),
                                               m_PencilInfo.halo_extents, desc.halo_periods.data(), i, stream));
      break;
    }
  }

  return S_OK;
};

template <typename real_t> HRESULT HaloExchange<real_t>::cleanUp(cudecompHandle_t handle) {
  //  XLA is doing the allocation
  // nothing to clean up
  return S_OK;
}

template class HaloExchange<float>;
template class HaloExchange<double>;
} // namespace jaxdecomp
