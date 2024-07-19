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

  static const char* cudalloc = std::getenv("JD_ALLOCATE_WITH_XLA");

  if (cudalloc = nullptr) {

    CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, m_GridConfig, reinterpret_cast<void**>(&m_WorkSizeBuffer),
                                       workspace_num_elements * dtype_size));

    StartTraceInfo(m_Tracer) << "cudaMalloc will Allocate for Halo_exchange" << std::endl;

  } else {
    m_WorkSizeBuffer = nullptr;
    StartTraceInfo(m_Tracer) << "XLA will Allocate for Halo_exchange" << std::endl;
  }

  return S_OK;
}

template <typename real_t>
HRESULT HaloExchange<real_t>::halo_exchange(cudecompHandle_t handle, haloDescriptor_t desc, cudaStream_t stream,
                                            void** buffers) {
  void* data_d = buffers[0];
  void* work_d = buffers[1];

  void* buffer_to_user = nullptr;
  if (m_WorkSizeBuffer != nullptr) {
    // CUDA allocate buffer and managed by me
    buffer_to_user = m_WorkSizeBuffer;
  } else {
    // XLA allocate buffer and managed by XLA
    buffer_to_user = work_d;
  }

  //  Perform halo exchange along the three dimensions
  for (int i = 0; i < 3; ++i) {
    switch (desc.axis) {
    case 0:
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, m_GridConfig, data_d, buffer_to_user,
                                               get_cudecomp_datatype(real_t(0)), m_PencilInfo.halo_extents,
                                               desc.halo_periods.data(), i, stream));
      break;
    case 1:
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosY(handle, m_GridConfig, data_d, buffer_to_user,
                                               get_cudecomp_datatype(real_t(0)), m_PencilInfo.halo_extents,
                                               desc.halo_periods.data(), i, stream));
      break;
    case 2:
      CHECK_CUDECOMP_EXIT(cudecompUpdateHalosZ(handle, m_GridConfig, data_d, buffer_to_user,
                                               get_cudecomp_datatype(real_t(0)), m_PencilInfo.halo_extents,
                                               desc.halo_periods.data(), i, stream));
      break;
    }
  }

  return S_OK;
};

template <typename real_t> HRESULT HaloExchange<real_t>::cleanUp(cudecompHandle_t handle) {
  // Destroy the memory buffer allocate in case of cudaMalloc
  // In case of XLA allocation, this buffer is nullptr
  if (m_WorkSizeBuffer != nullptr) { CHECK_CUDECOMP_EXIT(cudecompFree(handle, m_GridConfig, m_WorkSizeBuffer)); }
  return S_OK;
}

template class HaloExchange<float>;
template class HaloExchange<double>;
} // namespace jaxdecomp
