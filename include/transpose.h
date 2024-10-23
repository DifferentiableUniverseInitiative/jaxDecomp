#ifndef _JAX_DECOMP_TRANSPOSE_H_
#define _JAX_DECOMP_TRANSPOSE_H_

#include "checks.h"
#include <cstdint>
#include <cudecomp.h>
#include <pthread.h>
namespace jaxdecomp {

enum class TransposeType {
  TRANSPOSE_XY,
  TRANSPOSE_YZ,
  TRANSPOSE_ZY,
  TRANSPOSE_YX,
  TRANSPOSE_XZ,
  TRANSPOSE_ZX,
  UNKNOWN_TRANSPOSE
};

class transposeDescriptor {
public:
  TransposeType transpose_type = TransposeType::UNKNOWN_TRANSPOSE;
  bool contiguous = true;
  cudecompGridDescConfig_t config;
  bool double_precision = false;

  transposeDescriptor() = default;
  transposeDescriptor(const transposeDescriptor& other) = default;
  ~transposeDescriptor() = default;

  transposeDescriptor(cudecompGridDescConfig_t& config, const TransposeType& type, const bool& double_precision,
                      const bool& contiguous)
      : config(config), transpose_type(type), double_precision(double_precision), contiguous(contiguous) {
    this->config.transpose_axis_contiguous[0] = contiguous;
    this->config.transpose_axis_contiguous[1] = contiguous;
    this->config.transpose_axis_contiguous[2] = contiguous;
  }

  bool operator==(const transposeDescriptor& other) const {
    return (config.gdims[0] == other.config.gdims[0] && config.gdims[1] == other.config.gdims[1] &&
            config.gdims[2] == other.config.gdims[2] && config.pdims[0] == other.config.pdims[0] &&
            config.pdims[1] == other.config.pdims[1] && double_precision == other.double_precision &&
            contiguous == other.contiguous && transpose_type == other.transpose_type &&
            config.transpose_comm_backend == other.config.transpose_comm_backend &&
            config.halo_comm_backend == other.config.halo_comm_backend);
  }
};

template <typename real_t> class Transpose {
  friend class GridDescriptorManager;

public:
  Transpose() = default;
  ~Transpose() = default;

  HRESULT get_transpose_descriptor(cudecompHandle_t handle, size_t& work_size, transposeDescriptor& transpose_desc);
  HRESULT transpose(cudecompHandle_t handle, transposeDescriptor desc, cudaStream_t stream, void** buffers);
  HRESULT Release(cudecompHandle_t handle);

private:
  cudecompGridDesc_t m_GridConfig;
  cudecompGridDescConfig_t m_GridDescConfig;
  int64_t m_WorkSize;
};

} // namespace jaxdecomp
namespace std {
template <> struct hash<jaxdecomp::transposeDescriptor> {
  size_t operator()(const jaxdecomp::transposeDescriptor& desc) const {
    size_t hash = 0;
    hash = hash ^ std::hash<int32_t>()(desc.config.gdims[0]);
    hash = hash ^ std::hash<int32_t>()(desc.config.gdims[1]);
    hash = hash ^ std::hash<int32_t>()(desc.config.gdims[2]);
    hash = hash ^ std::hash<int32_t>()(desc.config.pdims[0]);
    hash = hash ^ std::hash<int32_t>()(desc.config.pdims[1]);
    hash = hash ^ std::hash<bool>()(desc.double_precision);
    hash = hash ^ std::hash<int>()(desc.config.transpose_comm_backend);
    hash = hash ^ std::hash<int>()(desc.config.halo_comm_backend);
    hash = hash ^ std::hash<bool>()(desc.contiguous);
    return hash;
  }
};
} // namespace std

#endif // _JAX_DECOMP_TRANSPOSE_H_
