#ifndef _JAX_DECOMP_HALO_H_
#define _JAX_DECOMP_HALO_H_

#include "checks.h"
#include <array>
#include <cstdint>
#include <cudecomp.h>
#include <pthread.h>

namespace jaxdecomp {

class haloDescriptor_t {
public:
  bool double_precision = false;
  std::array<int32_t, 3> halo_extents{0, 0, 0};
  std::array<bool, 3> halo_periods{true, true, true};
  int axis = 0;                    // The axis long which the pencil is aligned
  cudecompGridDescConfig_t config; // Descriptor for the grid

  haloDescriptor_t() = default;
  haloDescriptor_t(const haloDescriptor_t &other) = default;
  ~haloDescriptor_t() = default;

  bool operator==(const haloDescriptor_t &other) const {
    return (double_precision == other.double_precision &&
            halo_extents == other.halo_extents &&
            halo_periods == other.halo_periods && axis == other.axis &&
            config.gdims[0] == other.config.gdims[0] &&
            config.gdims[1] == other.config.gdims[1] &&
            config.gdims[2] == other.config.gdims[2] &&
            config.pdims[0] == other.config.pdims[0] &&
            config.pdims[1] == other.config.pdims[1]);
  }
};

template <typename real_t> class HaloExchange {
  friend class GridDescriptorManager;

public:
  HaloExchange() = default;
  // Grid descriptors are handled by the GridDescriptorManager
  ~HaloExchange() = default;

  HRESULT get_halo_descriptor(cudecompHandle_t handle, size_t &work_size,
                              haloDescriptor_t &halo_desc);
  HRESULT halo_exchange(cudecompHandle_t handle, haloDescriptor_t desc,
                        cudaStream_t stream, void **buffers);

private:
  cudecompGridDesc_t m_GridConfig;
  cudecompGridDescConfig_t m_GridDescConfig;
  cudecompPencilInfo_t m_PencilInfo;

  int64_t m_WorkSize;
};

} // namespace jaxdecomp

namespace std {
template <> struct hash<jaxdecomp::haloDescriptor_t> {
  std::size_t operator()(const jaxdecomp::haloDescriptor_t &descriptor) const {
    std::size_t h1 = std::hash<bool>{}(descriptor.double_precision);
    h1 ^= std::hash<int32_t>{}(descriptor.halo_extents[0]);
    h1 ^= std::hash<int32_t>{}(descriptor.halo_extents[1]);
    h1 ^= std::hash<int32_t>{}(descriptor.halo_extents[2]);
    h1 ^= std::hash<bool>{}(descriptor.halo_periods[0]);
    h1 ^= std::hash<bool>{}(descriptor.halo_periods[1]);
    h1 ^= std::hash<bool>{}(descriptor.halo_periods[2]);
    h1 ^= std::hash<int>{}(descriptor.axis);
    h1 ^= std::hash<int32_t>{}(descriptor.config.gdims[0]);
    h1 ^= std::hash<int32_t>{}(descriptor.config.gdims[1]);
    h1 ^= std::hash<int32_t>{}(descriptor.config.gdims[2]);
    h1 ^= std::hash<int32_t>{}(descriptor.config.pdims[0]);
    h1 ^= std::hash<int32_t>{}(descriptor.config.pdims[1]);

    return h1;
  }
};
} // namespace std

#endif
