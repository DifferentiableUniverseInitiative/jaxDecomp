#ifndef _JAX_DECOMP_HALO_H_
#define _JAX_DECOMP_HALO_H_

#include <cudecomp.h>

namespace jaxdecomp
{

    typedef struct
    {
        bool double_precision = false;
        std::array<bool, 3> halo_periods{true, true, true};
        int axis = 0;                    // The axis long which the pencil is aligned
        cudecompGridDescConfig_t config; // Descriptor for the grid
    } haloDescriptor_t;

    std::pair<int64_t, haloDescriptor_t> get_halo_descriptor(cudecompHandle_t handle,
                                                             cudecompGridDescConfig_t config,
                                                             std::array<bool, 3> halo_periods,
                                                             int axis, bool double_precision);

    template <typename real_t>
    void halo_exchange(cudecompHandle_t handle,
                       haloDescriptor_t desc,
                       cudaStream_t stream,
                       void **buffers);

};

#endif