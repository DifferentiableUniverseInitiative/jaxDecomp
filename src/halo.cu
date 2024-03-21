#include <cudecomp.h>
#include <cuda_runtime.h>

#include "halo.h"
#include "checks.h"

namespace jaxdecomp
{

    static inline cudecompDataType_t get_cudecomp_datatype(float) { return CUDECOMP_FLOAT; }
    static inline cudecompDataType_t get_cudecomp_datatype(double) { return CUDECOMP_DOUBLE; }

    std::pair<int64_t, haloDescriptor_t> get_halo_descriptor(cudecompHandle_t handle,
                                                             cudecompGridDescConfig_t config,
                                                             std::array<int32_t, 3> halo_extents,
                                                             std::array<bool, 3> halo_periods,
                                                             int axis, bool double_precision)
    {
        // Initializing the descriptor
        haloDescriptor_t desc;
        desc.double_precision = double_precision;
        desc.config = config;
        desc.axis = axis;
        desc.halo_periods = halo_periods;
        desc.halo_extents = halo_extents;

        /* Setting up cuDecomp grid specifications
         */
        cudecompGridDesc_t grid_desc;
        CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));

        // Get pencil information for the specified axis
        cudecompPencilInfo_t pinfo;
        CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pinfo, axis, halo_extents.data()));

        // Get workspace size
        int64_t workspace_num_elements;
        CHECK_CUDECOMP_EXIT(
            cudecompGetHaloWorkspaceSize(handle, grid_desc, axis, pinfo.halo_extents, &workspace_num_elements));

        int64_t dtype_size;
        if (double_precision)
            CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(CUDECOMP_DOUBLE, &dtype_size));
        else
            CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(CUDECOMP_FLOAT, &dtype_size));
        int64_t workspace_sz = dtype_size * workspace_num_elements;

        // Cleaning up the things that allocated memory
        CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));

        // Returning all we need to know about this transform
        return std::make_pair(workspace_sz, desc);
    };

    template <typename real_t>
    void halo_exchange(cudecompHandle_t handle,
                       haloDescriptor_t desc,
                       cudaStream_t stream,
                       void **buffers)
    {
        void *data_d = buffers[0];
        void *work_d = buffers[1];

        cudecompGridDesc_t grid_desc;
        CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &desc.config, nullptr));

        // Get pencil information for the specified axis
        cudecompPencilInfo_t pinfo;
        CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pinfo, desc.axis, desc.halo_extents.data()));

        // Perform halo exchange along the three dimensions
        for (int i = 0; i < 3; ++i)
        {
            switch (desc.axis)
            {
            case 0:
                CHECK_CUDECOMP_EXIT(cudecompUpdateHalosX(handle, grid_desc, data_d, work_d, get_cudecomp_datatype(real_t(0)),
                                                         pinfo.halo_extents, desc.halo_periods.data(), i, stream));
                break;
            case 1:
                CHECK_CUDECOMP_EXIT(cudecompUpdateHalosY(handle, grid_desc, data_d, work_d, get_cudecomp_datatype(real_t(0)),
                                                         pinfo.halo_extents, desc.halo_periods.data(), i, stream));
                break;
            case 2:
                CHECK_CUDECOMP_EXIT(cudecompUpdateHalosZ(handle, grid_desc, data_d, work_d, get_cudecomp_datatype(real_t(0)),
                                                         pinfo.halo_extents, desc.halo_periods.data(), i, stream));
                break;
            }
        }

        // Cleaning up the things that allocated memory
        CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));
    };

    template void halo_exchange<float>(cudecompHandle_t handle, haloDescriptor_t desc,
                                       cudaStream_t stream, void **buffers);
    template void halo_exchange<double>(cudecompHandle_t handle, haloDescriptor_t desc,
                                        cudaStream_t stream, void **buffers);
}
