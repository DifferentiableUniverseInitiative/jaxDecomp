#ifndef _JAX_DECOMP_FFT_H_
#define _JAX_DECOMP_FFT_H_

#include <cudecomp.h>
#include <cufftXt.h>

namespace jaxdecomp
{

    typedef struct
    {
        bool double_precision = false;
        bool slab_xy = false;
        bool slab_yz = false;
        cufftHandle cufft_plan_c2c_x;
        cufftHandle cufft_plan_c2c_y;
        cufftHandle cufft_plan_c2c_z;
        bool forward = true;             // Whether to compute a forward or backward fft
        bool adjoint = false;            // Whether to compute a forward or backward fft
        cudecompGridDescConfig_t config; // Descriptor for the grid
    } fftDescriptor_t;

    /**
     * @brief Get the fft3d descriptor object
     *
     * @tparam real_t  Type of the transform, float or double
     * @param handle cuDecomp handle
     * @param config Grid configuration
     * @param forward Whether to run a forward or reverse transform
     * @return std::pair<int64_t, fftDescriptor_t>
     */
    template <typename real_t>
    std::pair<int64_t, fftDescriptor_t> get_fft3d_descriptor(cudecompHandle_t handle,
                                                             cudecompGridDescConfig_t config,
                                                             bool forward, bool adjoint);

    /**
     * @brief Compute distributed 3D FFT based on pencil decomposition
     *
     * @tparam real_t Type of the transform, float or double
     * @param handle cuDecomp handle
     * @param config Grid configuration
     * @param buffers Input and output device buffers.
     */
    template <typename real_t>
    void fft3d(cudecompHandle_t handle,
               fftDescriptor_t desc,
               cudaStream_t stream,
               void **buffers);
};

#endif
