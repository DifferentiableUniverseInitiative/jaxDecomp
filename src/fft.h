#ifndef _JAX_DECOMP_FFT_H_
#define _JAX_DECOMP_FFT_H_

#include <cudecomp.h>

namespace jaxdecomp {

/**
 * @brief Compute distributed 3D FFT based on pencil decomposition
 * 
 * @tparam real_t Type of the transform, float or double
 * @param handle cuDecomp handle
 * @param config Grid configuration
 * @param buffers Input and output device buffers. 
 * Note, complex FFTs are done in-place (so need one buffer), real FFTs are done out of place (so need 2 buffers)
 * @param forward Run the forward FFT if true, reverse FFT if false
 * @param r2c Compute a real to complex transform if true
 */
template<typename real_t> void fft3d(cudecompHandle_t handle,
              cudecompGridDescConfig_t config,
              void** buffers,
              bool forward=true,
              bool r2c=false);
};



#endif