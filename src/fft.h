#ifndef _JAX_DECOMP_FFT_H_
#define _JAX_DECOMP_FFT_H_

#include <cudecomp.h>

namespace jaxdecomp {

template<typename real_t> void fft3d(cudecompHandle_t handle,
              cudecompGridDescConfig_t config,
              void* data_d,
              bool forward=true,
              bool r2c=false);
};



#endif