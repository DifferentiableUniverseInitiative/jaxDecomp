#ifndef _JAX_DECOMP_FFT_H_
#define _JAX_DECOMP_FFT_H_

#include "checks.h"
#include "logger.hpp"
#include <array>
#include <cmath> // has to be included before cuda/std/complex
#include <cstddef>
#include <cstdio>
#include <cuda/std/complex>
#include <cudecomp.h>
#include <cufftXt.h>
#include <mpi.h>

static bool get_double_precision(float) { return false; }
static bool get_double_precision(double) { return true; }
static cufftType get_cufft_type_r2c(double) { return CUFFT_D2Z; }
static cufftType get_cufft_type_r2c(float) { return CUFFT_R2C; }
static cufftType get_cufft_type_c2r(double) { return CUFFT_Z2D; }
static cufftType get_cufft_type_c2r(float) { return CUFFT_C2R; }
static cufftType get_cufft_type_c2c(double) { return CUFFT_Z2Z; }
static cufftType get_cufft_type_c2c(float) { return CUFFT_C2C; }
static cudecompDataType_t get_cudecomp_datatype(cuda::std::complex<float>) { return CUDECOMP_FLOAT_COMPLEX; }
static cudecompDataType_t get_cudecomp_datatype(cuda::std::complex<double>) { return CUDECOMP_DOUBLE_COMPLEX; }
namespace jaxdecomp {

enum Decomposition { slab_XY = 0, slab_YZ = 1, pencil = 2, no_decomp = 3 };

static Decomposition GetDecomposition(const int pdims[2]) {
  if (pdims[0] == 1 && pdims[1] > 1) {
    return Decomposition::slab_XY;
  } else if (pdims[0] > 1 && pdims[1] == 1) {
    return Decomposition::slab_YZ;
  } else if (pdims[0] > 1 && pdims[1] > 1) {
    return Decomposition::pencil;
  }
  return Decomposition::no_decomp;
}

// fftDescriptor hash should be triavially computable
// because it contains only bools and integers
class fftDescriptor {
public:
  bool adjoint = false;
  bool contiguous = true;
  bool forward = true; ///< forward or backward pass
  // fft_is_forward_pass and forwad are used for the Execution but not for the
  // hash This way IFFT and FFT have the same plans when operating with the same
  // grid and pdims
  int32_t gdims[3]; ///< dimensions of global data grid
  // Decomposition type is used in order to allow reusing plans
  // from the XY and XZ forward pass for ZY and YZ backward pass respectively
  Decomposition decomposition = Decomposition::no_decomp; ///< decomposition type
  bool double_precision = false;
  cudecompGridDescConfig_t config; // Descriptor for the grid

  // To make it trivially copyable
  fftDescriptor() = default;
  fftDescriptor(const fftDescriptor& other) = default;
  fftDescriptor& operator=(const fftDescriptor& other) = default;
  // Create a compare operator to be used in the unordered_map (a hash is also
  // created in the bottom of the file)
  bool operator==(const fftDescriptor& other) const {
    if (double_precision != other.double_precision || gdims[0] != other.gdims[0] || gdims[1] != other.gdims[1] ||
        gdims[2] != other.gdims[2] || decomposition != other.decomposition || contiguous != other.contiguous) {
      return false;
    }
    return true;
  }
  ~fftDescriptor() = default;

  // Initialize the descriptor from the grid configuration
  // this is used for subsequent ffts to find the Executor that was already
  // defined
  fftDescriptor(cudecompGridDescConfig_t& config, const bool& double_precision, const bool& iForward,
                const bool& iAdjoint, const bool& iContiguous, const Decomposition& iDecomposition)
      : double_precision(double_precision), config(config), forward(iForward), contiguous(iContiguous),
        adjoint(iAdjoint), decomposition(iDecomposition) {
    gdims[0] = config.gdims[0];
    gdims[1] = config.gdims[1];
    gdims[2] = config.gdims[2];
    this->config.transpose_axis_contiguous[0] = iContiguous;
    this->config.transpose_axis_contiguous[1] = iContiguous;
    this->config.transpose_axis_contiguous[2] = iContiguous;
  }
};

template <typename real_t> class FourierExecutor {

  using complex_t = cuda::std::complex<real_t>;
  // Allow the Manager to access the private members in order to destroy the
  // GridDesc
  friend class GridDescriptorManager;

public:
  FourierExecutor() : m_Tracer("JAXDECOMP") {}
  ~FourierExecutor();

  HRESULT Initialize(cudecompHandle_t handle, size_t& work_size, fftDescriptor& fft_descriptor);

  HRESULT forward(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream, void** buffers);

  HRESULT backward(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream, void** buffers);

private:
  AsyncLogger m_Tracer;
  // GridDesc exists in GridConfig, but I rather store this way because of the C
  // struct definition that cuDecomp team chose to do
  // typedef struct cudecompGridDesc* cudecompGridDesc_t;
  // This produces a warning of incomplete type in the forward declaration and
  // To avoid this warning and having to include internal/common.h (which is not
  // C++20 compliant) I chose to store the cudecompGridDescConfig_t

  cudecompGridDesc_t m_GridConfig;
  cudecompGridDescConfig_t m_GridDescConfig;
  cudecompPencilInfo_t m_XPencilInfo;
  cudecompPencilInfo_t m_YPencilInfo;
  cudecompPencilInfo_t m_ZPencilInfo;
  // For the sake of expressive code, plans have the name of their corresponding
  // goal Instead of reusing pencils plans for slabs, or even ZY to YZ we store
  // properly named plans

  // For Pencils
  cufftHandle m_Plan_c2c_x;
  cufftHandle m_Plan_c2c_y;
  cufftHandle m_Plan_c2c_z;
  // For Slabs XY  FFT (Y) FFT(XZ) but JAX redifines the axes to YZX as X pencil for cudecomp
  // so in the end it is FFT(X) FFT(YZ)
  // For Slabs XZ FFT (X) FFT(YZ)
  cufftHandle m_Plan_c2c_yz;
  cufftHandle m_Plan_c2c_xy;
  // work size
  int64_t m_WorkSize;

  // Internal functions
  HRESULT InitializePencils(int64_t& work_size, fftDescriptor& fft_descriptor);

  HRESULT InitializeSlabXY(int64_t& work_size, fftDescriptor& fft_descriptor);

  HRESULT InitializeSlabYZ(int64_t& work_size, fftDescriptor& fft_descriptor);

  HRESULT forwardXY(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream, complex_t* input,
                    complex_t* output, complex_t* work_buffer);

  HRESULT backwardXY(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream, complex_t* input,
                     complex_t* output, complex_t* work_buffer);

  HRESULT forwardYZ(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream, complex_t* input,
                    complex_t* output, complex_t* work_buffer);

  HRESULT backwardYZ(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream, complex_t* input,
                     complex_t* output, complex_t* work_buffer);

  HRESULT forwardPencil(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream, complex_t* input,
                        complex_t* output, complex_t* work_buffer);

  HRESULT backwardPencil(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream, complex_t* input,
                         complex_t* output, complex_t* work_buffer);

  HRESULT clearPlans();
};

} // namespace jaxdecomp

namespace std {
template <> struct hash<jaxdecomp::fftDescriptor> {
  std::size_t operator()(const jaxdecomp::fftDescriptor& descriptor) const {
    // Only hash The double precision and the gdims and pdims
    // If adjoint is changed then the plan should be the same
    // adjoint is to be used to execute the backward plan
    size_t hash = std::hash<double>()(descriptor.double_precision) ^ std::hash<int>()(descriptor.gdims[0]) ^
                  std::hash<int>()(descriptor.gdims[1]) ^ std::hash<int>()(descriptor.gdims[2]) ^
                  std::hash<int>()(descriptor.decomposition) ^ std::hash<bool>()(descriptor.contiguous);
    return hash;
  }
};
} // namespace std

#endif
