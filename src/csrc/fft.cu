#include "checks.h"
#include "fft.h"
#include "logger.hpp"
#include "perfostep.hpp"
#include <cstddef>
#include <cudecomp.h>
#include <cufft.h>
#include <ios>
#include <ostream>
// Inside your fftDescriptor class or in a separate header file

namespace jaxdecomp {

template <typename real_t>
HRESULT FourierExecutor<real_t>::Initialize(cudecompHandle_t handle, size_t& work_size, fftDescriptor& fft_descriptor) {
  Perfostep profiler;
  profiler.Start("CreateGridDesc");

  m_GridDescConfig = fft_descriptor.config;

  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &m_GridConfig, &m_GridDescConfig, nullptr));
  // Get x-pencil information (complex)
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, m_GridConfig, &m_XPencilInfo, 0, nullptr, nullptr));
  // Get y-pencil information (complex)
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, m_GridConfig, &m_YPencilInfo, 1, nullptr, nullptr));
  // Get z-pencil information (complex)
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, m_GridConfig, &m_ZPencilInfo, 2, nullptr, nullptr));
  // Get workspace size
  int64_t num_elements_work_c;
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, m_GridConfig, &num_elements_work_c));

  profiler.Stop();
  // Set up the FFT plan

  int64_t work_sz_cufft;

  HRESULT hr(E_FAIL);
  switch (fft_descriptor.decomposition) {
  case Decomposition::slab_XY:
    profiler.Start("InitializeSlabXY");
    hr = InitializeSlabXY(work_sz_cufft, fft_descriptor);
    break;
  case Decomposition::slab_YZ:
    profiler.Start("InitializeSlabYZ");
    hr = InitializeSlabYZ(work_sz_cufft, fft_descriptor);
    break;
  case Decomposition::pencil:
    profiler.Start("InitializePencils");
    hr = InitializePencils(work_sz_cufft, fft_descriptor);
    break;
  case Decomposition::no_decomp: hr = E_FAIL; break;
  }
  profiler.Stop();

  // Note: we can also allocat the workspace here rather than
  // requesting XLA to do it so it can be cleaned it up easily in the finalize
  // step CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, &m_Work,
  // num_elements_work_c,
  //                                    get_cudecomp_datatype(complex_t(0)),
  //                                    nullptr, nullptr, 0));
  if (SUCCEEDED(hr)) {
    int64_t work_sz_decomp;
    work_sz_decomp = 2 * num_elements_work_c * sizeof(real_t);
    m_WorkSize = std::max(work_sz_cufft, work_sz_decomp);

    work_size = m_WorkSize;
  }

  return hr;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::InitializePencils(int64_t& work_size, fftDescriptor& fft_descriptor) {

  int& gx = fft_descriptor.config.gdims[0]; // take reference to avoid copying
  int& gy = fft_descriptor.config.gdims[1];
  int& gz = fft_descriptor.config.gdims[2];
  // Create the plans
  CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_x));
  CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_y));
  CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_z));
  // Set the auto allocation
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_x, 0));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_y, 0));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_z, 0));

  // TODO(wassim) : non-contiguous plans need to be created aswell
  // The work size
  size_t work_sz_c2c_x, work_sz_c2c_y, work_sz_c2c_z;
  // The X plan
  CHECK_CUFFT_EXIT(cufftMakePlan1d(m_Plan_c2c_x, gx, get_cufft_type_c2c(real_t(0)),
                                   m_XPencilInfo.shape[1] * m_XPencilInfo.shape[2], &work_sz_c2c_x));

  if (fft_descriptor.contiguous) {

    // The Y plan
    CHECK_CUFFT_EXIT(cufftMakePlan1d(m_Plan_c2c_y, gy, get_cufft_type_c2c(real_t(0)),
                                     m_YPencilInfo.shape[1] * m_YPencilInfo.shape[2], &work_sz_c2c_y));

    // The Z plan
    CHECK_CUFFT_EXIT(cufftMakePlan1d(m_Plan_c2c_z, gz, get_cufft_type_c2c(real_t(0)),
                                     m_ZPencilInfo.shape[1] * m_ZPencilInfo.shape[2], &work_sz_c2c_z));
  } else {
    // The Y plan
    CHECK_CUFFT_EXIT(cufftMakePlanMany(m_Plan_c2c_y, /*rank*/ 1, /*n (size)*/ &gy /* inembed */, &gy,
                                       /*stride*/ m_YPencilInfo.shape[0], /*idist*/ 1,
                                       /*onembed*/ &gy,
                                       /*ostride*/ m_YPencilInfo.shape[0], /*odist*/ 1,
                                       /*type*/ get_cufft_type_c2c(real_t(0)),
                                       /*batchsize */ m_YPencilInfo.shape[0], &work_sz_c2c_y));

    // The Z plan
    CHECK_CUFFT_EXIT(
        cufftMakePlanMany(m_Plan_c2c_z, 1, &gz /* unused */, &gz, m_ZPencilInfo.shape[0] * m_ZPencilInfo.shape[1], 1,
                          &gz, m_ZPencilInfo.shape[0] * m_ZPencilInfo.shape[1], 1, get_cufft_type_c2c(real_t(0)),
                          m_ZPencilInfo.shape[0] * m_ZPencilInfo.shape[1], &work_sz_c2c_z));
  }

  work_size = std::max(work_sz_c2c_x, std::max(work_sz_c2c_y, work_sz_c2c_z));
  return work_size > 0 ? S_OK : E_FAIL;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::InitializeSlabXY(int64_t& work_size, fftDescriptor& fft_descriptor) {
  int& gx = fft_descriptor.config.gdims[0]; // take reference to avoid copying
  int& gy = fft_descriptor.config.gdims[1];
  int& gz = fft_descriptor.config.gdims[2];
  // The XY plan
  // The ZY plan
  // Get the plan sizes
  size_t work_size_x, work_size_yz, work_size_y, work_size_z;

  if (fft_descriptor.contiguous) {

    CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_x));
    CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_x, 0));

    CHECK_CUFFT_EXIT(cufftMakePlan1d(m_Plan_c2c_x, gx, get_cufft_type_c2c(real_t(0)),
                                     m_XPencilInfo.shape[1] * m_XPencilInfo.shape[2], &work_size_x));

    CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_yz));
    CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_yz, 0));
    // make the second plan YZ
    std::array<int, 2> n{gz, gy};
    CHECK_CUFFT_EXIT(cufftMakePlanMany(m_Plan_c2c_yz, 2, n.data(), nullptr, 1,
                                       m_YPencilInfo.shape[0] * m_YPencilInfo.shape[1], nullptr, 1,
                                       m_YPencilInfo.shape[0] * m_YPencilInfo.shape[1], get_cufft_type_c2c(real_t(0)),
                                       m_YPencilInfo.shape[2], &work_size_yz));
  } else {

    CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_xy));
    CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_xy, 0));
    std::array<int, 2> n{gy, gx};
    CHECK_CUFFT_EXIT(cufftMakePlanMany(m_Plan_c2c_xy, 2, n.data(), nullptr, 1,
                                       m_XPencilInfo.shape[0] * m_XPencilInfo.shape[1], nullptr, 1,
                                       m_XPencilInfo.shape[0] * m_XPencilInfo.shape[1], get_cufft_type_c2c(real_t(0)),
                                       m_XPencilInfo.shape[2], &work_size_x));

    CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_z));
    CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_z, 0));

    CHECK_CUFFT_EXIT(
        cufftMakePlanMany(m_Plan_c2c_z, 1, &gz /* unused */, &gz, m_ZPencilInfo.shape[0] * m_ZPencilInfo.shape[1], 1,
                          &gz, m_ZPencilInfo.shape[0] * m_ZPencilInfo.shape[1], 1, get_cufft_type_c2c(real_t(0)),
                          m_ZPencilInfo.shape[0] * m_ZPencilInfo.shape[1], &work_size_z));

    work_size_yz = std::max(work_size_y, work_size_z);
  }

  work_size = std::max(work_size_x, work_size_yz);

  return work_size > 0 ? S_OK : E_FAIL;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::InitializeSlabYZ(int64_t& work_size, fftDescriptor& fft_descriptor) {

  int& gx = fft_descriptor.config.gdims[0]; // take reference to avoid copying
  int& gy = fft_descriptor.config.gdims[1];
  int& gz = fft_descriptor.config.gdims[2];
  // The XY plan
  CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_x));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_x, 0));
  // The ZY plan
  // Get the plan sizes
  size_t work_size_x, work_size_yz, work_size_y, work_size_z;

  CHECK_CUFFT_EXIT(cufftMakePlan1d(m_Plan_c2c_x, gx, get_cufft_type_c2c(real_t(0)),
                                   m_XPencilInfo.shape[1] * m_XPencilInfo.shape[2], &work_size_x));

  if (fft_descriptor.contiguous) {

    CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_yz));
    CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_yz, 0));
    // make the second plan YZ
    std::array<int, 2> n{gz, gy};
    CHECK_CUFFT_EXIT(cufftMakePlanMany(m_Plan_c2c_yz, 2, n.data(), nullptr, 1,
                                       m_YPencilInfo.shape[0] * m_YPencilInfo.shape[1], nullptr, 1,
                                       m_YPencilInfo.shape[0] * m_YPencilInfo.shape[1], get_cufft_type_c2c(real_t(0)),
                                       m_YPencilInfo.shape[2], &work_size_yz));
  } else {

    CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_y));
    CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_z));

    CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_y, 0));
    CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_z, 0));

    // Both Y and Z are non contiguous create two plans
    CHECK_CUFFT_EXIT(cufftMakePlanMany(m_Plan_c2c_y, 1, &gy /* unused */, &gy, m_YPencilInfo.shape[0], 1, &gy,
                                       m_YPencilInfo.shape[0], 1, get_cufft_type_c2c(real_t(0)), m_YPencilInfo.shape[0],
                                       &work_size_y));
    CHECK_CUFFT_EXIT(
        cufftMakePlanMany(m_Plan_c2c_z, 1, &gz /* unused */, &gz, m_ZPencilInfo.shape[0] * m_ZPencilInfo.shape[1], 1,
                          &gz, m_ZPencilInfo.shape[0] * m_ZPencilInfo.shape[1], 1, get_cufft_type_c2c(real_t(0)),
                          m_ZPencilInfo.shape[0] * m_ZPencilInfo.shape[1], &work_size_z));
    work_size_yz = std::max(work_size_y, work_size_z);
  }

  work_size = std::max(work_size_x, work_size_yz);

  return work_size > 0 ? S_OK : E_FAIL;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::forward(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream,
                                         void** buffers) {
  Perfostep profiler;
  profiler.Start("forward");

  HRESULT hr(E_FAIL);
  void* data_d = buffers[0];
  void* work_d = buffers[1];
  complex_t* data_c_d = static_cast<complex_t*>(data_d);
  complex_t* input = data_c_d;
  complex_t* output = data_c_d;

  // Assign cuFFT work area and current XLA stream
  complex_t* work_c_d = static_cast<complex_t*>(work_d);
  switch (desc.decomposition) {
  case Decomposition::slab_XY:
    profiler.Start("forwardXY");
    hr = forwardXY(handle, desc, stream, input, output, work_c_d);
    break;
  case Decomposition::slab_YZ:
    profiler.Start("forwardYZ");
    hr = forwardYZ(handle, desc, stream, input, output, work_c_d);
    break;
  case Decomposition::pencil:
    profiler.Start("forwardPencil");
    hr = forwardPencil(handle, desc, stream, input, output, work_c_d);
    break;
  case Decomposition::no_decomp: hr = E_FAIL;
  }
  profiler.Stop();
  return hr;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::backward(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream,
                                          void** buffers) {
  Perfostep profiler;
  profiler.Start("backward");

  HRESULT hr(E_FAIL);
  void* data_d = buffers[0];
  void* work_d = buffers[1];
  complex_t* data_c_d = static_cast<complex_t*>(data_d);
  complex_t* input = data_c_d;
  complex_t* output = data_c_d;
  // Assign cuFFT work area and current XLA stream
  complex_t* work_c_d = static_cast<complex_t*>(work_d);
  switch (desc.decomposition) {
  case Decomposition::slab_XY:
    profiler.Start("backwardXY");
    hr = backwardXY(handle, desc, stream, input, output, work_c_d);
    break;
  case Decomposition::slab_YZ:
    profiler.Start("backwardYZ");
    hr = backwardYZ(handle, desc, stream, input, output, work_c_d);
    break;
  case Decomposition::pencil:
    profiler.Start("backwardPencil");
    hr = backwardPencil(handle, desc, stream, input, output, work_c_d);
    break;
  case Decomposition::no_decomp: hr = E_FAIL;
  }
  profiler.Stop();
  return hr;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::forwardXY(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream,
                                           complex_t* input, complex_t* output, complex_t* work_d) {

  const int DIRECTION = desc.adjoint ? CUFFT_INVERSE : CUFFT_FORWARD;

  // FFT on the first slab
  if (desc.contiguous) {
    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_yz, stream));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_yz, work_d));

    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_yz, input, output, DIRECTION));
    // Tranpose Y to X but it actually X to Z
    CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, m_GridConfig, output, output, work_d,
                                              get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
                                              stream));
    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_x, stream));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_x, work_d));
    // FFT on the second slab
    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_x, output, output, DIRECTION));

  } else {
    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_xy, stream));
    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_z, stream));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_xy, work_d));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_z, work_d));

    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_xy, input, output, DIRECTION));
    // Tranpose X to Y
    CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, m_GridConfig, output, output, work_d,
                                              get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
                                              stream));
    // Tranpose Y to z
    CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, m_GridConfig, output, output, work_d,
                                              get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
                                              stream));
    // FFT on the second slab
    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_z, output, output, DIRECTION));
  }
  return S_OK;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::backwardXY(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream,
                                            complex_t* input, complex_t* output, complex_t* work_d) {

  const int DIRECTION = desc.adjoint ? CUFFT_FORWARD : CUFFT_INVERSE;
  // IFFT on the second slab
  if (desc.contiguous) {

    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_x, stream));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_x, work_d));
    // FFT on the first slab
    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_x, input, output, DIRECTION));
    // Tranpose X to Y but it actually Z to X
    CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, m_GridConfig, output, output, work_d,
                                              get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
                                              stream));
    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_yz, stream));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_yz, work_d));
    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_yz, output, output, DIRECTION));
  } else {

    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_xy, stream));
    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_z, stream));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_xy, work_d));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_z, work_d));

    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_z, output, output, DIRECTION));
    // Tranpose X to Y
    CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, m_GridConfig, output, output, work_d,
                                              get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
                                              stream));
    // Tranpose Y to z
    CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, m_GridConfig, output, output, work_d,
                                              get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
                                              stream));
    // FFT on the second slab
    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_xy, input, output, DIRECTION));
  }

  return S_OK;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::forwardYZ(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream,
                                           complex_t* input, complex_t* output, complex_t* work_d) {

  const int DIRECTION = desc.adjoint ? CUFFT_INVERSE : CUFFT_FORWARD;

  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_x, stream));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_x, work_d));
  // FFT on the first slab
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_x, input, output, DIRECTION));
  // Tranpose X to Y
  CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, m_GridConfig, input, output, work_d,
                                            get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
                                            stream));
  // FFT on the second slab
  if (desc.contiguous) {
    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_yz, stream));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_yz, work_d));
    // 2D FFT with the first axis contiguous
    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_yz, output, output, DIRECTION));
  } else {
    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_y, stream));
    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_z, stream));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_y, work_d));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_z, work_d));
    // 1D strided FFT on the second axis (Y)
    // 1D strided FFT on the third axis (Z)
    int32_t z_stride = m_YPencilInfo.shape[0] * m_YPencilInfo.shape[1];
    for (int i = 0; i < m_YPencilInfo.shape[2]; i++) {
      CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_y, output + i * z_stride, output + i * z_stride, DIRECTION));
    }
    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_z, output, output, DIRECTION));
  }

  // Extra Y to Z transpose to give back a Z pencil to the user

  // CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, m_GridConfig, output, output, work_d,
  //                                           get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
  //                                           stream));

  return S_OK;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::backwardYZ(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream,
                                            complex_t* input, complex_t* output, complex_t* work_d) {

  const int DIRECTION = desc.adjoint ? CUFFT_FORWARD : CUFFT_INVERSE;

  // Input is Z pencil tranposed it back to Y pencil
  // CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, m_GridConfig, input, output, work_d,
  //                                         get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
  //                                         stream));

  // FFT on the first slab
  if (desc.contiguous) {
    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_yz, stream));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_yz, work_d));
    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_yz, input, output, DIRECTION));
  } else {
    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_y, stream));
    CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_z, stream));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_y, work_d));
    CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_z, work_d));
    // 1D strided IFFT on the second axis (Y)
    // 1D strided IFFT on the third axis (Z)
    int32_t z_stride = m_YPencilInfo.shape[0] * m_YPencilInfo.shape[1];
    for (int i = 0; i < m_YPencilInfo.shape[2]; i++) {
      CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_y, output + i * z_stride, output + i * z_stride, DIRECTION));
    }
    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_z, output, output, DIRECTION));
  }
  // Tranpose Y to X
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, m_GridConfig, input, output, work_d,
                                            get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
                                            stream));
  // IFFT the first axis (x)
  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_x, stream));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_x, work_d));
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_x, output, output, DIRECTION));

  return S_OK;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::forwardPencil(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream,
                                               complex_t* input, complex_t* output, complex_t* work_d) {

  const int DIRECTION = desc.adjoint ? CUFFT_INVERSE : CUFFT_FORWARD;

  // Set Stream and work area
  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_x, stream));
  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_y, stream));
  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_z, stream));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_x, work_d));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_y, work_d));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_z, work_d));
  // FFT on the first pencil
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_x, input, output, DIRECTION));
  // Tranpose X to Y
  // before
  CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, m_GridConfig, output, output, work_d,
                                            get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
                                            stream));
  if (desc.contiguous) {
    // FFT on the second pencil
    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_y, output, output, DIRECTION));
  } else {
    // FFT on the second pencil
    int32_t z_stride = m_YPencilInfo.shape[0] * m_YPencilInfo.shape[1];
    for (int i = 0; i < m_YPencilInfo.shape[2]; i++) {
      CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_y, output + i * z_stride, output + i * z_stride, DIRECTION));
    }
  }
  // Tranpose Y to Z
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, m_GridConfig, output, output, work_d,
                                            get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
                                            stream));
  // FFT on the third pencil
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_z, output, output, DIRECTION));
  return S_OK;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::backwardPencil(cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream,
                                                complex_t* input, complex_t* output, complex_t* work_d) {

  const int DIRECTION = desc.adjoint ? CUFFT_FORWARD : CUFFT_INVERSE;

  // Set Stream and work area
  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_x, stream));
  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_y, stream));
  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_z, stream));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_x, work_d));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_y, work_d));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_z, work_d));

  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_z, input, output, DIRECTION));
  // Tranpose Z to Y
  CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, m_GridConfig, output, output, work_d,
                                            get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
                                            stream));
  // FFT on the second pencil
  if (desc.contiguous) {
    CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_y, output, output, DIRECTION));
  } else {
    int32_t z_stride = m_YPencilInfo.shape[0] * m_YPencilInfo.shape[1];
    for (int i = 0; i < m_YPencilInfo.shape[2]; i++) {
      CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_y, output + i * z_stride, output + i * z_stride, DIRECTION));
    }
  }
  // Tranpose Y to X
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, m_GridConfig, output, output, work_d,
                                            get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, nullptr, nullptr,
                                            stream));
  // FFT on the third pencil
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_x, output, output, DIRECTION));

  return S_OK;
}

template <typename real_t> HRESULT FourierExecutor<real_t>::clearPlans() {
  // Destroy the plans
  switch (GetDecomposition(m_GridDescConfig.pdims)) {
  case Decomposition::slab_XY:
  case Decomposition::slab_YZ: cufftDestroy(m_Plan_c2c_yz); cufftDestroy(m_Plan_c2c_xy);
  case Decomposition::pencil:
    cufftDestroy(m_Plan_c2c_x);
    cufftDestroy(m_Plan_c2c_y);
    cufftDestroy(m_Plan_c2c_z);
    break;
  case Decomposition::no_decomp: break;
  }

  return S_OK;
}

template <typename real_t> FourierExecutor<real_t>::~FourierExecutor() {

  // Plans are destroyed by the manager
  // DO NOT DESTROY THE PLANS HERE
}

template class FourierExecutor<double>;
template class FourierExecutor<float>;

} // namespace jaxdecomp
