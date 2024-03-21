#include "logger.hpp"
#include "checks.h"
#include "fft.h"
#include <cstddef>
#include <cudecomp.h>
#include <cufft.h>
#include <ios>
#include <ostream>
// Inside your fftDescriptor class or in a separate header file

namespace jaxdecomp {

template <typename real_t>
HRESULT FourierExecutor<real_t>::Initialize(cudecompHandle_t handle,
                                            cudecompGridDescConfig_t config,
                                            size_t &work_size,
                                            fftDescriptor &fft_descriptor) {
  m_GridDescConfig = config;

  CHECK_CUDECOMP_EXIT(
      cudecompGridDescCreate(handle, &m_GridConfig, &config, nullptr));
  // Get x-pencil information (complex)
  cudecompPencilInfo_t pinfo_x_c;
  CHECK_CUDECOMP_EXIT(
      cudecompGetPencilInfo(handle, m_GridConfig, &pinfo_x_c, 0, nullptr));
  // Get y-pencil information (complex)
  cudecompPencilInfo_t pinfo_y_c;
  CHECK_CUDECOMP_EXIT(
      cudecompGetPencilInfo(handle, m_GridConfig, &pinfo_y_c, 1, nullptr));
  // Get z-pencil information (complex)
  cudecompPencilInfo_t pinfo_z_c;
  CHECK_CUDECOMP_EXIT(
      cudecompGetPencilInfo(handle, m_GridConfig, &pinfo_z_c, 2, nullptr));
  // Get workspace size
  int64_t num_elements_work_c;
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, m_GridConfig,
                                                        &num_elements_work_c));

  // Set up the FFT plan

  // Note: 0 is the Z axis here due to the fact the cudecomp is column-major
  // (Fortran indexing)

  // Simple code is better
  // No need to handle mixed contiguous and non-contiguous cases
  bool is_contiguous = config.transpose_axis_contiguous[0] ||
                       config.transpose_axis_contiguous[1] ||
                       config.transpose_axis_contiguous[2];

  is_contiguous = true; // Force only contiguous case for now because I need to
  // review the non-contiguous case
  // Expressive code : call properly named functions to handle the different
  // cases
  int64_t work_sz_cufft;

  HRESULT hr(E_FAIL);
  switch (GetDecomposition(config.pdims)) {
  case Decomposition::slab_XY:
    hr = InitializeSlabXY(config, pinfo_x_c, pinfo_y_c, pinfo_z_c,
                          work_sz_cufft, is_contiguous);
    break;
  case Decomposition::slab_YZ:
    hr = InitializeSlabYZ(config, pinfo_x_c, pinfo_y_c, pinfo_z_c,
                          work_sz_cufft, is_contiguous);
    break;
  case Decomposition::pencil:
    hr = InitializePencils(config, pinfo_x_c, pinfo_y_c, pinfo_z_c,
                           work_sz_cufft, is_contiguous);
    break;
  case Decomposition::unknown:
    hr = E_FAIL;
    break;
  }

  // Note: we can also allocat the workspace here rather than
  // requesting XLA to do it so it can be cleaned it up easily in the finalize
  // step CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, &m_Work,
  // num_elements_work_c,
  //                                    get_cudecomp_datatype(complex_t(0)),
  //                                    nullptr, nullptr, 0));
  if (SUCCEEDED(hr)) {

    fft_descriptor.double_precision = get_double_precision(real_t(0));
    fft_descriptor.config = config;
    fft_descriptor.gdims[0] = config.gdims[0];
    fft_descriptor.gdims[1] = config.gdims[1];
    fft_descriptor.gdims[2] = config.gdims[2];
    fft_descriptor.decomposition = GetDecomposition(config.pdims);
  }

  int64_t work_sz_decomp;
  work_sz_decomp = 2 * num_elements_work_c * sizeof(real_t);
  m_WorkSize = std::max(work_sz_cufft, work_sz_decomp);

  work_size = m_WorkSize;

  return hr;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::InitializePencils(
    cudecompGridDescConfig_t &iGridConfig, cudecompPencilInfo_t &x_pencil_info,
    cudecompPencilInfo_t &y_pencil_info, cudecompPencilInfo_t &z_pencil_info,
    int64_t &work_size, const bool &is_contiguous) {

  int &gx = iGridConfig.gdims[0]; // take reference to avoid copying
  int &gy = iGridConfig.gdims[1];
  int &gz = iGridConfig.gdims[2];
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
  CHECK_CUFFT_EXIT(cufftMakePlan1d(
      m_Plan_c2c_x, gx, get_cufft_type_c2c(real_t(0)),
      x_pencil_info.shape[1] * x_pencil_info.shape[2], &work_sz_c2c_x));

  // The Y plan
  CHECK_CUFFT_EXIT(cufftMakePlan1d(
      m_Plan_c2c_y, gy, get_cufft_type_c2c(real_t(0)),
      y_pencil_info.shape[1] * y_pencil_info.shape[2], &work_sz_c2c_y));

  // The Z plan
  CHECK_CUFFT_EXIT(cufftMakePlan1d(
      m_Plan_c2c_z, gz, get_cufft_type_c2c(real_t(0)),
      z_pencil_info.shape[1] * z_pencil_info.shape[2], &work_sz_c2c_z));

  work_size = std::max(work_sz_c2c_x, std::max(work_sz_c2c_y, work_sz_c2c_z));
  return work_size > 0 ? S_OK : E_FAIL;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::InitializeSlabXY(
    cudecompGridDescConfig_t &iGridConfig, cudecompPencilInfo_t &x_pencil_info,
    cudecompPencilInfo_t &y_pencil_info, cudecompPencilInfo_t &z_pencil_info,
    int64_t &work_size, const bool &is_contiguous) {
  int &gx = iGridConfig.gdims[0]; // take reference to avoid copying
  int &gy = iGridConfig.gdims[1];
  int &gz = iGridConfig.gdims[2];
  // The XY plan
  CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_xy));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_xy, 0));
  // The ZY plan
  CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_z));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_z, 0));
  // Get the plan sizes
  size_t work_size_xy, work_size_z;

  //  n[In] – Array of size rank, describing the size of each dimension, n[0]
  //  being the size of the outermost and n[rank-1]
  // in this case n is Y and X
  // (Side note: the first axis is always contiguous in cuDecomp)
  std::array<int, 2> y_x{gy, gx};

  CHECK_CUFFT_EXIT(cufftMakePlanMany(
      m_Plan_c2c_xy, 2, y_x.data(), nullptr, 1,
      x_pencil_info.shape[0] * x_pencil_info.shape[1], nullptr, 1,
      x_pencil_info.shape[0] * x_pencil_info.shape[1],
      get_cufft_type_c2c(real_t(0)), x_pencil_info.shape[2], &work_size_xy));

  if (is_contiguous) {

    // make the second plan
    CHECK_CUFFT_EXIT(cufftMakePlan1d(
        m_Plan_c2c_z, gz, get_cufft_type_c2c(real_t(0)),
        z_pencil_info.shape[1] * z_pencil_info.shape[2], &work_size_z));

  } else {

    // TODO(wassim) : I did not understand this yet
    //  Making the second non contiguous plans first for Z Y slab Y is not
    //  contiguous here
    CHECK_CUFFT_EXIT(cufftMakePlanMany(
        m_Plan_c2c_z, 1, &gz /* unused */, &gz,
        z_pencil_info.shape[0] * z_pencil_info.shape[1], 1, &gz,
        z_pencil_info.shape[0] * z_pencil_info.shape[1], 1,
        get_cufft_type_c2c(real_t(0)),
        z_pencil_info.shape[0] * z_pencil_info.shape[1], &work_size_z));
    // Another Batched many plan should be made here
  }

  work_size = std::max(work_size_xy, work_size_z);

  return work_size > 0 ? S_OK : E_FAIL;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::InitializeSlabYZ(
    cudecompGridDescConfig_t &iGridConfig, cudecompPencilInfo_t &x_pencil_info,
    cudecompPencilInfo_t &y_pencil_info, cudecompPencilInfo_t &z_pencil_info,
    int64_t &work_size, const bool &is_contiguous) {

  int &gx = iGridConfig.gdims[0]; // take reference to avoid copying
  int &gy = iGridConfig.gdims[1];
  int &gz = iGridConfig.gdims[2];
  // The XY plan
  CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_x));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_x, 0));
  // The ZY plan
  CHECK_CUFFT_EXIT(cufftCreate(&m_Plan_c2c_yz));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(m_Plan_c2c_yz, 0));
  // Get the plan sizes
  size_t work_size_x, work_size_yz;

  CHECK_CUFFT_EXIT(cufftMakePlan1d(
      m_Plan_c2c_x, gx, get_cufft_type_c2c(real_t(0)),
      x_pencil_info.shape[1] * x_pencil_info.shape[2], &work_size_x));

  if (is_contiguous) {

    // make the second plan YZ
    std::array<int, 2> n{gz, gy};
    CHECK_CUFFT_EXIT(cufftMakePlanMany(
        m_Plan_c2c_yz, 2, n.data(), nullptr, 1,
        y_pencil_info.shape[0] * y_pencil_info.shape[1], nullptr, 1,
        y_pencil_info.shape[0] * y_pencil_info.shape[1],
        get_cufft_type_c2c(real_t(0)), y_pencil_info.shape[2], &work_size_yz));

  } else {

    // TODO(wassim) : I did not understand this yet
    //  Making the second non contiguous plans first for Z Y slab Y is not
    //  contiguous here
    CHECK_CUFFT_EXIT(cufftMakePlanMany(
        m_Plan_c2c_y, 1, &gy /* unused */, &gy, y_pencil_info.shape[0], 1, &gy,
        y_pencil_info.shape[0], 1, get_cufft_type_c2c(real_t(0)),
        y_pencil_info.shape[0], &work_size_yz));
    // Another Batched many plan should be made here
  }

  work_size = std::max(work_size_x, work_size_yz);

  return work_size > 0 ? S_OK : E_FAIL;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::forward(cudecompHandle_t handle,
                                         fftDescriptor desc,
                                         cudaStream_t stream, void **buffers) {
  HRESULT hr(E_FAIL);
  void *data_d = buffers[0];
  void *work_d = buffers[1];
  complex_t *data_c_d = static_cast<complex_t *>(data_d);
  complex_t *input = data_c_d;
  complex_t *output = data_c_d;

  // Assign cuFFT work area and current XLA stream
  complex_t *work_c_d = static_cast<complex_t *>(work_d);
  switch (desc.decomposition) {
  case Decomposition::slab_XY:
    hr = forwardXY(handle, desc, stream, input, output, work_c_d);
    break;
  case Decomposition::slab_YZ:
    hr = forwardYZ(handle, desc, stream, input, output, work_c_d);
    break;
  case Decomposition::pencil:
    hr = forwardPencil(handle, desc, stream, input, output, work_c_d);
    break;
  case Decomposition::unknown:
    hr = E_FAIL;
  }
  return hr;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::backward(cudecompHandle_t handle,
                                          fftDescriptor desc,
                                          cudaStream_t stream, void **buffers) {
  HRESULT hr(E_FAIL);
  void *data_d = buffers[0];
  void *work_d = buffers[1];
  complex_t *data_c_d = static_cast<complex_t *>(data_d);
  complex_t *input = data_c_d;
  complex_t *output = data_c_d;
  // Assign cuFFT work area and current XLA stream
  complex_t *work_c_d = static_cast<complex_t *>(work_d);
  switch (desc.decomposition) {
  case Decomposition::slab_XY:
    hr = backwardXY(handle, desc, stream, input, output, work_c_d);
    break;
  case Decomposition::slab_YZ:
    hr = backwardYZ(handle, desc, stream, input, output, work_c_d);
    break;
  case Decomposition::pencil:
    hr = backwardPencil(handle, desc, stream, input, output, work_c_d);
    break;
  case Decomposition::unknown:
    hr = E_FAIL;
  }
  return hr;
}

template <typename real_t>
HRESULT
FourierExecutor<real_t>::forwardXY(cudecompHandle_t handle, fftDescriptor desc,
                                   cudaStream_t stream, complex_t *input,
                                   complex_t *output, complex_t *work_d) {

  const int DIRECTION = desc.adjoint ? CUFFT_INVERSE : CUFFT_FORWARD;

  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_xy, stream));
  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_z, stream));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_xy, work_d));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_z, work_d));
  // FFT on the first slab
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_xy, input, output, DIRECTION));
  // Tranpose X to Y
  CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(
      handle, m_GridConfig, output, output, work_d,
      get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));
  // Tranpose Y to Z
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(
      handle, m_GridConfig, output, output, work_d,
      get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));
  // FFT on the second slab
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_z, output, output, DIRECTION));

  return S_OK;
}

template <typename real_t>
HRESULT
FourierExecutor<real_t>::backwardXY(cudecompHandle_t handle, fftDescriptor desc,
                                    cudaStream_t stream, complex_t *input,
                                    complex_t *output, complex_t *work_d) {

  const int DIRECTION = desc.adjoint ? CUFFT_FORWARD : CUFFT_INVERSE;

  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_xy, stream));
  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_z, stream));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_xy, work_d));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_z, work_d));

  // FFT on the first slab
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_z, input, output, DIRECTION));
  // Tranpose Z to Y
  CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(
      handle, m_GridConfig, output, output, work_d,
      get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));
  // Tranpose Y to X
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(
      handle, m_GridConfig, output, output, work_d,
      get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));
  // IFFT on the second slab
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_xy, output, output, DIRECTION));

  return S_OK;
}

template <typename real_t>
HRESULT
FourierExecutor<real_t>::forwardYZ(cudecompHandle_t handle, fftDescriptor desc,
                                   cudaStream_t stream, complex_t *input,
                                   complex_t *output, complex_t *work_d) {

  const int DIRECTION = desc.adjoint ? CUFFT_INVERSE : CUFFT_FORWARD;

  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_x, stream));
  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_yz, stream));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_x, work_d));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_yz, work_d));
  // FFT on the first slab
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_x, input, output, DIRECTION));
  // Tranpose X to Y
  CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(
      handle, m_GridConfig, output, output, work_d,
      get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));
  // FFT on the second slab
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_yz, output, output, DIRECTION));

  return S_OK;
}

template <typename real_t>
HRESULT
FourierExecutor<real_t>::backwardYZ(cudecompHandle_t handle, fftDescriptor desc,
                                    cudaStream_t stream, complex_t *input,
                                    complex_t *output, complex_t *work_d) {

  const int DIRECTION = desc.adjoint ? CUFFT_FORWARD : CUFFT_INVERSE;

  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_x, stream));
  CHECK_CUFFT_EXIT(cufftSetStream(m_Plan_c2c_yz, stream));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_x, work_d));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(m_Plan_c2c_yz, work_d));

  // FFT on the first slab
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_yz, input, output, DIRECTION));
  // Tranpose Y to X
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(
      handle, m_GridConfig, output, output, work_d,
      get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));
  // IFFT on the second slab
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_x, output, output, DIRECTION));

  return S_OK;
}

// DEBUG ONLY ... I WARN YOU
template <typename real_t>
void FourierExecutor<real_t>::inspect_device_array(complex_t *data, int size,
                                                   cudaStream_t stream) {
  int rank;
  CHECK_MPI_EXIT(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  // Copy input to host
  const int local_size = 4 * 4 * size;
  complex_t *host = new complex_t[local_size];

  cudaMemcpyAsync(host, data, sizeof(complex_t) * local_size,
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "Flat print" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  for (int r = 0; r < 4; r++) {

    if (rank == r) // to force printing in order so I have less headache
      for (int i = 0; i < size * size * size; i++) {
        std::cout << "Rank[" << rank << "] Element [" << i
                  << "] : " << host[i].real() << " + " << host[i].imag() << "i"
                  << std::endl;
      }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  cudaStreamSynchronize(stream);
  std::cout << "md array print" << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  for (int r = 0; r < 4; r++) {
    for (int z = 0; z < size; z++) {
      for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
          if (rank == r) {
            int indx = x + y * size + z * size * size;
            std::cout << "Rank[" << rank << "] Element (" << x << "," << y
                      << "," << z << ") : " << host[indx].real() << " + "
                      << host[indx].imag() << "i" << std::endl;
          }
          MPI_Barrier(MPI_COMM_WORLD);
        }
      }
    }
  }
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::forwardPencil(
    cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream,
    complex_t *input, complex_t *output, complex_t *work_d) {

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
  CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(
      handle, m_GridConfig, output, output, work_d,
      get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));
  // FFT on the second pencil
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_y, output, output, DIRECTION));
  // Tranpose Y to Z
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(
      handle, m_GridConfig, output, output, work_d,
      get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));
  // FFT on the third pencil
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_z, output, output, DIRECTION));
  return S_OK;
}

template <typename real_t>
HRESULT FourierExecutor<real_t>::backwardPencil(
    cudecompHandle_t handle, fftDescriptor desc, cudaStream_t stream,
    complex_t *input, complex_t *output, complex_t *work_d) {

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
  CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(
      handle, m_GridConfig, output, output, work_d,
      get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));
  // FFT on the second pencil
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_y, output, output, DIRECTION));
  // Tranpose Y to X
  CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(
      handle, m_GridConfig, output, output, work_d,
      get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));
  // FFT on the third pencil
  CHECK_CUFFT_EXIT(cufftXtExec(m_Plan_c2c_x, output, output, DIRECTION));

  return S_OK;
}

template <typename real_t> HRESULT FourierExecutor<real_t>::clearPlans() {
  // Destroy the plans
  switch (GetDecomposition(m_GridDescConfig.pdims)) {
  case Decomposition::slab_XY:
    cufftDestroy(m_Plan_c2c_xy);
    cufftDestroy(m_Plan_c2c_z);
    break;
  case Decomposition::slab_YZ:
    cufftDestroy(m_Plan_c2c_x);
    cufftDestroy(m_Plan_c2c_yz);
    break;
  case Decomposition::pencil:
    cufftDestroy(m_Plan_c2c_x);
    cufftDestroy(m_Plan_c2c_y);
    cufftDestroy(m_Plan_c2c_z);
    break;
  case Decomposition::unknown:
    break;
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