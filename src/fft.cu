#include <algorithm>
#include <array>
#include <complex>
#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <cudecomp.h>
#include <cufftXt.h>
#include <numeric>

#include "checks.h"
#include "fft.h"

using namespace std;

namespace jaxdecomp {

static cufftType get_cufft_type_c2c(float) { return CUFFT_C2C; }
static cufftType get_cufft_type_c2c(double) { return CUFFT_Z2Z; }
static cudecompDataType_t get_cudecomp_datatype(cuda::std::complex<float>) {
  return CUDECOMP_FLOAT_COMPLEX;
}
static cudecompDataType_t get_cudecomp_datatype(cuda::std::complex<double>) {
  return CUDECOMP_DOUBLE_COMPLEX;
}
static bool get_double_precision(float) { return false; }
static bool get_double_precision(double) { return true; }

template <typename real_t>
std::pair<int64_t, fftDescriptor_t>
get_fft3d_descriptor(cudecompHandle_t handle, cudecompGridDescConfig_t config,
                     bool forward, bool adjoint) {

  using complex_t = cuda::std::complex<real_t>;

  // Initializing the descriptor
  fftDescriptor_t desc;
  desc.double_precision = get_double_precision(real_t(0));
  desc.config = config;
  desc.forward = forward;
  desc.adjoint = adjoint;

  /* Setting up cuDecomp grid specifications
   */
  int gx = config.gdims[0];
  int gy = config.gdims[1];
  int gz = config.gdims[2];
  cudecompGridDesc_t grid_desc_c; // complex grid
  CHECK_CUDECOMP_EXIT(
      cudecompGridDescCreate(handle, &grid_desc_c, &config, nullptr));

  // Get x-pencil information (complex)
  cudecompPencilInfo_t pinfo_x_c;
  CHECK_CUDECOMP_EXIT(
      cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_x_c, 0, nullptr));

  // Get y-pencil information (complex)
  cudecompPencilInfo_t pinfo_y_c;
  CHECK_CUDECOMP_EXIT(
      cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_y_c, 1, nullptr));

  // Get z-pencil information (complex)
  cudecompPencilInfo_t pinfo_z_c;
  CHECK_CUDECOMP_EXIT(
      cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_z_c, 2, nullptr));

  // Get workspace size
  int64_t num_elements_work_c;
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc_c,
                                                        &num_elements_work_c));

  /**
   * Setting up cuFFT
   * Starting with everything for the x axis
   */
  desc.slab_xy = false;
  desc.slab_yz = false;
  size_t work_sz_c2c_x;

  // x-axis complex-to-complex
  CHECK_CUFFT_EXIT(cufftCreate(&desc.cufft_plan_c2c_x));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(desc.cufft_plan_c2c_x, 0));

  if (config.pdims[0] == 1) {
    // x-y slab: use 2D FFT
    desc.slab_xy = true;
    std::array<int, 2> n{gy, gx};
    CHECK_CUFFT_EXIT(cufftMakePlanMany(
        desc.cufft_plan_c2c_x, 2, n.data(), nullptr, 1,
        pinfo_x_c.shape[0] * pinfo_x_c.shape[1], nullptr, 1,
        pinfo_x_c.shape[0] * pinfo_x_c.shape[1], get_cufft_type_c2c(real_t(0)),
        pinfo_x_c.shape[2], &work_sz_c2c_x));
  } else {
    CHECK_CUFFT_EXIT(cufftMakePlan1d(
        desc.cufft_plan_c2c_x, gx, get_cufft_type_c2c(real_t(0)),
        pinfo_x_c.shape[1] * pinfo_x_c.shape[2], &work_sz_c2c_x));
  }

  // y-axis complex-to-complex
  CHECK_CUFFT_EXIT(cufftCreate(&desc.cufft_plan_c2c_y));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(desc.cufft_plan_c2c_y, 0));
  size_t work_sz_c2c_y;
  if (config.pdims[1] == 1) {
    // y-z slab: use 2D FFT
    desc.slab_yz = true;
    if (config.transpose_axis_contiguous[1]) {
      std::array<int, 2> n{gz, gy};
      CHECK_CUFFT_EXIT(cufftMakePlanMany(
          desc.cufft_plan_c2c_y, 2, n.data(), nullptr, 1,
          pinfo_y_c.shape[0] * pinfo_y_c.shape[1], nullptr, 1,
          pinfo_y_c.shape[0] * pinfo_y_c.shape[1],
          get_cufft_type_c2c(real_t(0)), pinfo_y_c.shape[2], &work_sz_c2c_y));
    } else {
      // Note: In this case, both slab dimensions are strided, leading to slower
      // performance using 2D FFT. Run 1D + 1D instead.
      desc.slab_yz = false;
      CHECK_CUFFT_EXIT(cufftMakePlanMany(
          desc.cufft_plan_c2c_y, 1, &gy /* unused */, &gy, pinfo_y_c.shape[0],
          1, &gy, pinfo_y_c.shape[0], 1, get_cufft_type_c2c(real_t(0)),
          pinfo_y_c.shape[0], &work_sz_c2c_y));
    }
  } else {
    if (config.transpose_axis_contiguous[1]) {
      CHECK_CUFFT_EXIT(cufftMakePlan1d(
          desc.cufft_plan_c2c_y, gy, get_cufft_type_c2c(real_t(0)),
          pinfo_y_c.shape[1] * pinfo_y_c.shape[2], &work_sz_c2c_y));
    } else {
      CHECK_CUFFT_EXIT(cufftMakePlanMany(
          desc.cufft_plan_c2c_y, 1, &gy /* unused */, &gy, pinfo_y_c.shape[0],
          1, &gy, pinfo_y_c.shape[0], 1, get_cufft_type_c2c(real_t(0)),
          pinfo_y_c.shape[0], &work_sz_c2c_y));
    }
  }

  // z-axis complex-to-complex
  CHECK_CUFFT_EXIT(cufftCreate(&desc.cufft_plan_c2c_z));
  CHECK_CUFFT_EXIT(cufftSetAutoAllocation(desc.cufft_plan_c2c_z, 0));
  size_t work_sz_c2c_z;
  if (config.transpose_axis_contiguous[2]) {
    CHECK_CUFFT_EXIT(cufftMakePlan1d(
        desc.cufft_plan_c2c_z, gz, get_cufft_type_c2c(real_t(0)),
        pinfo_z_c.shape[1] * pinfo_z_c.shape[2], &work_sz_c2c_z));
  } else {
    CHECK_CUFFT_EXIT(cufftMakePlanMany(
        desc.cufft_plan_c2c_z, 1, &gz /* unused */, &gz,
        pinfo_z_c.shape[0] * pinfo_z_c.shape[1], 1, &gz,
        pinfo_z_c.shape[0] * pinfo_z_c.shape[1], 1,
        get_cufft_type_c2c(real_t(0)), pinfo_z_c.shape[0] * pinfo_z_c.shape[1],
        &work_sz_c2c_z));
  }

  // Allocate workspace
  int64_t work_sz_decomp, work_sz_cufft, work_sz;
  work_sz_decomp = 2 * num_elements_work_c * sizeof(real_t);
  work_sz_cufft =
      std::max(work_sz_c2c_x, std::max(work_sz_c2c_y, work_sz_c2c_z));
  work_sz = std::max(work_sz_decomp, work_sz_cufft);

  // Cleaning up the things that allocated memory
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc_c));

  // Returning all we need to know about this transform
  return std::make_pair(work_sz, desc);
}

// These functions are adapted from the cuDecomp benchmark code
template <typename real_t>
void fft3d(cudecompHandle_t handle, fftDescriptor_t desc, cudaStream_t stream,
           void **buffers) {

  using complex_t = cuda::std::complex<real_t>;

  void *data_d = buffers[0];
  void *work_d = buffers[1];
  complex_t *data_c_d = static_cast<complex_t *>(data_d);
  complex_t *input = data_c_d;
  complex_t *output = data_c_d;

  cudecompGridDesc_t grid_desc_c; // complex grid
  CHECK_CUDECOMP_EXIT(
      cudecompGridDescCreate(handle, &grid_desc_c, &desc.config, nullptr));

  // Get x-pencil information (complex)
  cudecompPencilInfo_t pinfo_x_c;
  CHECK_CUDECOMP_EXIT(
      cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_x_c, 0, nullptr));

  // Get y-pencil information (complex)
  cudecompPencilInfo_t pinfo_y_c;
  CHECK_CUDECOMP_EXIT(
      cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_y_c, 1, nullptr));

  // Get z-pencil information (complex)
  cudecompPencilInfo_t pinfo_z_c;
  CHECK_CUDECOMP_EXIT(
      cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_z_c, 2, nullptr));

  // Assign cuFFT work area and current XLA stream
  complex_t *work_c_d = static_cast<complex_t *>(work_d);
  CHECK_CUFFT_EXIT(cufftSetStream(desc.cufft_plan_c2c_x, stream));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(desc.cufft_plan_c2c_x, work_d));
  CHECK_CUFFT_EXIT(cufftSetStream(desc.cufft_plan_c2c_y, stream));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(desc.cufft_plan_c2c_y, work_d));
  CHECK_CUFFT_EXIT(cufftSetStream(desc.cufft_plan_c2c_z, stream));
  CHECK_CUFFT_EXIT(cufftSetWorkArea(desc.cufft_plan_c2c_z, work_d));

  // Run 3D FFT sequence

  /*
   * Perform FFT along x and transpose array
   * It assumes that x is initially not distributed.
   */
  if (desc.forward) {
    CHECK_CUFFT_EXIT(cufftXtExec(desc.cufft_plan_c2c_x, input, input,
                                 desc.adjoint ? CUFFT_INVERSE : CUFFT_FORWARD));
    CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(
        handle, grid_desc_c, input, output, work_c_d,
        get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));

    /*
     * Perform FFT along y and transpose
     */
    if (!desc.slab_xy) {
      if (desc.config.transpose_axis_contiguous[1] || desc.slab_yz) {
        CHECK_CUFFT_EXIT(
            cufftXtExec(desc.cufft_plan_c2c_y, output, output,
                        desc.adjoint ? CUFFT_INVERSE : CUFFT_FORWARD));
      } else {
        for (int i = 0; i < pinfo_y_c.shape[2]; ++i) {
          CHECK_CUFFT_EXIT(cufftXtExec(
              desc.cufft_plan_c2c_y,
              output + i * (pinfo_y_c.shape[0] * pinfo_y_c.shape[1]),
              output + i * (pinfo_y_c.shape[0] * pinfo_y_c.shape[1]),
              desc.adjoint ? CUFFT_INVERSE : CUFFT_FORWARD));
        }
      }
    }
    // For y-z slab case, no need to perform yz transposes or z-axis FFT
    if (!desc.slab_yz) {
      CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(
          handle, grid_desc_c, input, output, work_c_d,
          get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));

      /*
       * Perform FFT along z
       */
      CHECK_CUFFT_EXIT(
          cufftXtExec(desc.cufft_plan_c2c_z, output, output,
                      desc.adjoint ? CUFFT_INVERSE : CUFFT_FORWARD));
    }
  } else {

    if (!desc.slab_yz) {

      /* Inverse FFT along z and transpose array
       */
      CHECK_CUFFT_EXIT(
          cufftXtExec(desc.cufft_plan_c2c_z, input, output,
                      desc.adjoint ? CUFFT_FORWARD : CUFFT_INVERSE));

      CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(
          handle, grid_desc_c, input, output, work_c_d,
          get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));
    }

    /* Inverse FFT along y and transpose array
     */
    if (!desc.slab_xy) {
      if (desc.config.transpose_axis_contiguous[1] || desc.slab_yz) {
        CHECK_CUFFT_EXIT(
            cufftXtExec(desc.cufft_plan_c2c_y, output, output,
                        desc.adjoint ? CUFFT_FORWARD : CUFFT_INVERSE));
      } else {
        for (int i = 0; i < pinfo_y_c.shape[2]; ++i) {
          CHECK_CUFFT_EXIT(cufftXtExec(
              desc.cufft_plan_c2c_y,
              output + i * (pinfo_y_c.shape[0] * pinfo_y_c.shape[1]),
              output + i * (pinfo_y_c.shape[0] * pinfo_y_c.shape[1]),
              desc.adjoint ? CUFFT_FORWARD : CUFFT_INVERSE));
        }
      }
    }

    CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(
        handle, grid_desc_c, input, output, work_c_d,
        get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, stream));

    /* Inverse FFT along x and we are back to the real world
     */
    CHECK_CUFFT_EXIT(cufftXtExec(desc.cufft_plan_c2c_x, output, output,
                                 desc.adjoint ? CUFFT_FORWARD : CUFFT_INVERSE));
  }

  // TODO: Check if we need to clean the cufft stuff as well
  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc_c));
};

// Declare specialisations for float and double
template std::pair<int64_t, fftDescriptor_t>
get_fft3d_descriptor<float>(cudecompHandle_t handle,
                            cudecompGridDescConfig_t config, bool forward,
                            bool adjoint);
template std::pair<int64_t, fftDescriptor_t>
get_fft3d_descriptor<double>(cudecompHandle_t handle,
                             cudecompGridDescConfig_t config, bool forward,
                             bool adjoint);

template void fft3d<float>(cudecompHandle_t handle, fftDescriptor_t desc,
                           cudaStream_t stream, void **buffers);
template void fft3d<double>(cudecompHandle_t handle, fftDescriptor_t desc,
                            cudaStream_t stream, void **buffers);
}; // namespace jaxdecomp
