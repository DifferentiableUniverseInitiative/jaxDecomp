#include "jaxdecomp.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#ifdef JD_CUDECOMP_BACKEND
#include "checks.h"
#include "fft.h"
#include "grid_descriptor_mgr.h"
#include "halo.h"
#include "logger.hpp"
#include "transpose.h"
#include <cudecomp.h>
#include <mpi.h>
#else
void print_error() {
  throw std::runtime_error("This extension was compiled without CUDA support. cuDecomp functions are not supported.");
}
#endif

#include "xla/ffi/api/ffi.h"

namespace nb = nanobind;
namespace ffi = xla::ffi;
namespace jd = jaxdecomp;

template <ffi::DataType T> constexpr bool is_double_precision_v = (T == ffi::DataType::C128 || T == ffi::DataType::F64);

template <ffi::DataType T> using fft_real_t = std::conditional_t<is_double_precision_v<T>, double, float>;

namespace jaxdecomp {

/**
 * @brief Initializes the global handle
 */
void init() {
#ifdef JD_CUDECOMP_BACKEND
  jd::GridDescriptorManager::getInstance();
#endif
};
/**
 * @brief Finalizes the cuDecomp library
 */
void finalize() {
#ifdef JD_CUDECOMP_BACKEND
  jd::GridDescriptorManager::getInstance().finalize();
#endif
};

#ifdef JD_CUDECOMP_BACKEND
/**
 * @brief Returns Pencil information for a given grid
 */
decompPencilInfo_t getPencilInfo(decompGridDescConfig_t grid_config, int32_t axis) {
  cudecompHandle_t handle(jd::GridDescriptorManager::getInstance().getHandle());
  // Create cuDecomp grid descriptor
  cudecompGridDescConfig_t config;
  cudecompGridDescConfigSet(&config, &grid_config);
  // Create the grid description
  cudecompGridDesc_t grid_desc;
  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));
  cudecompPencilInfo_t pencil_info;
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pencil_info, axis, nullptr, nullptr));
  decompPencilInfo_t result;
  decompPencilInfoSet(&result, &pencil_info);

  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));

  return result;
};

/**
 * @brief Get the Autotuned Grid Config object
 */
decompGridDescConfig_t getAutotunedGridConfig(decompGridDescConfig_t grid_config, bool double_precision,
                                              bool disable_nccl_backends, bool disable_nvshmem_backends,
                                              bool tune_with_transpose, std::array<int32_t, 3> halo_extents,
                                              std::array<bool, 3> halo_periods) {
  // Create cuDecomp grid descriptor
  cudecompHandle_t handle(jd::GridDescriptorManager::getInstance().getHandle());
  cudecompGridDescConfig_t config;
  cudecompGridDescConfigSet(&config, &grid_config);

  // Set up autotune options structure
  cudecompGridDescAutotuneOptions_t options;
  CHECK_CUDECOMP_EXIT(cudecompGridDescAutotuneOptionsSetDefaults(&options));

  // General options
  options.dtype = double_precision ? CUDECOMP_DOUBLE : CUDECOMP_FLOAT;
  options.disable_nccl_backends = disable_nccl_backends;
  options.disable_nvshmem_backends = disable_nvshmem_backends;

  // Process grid autotuning options
  options.grid_mode = tune_with_transpose ? CUDECOMP_AUTOTUNE_GRID_TRANSPOSE : CUDECOMP_AUTOTUNE_GRID_HALO;
  options.allow_uneven_decompositions = false;

  // Transpose communication backend autotuning options
  options.autotune_transpose_backend = true;
  options.transpose_use_inplace_buffers[0] = true;
  options.transpose_use_inplace_buffers[1] = true;
  options.transpose_use_inplace_buffers[2] = true;
  options.transpose_use_inplace_buffers[3] = true;

  // Halo communication backend autotuning options
  options.autotune_halo_backend = true;

  options.halo_axis = 0;

  options.halo_extents[0] = halo_extents[0];
  options.halo_extents[1] = halo_extents[1];
  options.halo_extents[2] = halo_extents[2];

  options.halo_periods[0] = halo_periods[0];
  options.halo_periods[1] = halo_periods[1];
  options.halo_periods[2] = halo_periods[2];

  cudecompGridDesc_t grid_desc;
  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, &options));

  decompGridDescConfig_t output_config;
  output_config.halo_comm_backend = config.halo_comm_backend;
  output_config.transpose_comm_backend = config.transpose_comm_backend;
  for (int i = 0; i < 3; i++) output_config.gdims[i] = config.gdims[i];
  for (int i = 0; i < 2; i++) output_config.pdims[i] = config.pdims[i];

  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));

  return output_config;
};

// ============================================================================
// FFI Handlers
// ============================================================================

/**
 * @brief FFI Handler for parallel 3D FFT
 */
template <ffi::DataType T>
ffi::Error pfft3d_ffi(cudaStream_t stream, ffi::Span<const int64_t> gdims, ffi::Span<const int64_t> pdims,
                      int64_t transpose_comm_backend, int64_t halo_comm_backend, bool forward, bool adjoint,
                      bool contiguous, int64_t decomposition, ffi::Buffer<T> input,
                      ffi::Buffer<ffi::DataType::S8> workspace, ffi::Result<ffi::Buffer<T>> output) {
  if (gdims.size() != 3) { return ffi::Error::InvalidArgument("gdims must have 3 elements"); }
  if (pdims.size() != 2) { return ffi::Error::InvalidArgument("pdims must have 2 elements"); }

  constexpr bool is_double = is_double_precision_v<T>;

  // Reconstruct cudecompGridDescConfig_t from FFI attributes
  cudecompGridDescConfig_t cuconfig;
  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&cuconfig));
  cuconfig.gdims[0] = static_cast<int32_t>(gdims[0]);
  cuconfig.gdims[1] = static_cast<int32_t>(gdims[1]);
  cuconfig.gdims[2] = static_cast<int32_t>(gdims[2]);
  cuconfig.pdims[0] = static_cast<int32_t>(pdims[0]);
  cuconfig.pdims[1] = static_cast<int32_t>(pdims[1]);
  cuconfig.transpose_comm_backend = static_cast<cudecompTransposeCommBackend_t>(transpose_comm_backend);
  cuconfig.halo_comm_backend = static_cast<cudecompHaloCommBackend_t>(halo_comm_backend);
  for (int i = 0; i < 3; i++) { cuconfig.transpose_axis_contiguous[i] = contiguous; }

  // Build descriptor
  jd::fftDescriptor descriptor(cuconfig, is_double, forward, adjoint, contiguous,
                               static_cast<jd::Decomposition>(decomposition));

  // Get buffer pointers
  void* buffers[3] = {const_cast<void*>(input.untyped_data()), const_cast<void*>(workspace.untyped_data()),
                      output->untyped_data()};

  // Execute
  size_t work_size;
  cudecompHandle_t handle(jd::GridDescriptorManager::getInstance().getHandle());
  using real_t = fft_real_t<T>;
  auto executor = std::make_shared<jd::FourierExecutor<real_t>>();
  jd::GridDescriptorManager::getInstance().createFFTExecutor(descriptor, work_size, executor);
  if (descriptor.forward)
    executor->forward(handle, descriptor, stream, buffers);
  else
    executor->backward(handle, descriptor, stream, buffers);

  return ffi::Error::Success();
}

/**
 * @brief FFI Handler for halo exchange
 */
template <ffi::DataType T>
ffi::Error halo_ffi(cudaStream_t stream, ffi::Span<const int64_t> gdims, ffi::Span<const int64_t> pdims,
                    int64_t transpose_comm_backend, int64_t halo_comm_backend, ffi::Span<const int64_t> halo_extents,
                    ffi::Span<const int64_t> halo_periods, int64_t axis, ffi::Buffer<T> input,
                    ffi::Buffer<ffi::DataType::S8> workspace, ffi::Result<ffi::Buffer<T>> output) {
  if (gdims.size() != 3) { return ffi::Error::InvalidArgument("gdims must have 3 elements"); }
  if (pdims.size() != 2) { return ffi::Error::InvalidArgument("pdims must have 2 elements"); }
  if (halo_extents.size() != 3) { return ffi::Error::InvalidArgument("halo_extents must have 3 elements"); }
  if (halo_periods.size() != 3) { return ffi::Error::InvalidArgument("halo_periods must have 3 elements"); }

  constexpr bool is_double = is_double_precision_v<T>;

  // Reconstruct cudecompGridDescConfig_t from FFI attributes
  cudecompGridDescConfig_t cuconfig;
  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&cuconfig));
  cuconfig.gdims[0] = static_cast<int32_t>(gdims[0]);
  cuconfig.gdims[1] = static_cast<int32_t>(gdims[1]);
  cuconfig.gdims[2] = static_cast<int32_t>(gdims[2]);
  cuconfig.pdims[0] = static_cast<int32_t>(pdims[0]);
  cuconfig.pdims[1] = static_cast<int32_t>(pdims[1]);
  cuconfig.transpose_comm_backend = static_cast<cudecompTransposeCommBackend_t>(transpose_comm_backend);
  cuconfig.halo_comm_backend = static_cast<cudecompHaloCommBackend_t>(halo_comm_backend);

  // Build halo descriptor
  jd::haloDescriptor_t descriptor;
  descriptor.double_precision = is_double;
  descriptor.halo_extents = {static_cast<int32_t>(halo_extents[0]), static_cast<int32_t>(halo_extents[1]),
                             static_cast<int32_t>(halo_extents[2])};
  descriptor.halo_periods = {static_cast<bool>(halo_periods[0]), static_cast<bool>(halo_periods[1]),
                             static_cast<bool>(halo_periods[2])};
  descriptor.axis = static_cast<int>(axis);
  descriptor.config = cuconfig;

  // Get buffer pointers
  void* buffers[3] = {const_cast<void*>(input.untyped_data()), const_cast<void*>(workspace.untyped_data()),
                      output->untyped_data()};

  // Execute
  size_t work_size;
  cudecompHandle_t handle(jd::GridDescriptorManager::getInstance().getHandle());
  using real_t = fft_real_t<T>;
  auto executor = std::make_shared<jd::HaloExchange<real_t>>();
  jd::GridDescriptorManager::getInstance().createHaloExecutor(descriptor, work_size, executor);
  executor->halo_exchange(handle, descriptor, stream, buffers);

  return ffi::Error::Success();
}

/**
 * @brief FFI Handler for transpose operations
 */
template <ffi::DataType T>
ffi::Error transpose_ffi(cudaStream_t stream, ffi::Span<const int64_t> gdims, ffi::Span<const int64_t> pdims,
                         int64_t transpose_comm_backend, int64_t halo_comm_backend, int64_t transpose_type,
                         bool contiguous, ffi::Buffer<T> input, ffi::Buffer<ffi::DataType::S8> workspace,
                         ffi::Result<ffi::Buffer<T>> output) {
  if (gdims.size() != 3) { return ffi::Error::InvalidArgument("gdims must have 3 elements"); }
  if (pdims.size() != 2) { return ffi::Error::InvalidArgument("pdims must have 2 elements"); }

  constexpr bool is_double = is_double_precision_v<T>;

  // Reconstruct cudecompGridDescConfig_t from FFI attributes
  cudecompGridDescConfig_t cuconfig;
  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&cuconfig));
  cuconfig.gdims[0] = static_cast<int32_t>(gdims[0]);
  cuconfig.gdims[1] = static_cast<int32_t>(gdims[1]);
  cuconfig.gdims[2] = static_cast<int32_t>(gdims[2]);
  cuconfig.pdims[0] = static_cast<int32_t>(pdims[0]);
  cuconfig.pdims[1] = static_cast<int32_t>(pdims[1]);
  cuconfig.transpose_comm_backend = static_cast<cudecompTransposeCommBackend_t>(transpose_comm_backend);
  cuconfig.halo_comm_backend = static_cast<cudecompHaloCommBackend_t>(halo_comm_backend);
  for (int i = 0; i < 3; i++) { cuconfig.transpose_axis_contiguous[i] = contiguous; }

  // Build descriptor
  jd::transposeDescriptor descriptor(cuconfig, static_cast<jd::TransposeType>(transpose_type), is_double, contiguous);

  // Get buffer pointers
  void* buffers[3] = {const_cast<void*>(input.untyped_data()), const_cast<void*>(workspace.untyped_data()),
                      output->untyped_data()};

  // Execute
  size_t work_size;
  cudecompHandle_t handle(jd::GridDescriptorManager::getInstance().getHandle());
  using real_t = fft_real_t<T>;
  auto executor = std::make_shared<jd::Transpose<real_t>>();
  jd::GridDescriptorManager::getInstance().createTransposeExecutor(descriptor, work_size, executor);
  executor->transpose(handle, descriptor, stream, buffers);

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(pfft_C64, pfft3d_ffi<ffi::DataType::C64>,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Span<const int64_t>>("gdims")
                                  .Attr<ffi::Span<const int64_t>>("pdims")
                                  .Attr<int64_t>("transpose_comm_backend")
                                  .Attr<int64_t>("halo_comm_backend")
                                  .Attr<bool>("forward")
                                  .Attr<bool>("adjoint")
                                  .Attr<bool>("contiguous")
                                  .Attr<int64_t>("decomposition")
                                  .Arg<ffi::Buffer<ffi::DataType::C64>>() // input
                                  .Arg<ffi::Buffer<ffi::DataType::S8>>()  // workspace (bytes)
                                  .Ret<ffi::Buffer<ffi::DataType::C64>>() // output
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(pfft_C128, pfft3d_ffi<ffi::DataType::C128>,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Span<const int64_t>>("gdims")
                                  .Attr<ffi::Span<const int64_t>>("pdims")
                                  .Attr<int64_t>("transpose_comm_backend")
                                  .Attr<int64_t>("halo_comm_backend")
                                  .Attr<bool>("forward")
                                  .Attr<bool>("adjoint")
                                  .Attr<bool>("contiguous")
                                  .Attr<int64_t>("decomposition")
                                  .Arg<ffi::Buffer<ffi::DataType::C128>>() // input
                                  .Arg<ffi::Buffer<ffi::DataType::S8>>()   // workspace (bytes)
                                  .Ret<ffi::Buffer<ffi::DataType::C128>>() // output
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(halo_F32, halo_ffi<ffi::DataType::F32>,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Span<const int64_t>>("gdims")
                                  .Attr<ffi::Span<const int64_t>>("pdims")
                                  .Attr<int64_t>("transpose_comm_backend")
                                  .Attr<int64_t>("halo_comm_backend")
                                  .Attr<ffi::Span<const int64_t>>("halo_extents")
                                  .Attr<ffi::Span<const int64_t>>("halo_periods")
                                  .Attr<int64_t>("axis")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>() // input
                                  .Arg<ffi::Buffer<ffi::DataType::S8>>()  // workspace (bytes)
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>() // output
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(halo_F64, halo_ffi<ffi::DataType::F64>,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Span<const int64_t>>("gdims")
                                  .Attr<ffi::Span<const int64_t>>("pdims")
                                  .Attr<int64_t>("transpose_comm_backend")
                                  .Attr<int64_t>("halo_comm_backend")
                                  .Attr<ffi::Span<const int64_t>>("halo_extents")
                                  .Attr<ffi::Span<const int64_t>>("halo_periods")
                                  .Attr<int64_t>("axis")
                                  .Arg<ffi::Buffer<ffi::DataType::F64>>() // input
                                  .Arg<ffi::Buffer<ffi::DataType::S8>>()  // workspace (bytes)
                                  .Ret<ffi::Buffer<ffi::DataType::F64>>() // output
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(transpose_C64, transpose_ffi<ffi::DataType::C64>,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Span<const int64_t>>("gdims")
                                  .Attr<ffi::Span<const int64_t>>("pdims")
                                  .Attr<int64_t>("transpose_comm_backend")
                                  .Attr<int64_t>("halo_comm_backend")
                                  .Attr<int64_t>("transpose_type")
                                  .Attr<bool>("contiguous")
                                  .Arg<ffi::Buffer<ffi::DataType::C64>>() // input
                                  .Arg<ffi::Buffer<ffi::DataType::S8>>()  // workspace (bytes)
                                  .Ret<ffi::Buffer<ffi::DataType::C64>>() // output
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(transpose_C128, transpose_ffi<ffi::DataType::C128>,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Span<const int64_t>>("gdims")
                                  .Attr<ffi::Span<const int64_t>>("pdims")
                                  .Attr<int64_t>("transpose_comm_backend")
                                  .Attr<int64_t>("halo_comm_backend")
                                  .Attr<int64_t>("transpose_type")
                                  .Attr<bool>("contiguous")
                                  .Arg<ffi::Buffer<ffi::DataType::C128>>() // input
                                  .Arg<ffi::Buffer<ffi::DataType::S8>>()   // workspace (bytes)
                                  .Ret<ffi::Buffer<ffi::DataType::C128>>() // output
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(transpose_F32, transpose_ffi<ffi::DataType::F32>,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Span<const int64_t>>("gdims")
                                  .Attr<ffi::Span<const int64_t>>("pdims")
                                  .Attr<int64_t>("transpose_comm_backend")
                                  .Attr<int64_t>("halo_comm_backend")
                                  .Attr<int64_t>("transpose_type")
                                  .Attr<bool>("contiguous")
                                  .Arg<ffi::Buffer<ffi::DataType::F32>>() // input
                                  .Arg<ffi::Buffer<ffi::DataType::S8>>()  // workspace (bytes)
                                  .Ret<ffi::Buffer<ffi::DataType::F32>>() // output
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(transpose_F64, transpose_ffi<ffi::DataType::F64>,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Attr<ffi::Span<const int64_t>>("gdims")
                                  .Attr<ffi::Span<const int64_t>>("pdims")
                                  .Attr<int64_t>("transpose_comm_backend")
                                  .Attr<int64_t>("halo_comm_backend")
                                  .Attr<int64_t>("transpose_type")
                                  .Attr<bool>("contiguous")
                                  .Arg<ffi::Buffer<ffi::DataType::F64>>() // input
                                  .Arg<ffi::Buffer<ffi::DataType::S8>>()  // workspace (bytes)
                                  .Ret<ffi::Buffer<ffi::DataType::F64>>() // output
);

// ============================================================================
// Workspace size calculation functions
// ============================================================================

/**
 * @brief Get workspace size for FFT operations
 */
int64_t get_fft_workspace_size(std::array<int32_t, 3> gdims, std::array<int32_t, 2> pdims, int transpose_comm_backend,
                               int halo_comm_backend, bool forward, bool double_precision, bool adjoint,
                               bool contiguous, int decomposition) {
  cudecompGridDescConfig_t cuconfig;
  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&cuconfig));
  cuconfig.gdims[0] = gdims[0];
  cuconfig.gdims[1] = gdims[1];
  cuconfig.gdims[2] = gdims[2];
  cuconfig.pdims[0] = pdims[0];
  cuconfig.pdims[1] = pdims[1];
  cuconfig.transpose_comm_backend = static_cast<cudecompTransposeCommBackend_t>(transpose_comm_backend);
  cuconfig.halo_comm_backend = static_cast<cudecompHaloCommBackend_t>(halo_comm_backend);
  for (int i = 0; i < 3; i++) { cuconfig.transpose_axis_contiguous[i] = contiguous; }

  size_t work_size;
  jd::fftDescriptor fftdesc(cuconfig, double_precision, forward, adjoint, contiguous,
                            static_cast<jd::Decomposition>(decomposition));

  if (double_precision) {
    auto executor = std::make_shared<jd::FourierExecutor<double>>();
    jd::GridDescriptorManager::getInstance().createFFTExecutor(fftdesc, work_size, executor);
  } else {
    auto executor = std::make_shared<jd::FourierExecutor<float>>();
    jd::GridDescriptorManager::getInstance().createFFTExecutor(fftdesc, work_size, executor);
  }

  return static_cast<int64_t>(work_size);
}

/**
 * @brief Get workspace size for transpose operations
 */
int64_t get_transpose_workspace_size(std::array<int32_t, 3> gdims, std::array<int32_t, 2> pdims,
                                     int transpose_comm_backend, int halo_comm_backend, int transpose_type,
                                     bool double_precision, bool contiguous) {
  cudecompGridDescConfig_t cuconfig;
  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&cuconfig));
  cuconfig.gdims[0] = gdims[0];
  cuconfig.gdims[1] = gdims[1];
  cuconfig.gdims[2] = gdims[2];
  cuconfig.pdims[0] = pdims[0];
  cuconfig.pdims[1] = pdims[1];
  cuconfig.transpose_comm_backend = static_cast<cudecompTransposeCommBackend_t>(transpose_comm_backend);
  cuconfig.halo_comm_backend = static_cast<cudecompHaloCommBackend_t>(halo_comm_backend);
  for (int i = 0; i < 3; i++) { cuconfig.transpose_axis_contiguous[i] = contiguous; }

  size_t work_size;
  jd::transposeDescriptor desc(cuconfig, static_cast<jd::TransposeType>(transpose_type), double_precision, contiguous);

  if (double_precision) {
    auto executor = std::make_shared<jd::Transpose<double>>();
    jd::GridDescriptorManager::getInstance().createTransposeExecutor(desc, work_size, executor);
  } else {
    auto executor = std::make_shared<jd::Transpose<float>>();
    jd::GridDescriptorManager::getInstance().createTransposeExecutor(desc, work_size, executor);
  }

  return static_cast<int64_t>(work_size);
}

/**
 * @brief Get workspace size for halo exchange operations
 */
int64_t get_halo_workspace_size(std::array<int32_t, 3> gdims, std::array<int32_t, 2> pdims, int transpose_comm_backend,
                                int halo_comm_backend, std::array<int32_t, 3> halo_extents,
                                std::array<bool, 3> halo_periods, int axis, bool double_precision) {
  cudecompGridDescConfig_t cuconfig;
  CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&cuconfig));
  cuconfig.gdims[0] = gdims[0];
  cuconfig.gdims[1] = gdims[1];
  cuconfig.gdims[2] = gdims[2];
  cuconfig.pdims[0] = pdims[0];
  cuconfig.pdims[1] = pdims[1];
  cuconfig.transpose_comm_backend = static_cast<cudecompTransposeCommBackend_t>(transpose_comm_backend);
  cuconfig.halo_comm_backend = static_cast<cudecompHaloCommBackend_t>(halo_comm_backend);

  size_t work_size;
  jd::haloDescriptor_t halo_desc;
  halo_desc.double_precision = double_precision;
  halo_desc.halo_extents = halo_extents;
  halo_desc.halo_periods = halo_periods;
  halo_desc.axis = axis;
  halo_desc.config = cuconfig;

  if (double_precision) {
    auto executor = std::make_shared<jd::HaloExchange<double>>();
    jd::GridDescriptorManager::getInstance().createHaloExecutor(halo_desc, work_size, executor);
  } else {
    auto executor = std::make_shared<jd::HaloExchange<float>>();
    jd::GridDescriptorManager::getInstance().createHaloExecutor(halo_desc, work_size, executor);
  }

  return static_cast<int64_t>(work_size);
}

#else
// Stubs for non-CUDA builds
void getAutotunedGridConfig() { print_error(); }
void getPencilInfo() { print_error(); }
int64_t get_fft_workspace_size(std::array<int32_t, 3>, std::array<int32_t, 2>, int, int, bool, bool, bool, bool, int) {
  print_error();
  return 0;
}
int64_t get_transpose_workspace_size(std::array<int32_t, 3>, std::array<int32_t, 2>, int, int, int, bool, bool) {
  print_error();
  return 0;
}
int64_t get_halo_workspace_size(std::array<int32_t, 3>, std::array<int32_t, 2>, int, int, std::array<int32_t, 3>,
                                std::array<bool, 3>, int, bool) {
  print_error();
  return 0;
}
#endif

/**
 * @brief Encapsulates an FFI handler into a nanobind capsule.
 *
 * This helper function is used to wrap C++ FFI handlers so they can be exposed
 * to Python via nanobind.
 *
 * @tparam T The function type of the FFI handler.
 * @param fn Pointer to the FFI handler function.
 * @return nb::capsule A nanobind capsule containing the FFI handler.
 */
template <typename T> nb::capsule EncapsulateFfiCall(T* fn) {
  // Step 1: Assert that the provided function is a valid XLA FFI handler.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error*, T, XLA_FFI_CallFrame*>,
                "Encapsulated function must be an XLA FFI handler");
  // Step 2: Return a nanobind capsule wrapping the function pointer.
  return nb::capsule(reinterpret_cast<void*>(fn));
}

// Utility to export ops to XLA
nb::dict Registrations() {
  nb::dict dict;
#ifdef JD_CUDECOMP_BACKEND
  dict["pfft_C64"] = EncapsulateFfiCall(pfft_C64);
  dict["pfft_C128"] = EncapsulateFfiCall(pfft_C128);
  dict["transpose_C64"] = EncapsulateFfiCall(transpose_C64);
  dict["transpose_C128"] = EncapsulateFfiCall(transpose_C128);
  dict["transpose_F32"] = EncapsulateFfiCall(transpose_F32);
  dict["transpose_F64"] = EncapsulateFfiCall(transpose_F64);
  dict["halo_F32"] = EncapsulateFfiCall(halo_F32);
  dict["halo_F64"] = EncapsulateFfiCall(halo_F64);
#endif
  return dict;
}

} // namespace jaxdecomp

NB_MODULE(_jaxdecomp, m) {
  // Utilities
  m.def("init", &jd::init);
  m.def("finalize", &jd::finalize);
  m.def("get_pencil_info", &jd::getPencilInfo);
  m.def("get_autotuned_config", &jd::getAutotunedGridConfig);

  // Function registering the custom ops
  m.def("registrations", &jd::Registrations);

  // Workspace size calculation functions
  m.def("get_fft_workspace_size", &jd::get_fft_workspace_size, nb::arg("gdims"), nb::arg("pdims"),
        nb::arg("transpose_comm_backend"), nb::arg("halo_comm_backend"), nb::arg("forward"),
        nb::arg("double_precision"), nb::arg("adjoint"), nb::arg("contiguous"), nb::arg("decomposition"));

  m.def("get_transpose_workspace_size", &jd::get_transpose_workspace_size, nb::arg("gdims"), nb::arg("pdims"),
        nb::arg("transpose_comm_backend"), nb::arg("halo_comm_backend"), nb::arg("transpose_type"),
        nb::arg("double_precision"), nb::arg("contiguous"));

  m.def("get_halo_workspace_size", &jd::get_halo_workspace_size, nb::arg("gdims"), nb::arg("pdims"),
        nb::arg("transpose_comm_backend"), nb::arg("halo_comm_backend"), nb::arg("halo_extents"),
        nb::arg("halo_periods"), nb::arg("axis"), nb::arg("double_precision"));

  // Exported types
  nb::enum_<cudecompTransposeCommBackend_t>(m, "TransposeCommBackend")
      .value("TRANSPOSE_COMM_MPI_P2P", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_MPI_P2P)
      .value("TRANSPOSE_COMM_MPI_P2P_PL", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL)
      .value("TRANSPOSE_COMM_MPI_A2A", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_MPI_A2A)
      .value("TRANSPOSE_COMM_NCCL", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_NCCL)
      .value("TRANSPOSE_COMM_NCCL_PL", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_NCCL_PL)
      .value("TRANSPOSE_COMM_NVSHMEM", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_NVSHMEM)
      .value("TRANSPOSE_COMM_NVSHMEM_PL", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL)
      .export_values();

  nb::enum_<cudecompHaloCommBackend_t>(m, "HaloCommBackend")
      .value("HALO_COMM_MPI", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_MPI)
      .value("HALO_COMM_MPI_BLOCKING", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_MPI_BLOCKING)
      .value("HALO_COMM_NCCL", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_NCCL)
      .value("HALO_COMM_NVSHMEM", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_NVSHMEM)
      .value("HALO_COMM_NVSHMEM_BLOCKING", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING)
      .export_values();

  nb::enum_<jd::TransposeType>(m, "TransposeType")
      .value("TRANSPOSE_XY", jd::TransposeType::TRANSPOSE_XY)
      .value("TRANSPOSE_YZ", jd::TransposeType::TRANSPOSE_YZ)
      .value("TRANSPOSE_ZY", jd::TransposeType::TRANSPOSE_ZY)
      .value("TRANSPOSE_YX", jd::TransposeType::TRANSPOSE_YX)
      .value("TRANSPOSE_XZ", jd::TransposeType::TRANSPOSE_ZX)
      .value("TRANSPOSE_ZX", jd::TransposeType::TRANSPOSE_XZ)
      .export_values();

  nb::enum_<jd::Decomposition>(m, "Decomposition")
      .value("NO_DECOMP", jd::Decomposition::no_decomp)
      .value("SLAB_XY", jd::Decomposition::slab_XY)
      .value("SLAB_YZ", jd::Decomposition::slab_YZ)
      .value("PENCILS", jd::Decomposition::pencil)
      .export_values();

  nb::class_<jd::decompPencilInfo_t>(m, "PencilInfo")
      .def(nb::init<>())
      .def_ro("shape", &jd::decompPencilInfo_t::shape)
      .def_ro("lo", &jd::decompPencilInfo_t::lo)
      .def_ro("hi", &jd::decompPencilInfo_t::hi)
      .def_ro("order", &jd::decompPencilInfo_t::order)
      .def_ro("halo_extents", &jd::decompPencilInfo_t::halo_extents)
      .def_ro("size", &jd::decompPencilInfo_t::size);

  nb::class_<jaxdecomp::decompGridDescConfig_t>(m, "GridConfig")
      .def(nb::init<>())
      .def_rw("gdims", &jd::decompGridDescConfig_t::gdims)
      .def_rw("pdims", &jd::decompGridDescConfig_t::pdims)
      .def_rw("transpose_comm_backend", &jd::decompGridDescConfig_t::transpose_comm_backend)
      .def_rw("halo_comm_backend", &jd::decompGridDescConfig_t::halo_comm_backend);
}
