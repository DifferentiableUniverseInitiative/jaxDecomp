#include "jaxdecomp.h"
#include "checks.h"
#include "fft.h"
#include "grid_descriptor_mgr.h"
#include "halo.h"
#include "helpers.h"
#include "logger.hpp"
#include <cudecomp.h>
#include <mpi.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace jd = jaxdecomp;

namespace jaxdecomp {

// Global cuDecomp handle, initialized once from Python when the
// library is imported, and then implicitly reused in all functions
// cudecompHandle_t handle;

/**
 * @brief Initializes the global handle
 */
void init() { jd::GridDescriptorManager::getInstance(); };
/**
 * @brief Finalizes the cuDecomp library
 */
void finalize() { jd::GridDescriptorManager::getInstance().finalize(); };

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
  CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pencil_info, axis, nullptr));
  decompPencilInfo_t result;
  decompPencilInfoSet(&result, &pencil_info);

  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));

  return result;
};

/**
 * @brief Get the Autotuned Grid Config object
 *
 * @param grid_config
 * @param double_precision
 * @param disable_nccl_backends
 * @param disable_nvshmem_backends
 * @param tune_with_transpose
 * @param halo_extents
 * @param halo_periods
 * @return decompGridDescConfig_t
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
  // TODO(wassim) this was updated in cuDecomp check if autotuning works
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

/// XLA interface ops
void transposeXtoY(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {

  cudecompHandle_t handle(jd::GridDescriptorManager::getInstance().getHandle());
  void* data_d = buffers[0]; // In place operations, so only one buffer

  // Create cuDecomp grid descriptor
  cudecompGridDescConfig_t config = *UnpackDescriptor<cudecompGridDescConfig_t>(opaque, opaque_len);

  // Create the grid description
  cudecompGridDesc_t grid_desc;
  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));

  // Get workspace sizes
  int64_t transpose_work_num_elements;
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, &transpose_work_num_elements));

  int64_t dtype_size;
  CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(CUDECOMP_FLOAT, &dtype_size));

  double* transpose_work_d;
  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&transpose_work_d),
                                     transpose_work_num_elements * dtype_size));

  CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_FLOAT,
                                            nullptr, nullptr, stream));

  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, transpose_work_d));

  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));
}

void transposeYtoZ(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
  cudecompHandle_t handle(jd::GridDescriptorManager::getInstance().getHandle());

  void* data_d = buffers[0]; // In place operations, so only one buffer

  // Create cuDecomp grid descriptor
  cudecompGridDescConfig_t config = *UnpackDescriptor<cudecompGridDescConfig_t>(opaque, opaque_len);

  // Create the grid description
  cudecompGridDesc_t grid_desc;
  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));

  // Get workspace sizes
  int64_t transpose_work_num_elements;
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, &transpose_work_num_elements));

  int64_t dtype_size;
  CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(CUDECOMP_FLOAT, &dtype_size));

  double* transpose_work_d;
  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&transpose_work_d),
                                     transpose_work_num_elements * dtype_size));

  CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_FLOAT,
                                            nullptr, nullptr, stream));

  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, transpose_work_d));

  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));
}

void transposeZtoY(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
  cudecompHandle_t handle(jd::GridDescriptorManager::getInstance().getHandle());

  void* data_d = buffers[0]; // In place operations, so only one buffer

  // Create cuDecomp grid descriptor
  cudecompGridDescConfig_t config = *UnpackDescriptor<cudecompGridDescConfig_t>(opaque, opaque_len);

  // Create the grid description
  cudecompGridDesc_t grid_desc;
  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));

  // Get workspace sizes
  int64_t transpose_work_num_elements;
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, &transpose_work_num_elements));

  int64_t dtype_size;
  CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(CUDECOMP_FLOAT, &dtype_size));

  double* transpose_work_d;
  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&transpose_work_d),
                                     transpose_work_num_elements * dtype_size));

  CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_FLOAT,
                                            nullptr, nullptr, stream));

  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, transpose_work_d));

  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));
}

void transposeYtoX(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
  cudecompHandle_t handle(jd::GridDescriptorManager::getInstance().getHandle());

  void* data_d = buffers[0]; // In place operations, so only one buffer

  // Create cuDecomp grid descriptor
  cudecompGridDescConfig_t config = *UnpackDescriptor<cudecompGridDescConfig_t>(opaque, opaque_len);

  // Create the grid description
  cudecompGridDesc_t grid_desc;
  CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));

  // Get workspace sizes
  int64_t transpose_work_num_elements;
  CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, &transpose_work_num_elements));

  int64_t dtype_size;
  CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(CUDECOMP_FLOAT, &dtype_size));

  double* transpose_work_d;
  CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&transpose_work_d),
                                     transpose_work_num_elements * dtype_size));

  CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_FLOAT,
                                            nullptr, nullptr, stream));

  CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, transpose_work_d));

  CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));
}

/**
 * @brief Wrapper to cuDecomp-based FFTs
 */
void pfft3d(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {

  fftDescriptor descriptor = *UnpackDescriptor<fftDescriptor>(opaque, opaque_len);

  size_t work_size;
  cudecompHandle_t my_handle(jd::GridDescriptorManager::getInstance().getHandle());
  // Execute the correct version of the FFT
  if (descriptor.double_precision) {

    auto executor = std::make_shared<jd::FourierExecutor<double>>();
    jd::GridDescriptorManager::getInstance().createFFTExecutor(descriptor, work_size, executor);

    if (descriptor.forward)
      executor->forward(my_handle, descriptor, stream, buffers);
    else
      executor->backward(my_handle, descriptor, stream, buffers);
  } else {

    auto executor = std::make_shared<jd::FourierExecutor<float>>();
    jd::GridDescriptorManager::getInstance().createFFTExecutor(descriptor, work_size, executor);

    if (descriptor.forward)
      executor->forward(my_handle, descriptor, stream, buffers);
    else
      executor->backward(my_handle, descriptor, stream, buffers);
  }
}

/**
 * @brief Perfom a halo exchange along the 3 dimensions
 *
 */
void halo(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
  cudecompHandle_t handle(jd::GridDescriptorManager::getInstance().getHandle());

  haloDescriptor_t descriptor = *UnpackDescriptor<haloDescriptor_t>(opaque, opaque_len);

  size_t work_size;
  // Execute the correct version of the halo exchange
  if (descriptor.double_precision) {

    auto executor = std::make_shared<jd::HaloExchange<double>>();
    jd::GridDescriptorManager::getInstance().createHaloExecutor(descriptor, work_size, executor);

    executor->halo_exchange(handle, descriptor, stream, buffers);
  } else {

    auto executor = std::make_shared<jd::HaloExchange<float>>();
    jd::GridDescriptorManager::getInstance().createHaloExecutor(descriptor, work_size, executor);

    executor->halo_exchange(handle, descriptor, stream, buffers);
  }
}

// Utility to export ops to XLA
py::dict Registrations() {
  py::dict dict;
  dict["transpose_x_y"] = EncapsulateFunction(transposeXtoY);
  dict["transpose_y_z"] = EncapsulateFunction(transposeYtoZ);
  dict["transpose_z_y"] = EncapsulateFunction(transposeZtoY);
  dict["transpose_y_x"] = EncapsulateFunction(transposeYtoX);
  dict["pfft3d"] = EncapsulateFunction(pfft3d);
  dict["halo"] = EncapsulateFunction(halo);
  return dict;
}
} // namespace jaxdecomp

PYBIND11_MODULE(_jaxdecomp, m) {
  // Utilities
  m.def("init", &jd::init);
  m.def("finalize", &jd::finalize);
  m.def("get_pencil_info", &jd::getPencilInfo);
  m.def("get_autotuned_config", &jd::getAutotunedGridConfig);

  // Function registering the custom ops
  m.def("registrations", &jd::Registrations);

  // Utilities for exported ops
  m.def("build_transpose_descriptor", [](jd::decompGridDescConfig_t config) {
    cudecompGridDescConfig_t cuconfig;
    cudecompGridDescConfigSet(&cuconfig, &config);
    return jd::PackDescriptor(cuconfig);
  });

  m.def("build_fft_descriptor",
        [](jd::decompGridDescConfig_t config, bool forward, bool double_precision, bool adjoint) {
          // Create a real cuDecomp grid descriptor
          cudecompGridDescConfig_t cuconfig;
          cudecompGridDescConfigSet(&cuconfig, &config);

          size_t work_size;
          jd::fftDescriptor fftdesc(cuconfig, double_precision, forward, adjoint);
          if (double_precision) {

            auto executor = std::make_shared<jd::FourierExecutor<double>>();

            HRESULT hr = jd::GridDescriptorManager::getInstance().createFFTExecutor(fftdesc, work_size, executor);

            return std::pair<int64_t, pybind11::bytes>(work_size, PackDescriptor(fftdesc));

          } else {
            auto executor = std::make_shared<jd::FourierExecutor<float>>();

            HRESULT hr = jd::GridDescriptorManager::getInstance().createFFTExecutor(fftdesc, work_size, executor);

            return std::pair<int64_t, pybind11::bytes>(work_size, PackDescriptor(fftdesc));
          }
        });

  m.def("build_halo_descriptor",
        [](jd::decompGridDescConfig_t config, bool double_precision, std::array<int32_t, 3> halo_extents,
           std::array<bool, 3> halo_periods, int axis = 0) {
          // Create a real cuDecomp grid descriptor
          cudecompGridDescConfig_t cuconfig;
          cudecompGridDescConfigSet(&cuconfig, &config);
          cudecompHandle_t handle(jd::GridDescriptorManager::getInstance().getHandle());

          size_t work_size;
          jd::haloDescriptor_t halo_desc;
          halo_desc.double_precision = double_precision;
          halo_desc.halo_extents = halo_extents;
          halo_desc.halo_periods = halo_periods;
          halo_desc.axis = axis;
          halo_desc.config = cuconfig;

          if (double_precision) {
            auto executor = std::make_shared<jd::HaloExchange<double>>();
            HRESULT hr = jd::GridDescriptorManager::getInstance().createHaloExecutor(halo_desc, work_size, executor);
          } else {
            auto executor = std::make_shared<jd::HaloExchange<float>>();
            HRESULT hr = jd::GridDescriptorManager::getInstance().createHaloExecutor(halo_desc, work_size, executor);
          }

          return std::pair<int64_t, pybind11::bytes>(work_size, PackDescriptor(halo_desc));
        });

  // Exported types
  py::enum_<cudecompTransposeCommBackend_t>(m, "TransposeCommBackend")
      .value("TRANSPOSE_COMM_MPI_P2P", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_MPI_P2P)
      .value("TRANSPOSE_COMM_MPI_P2P_PL", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL)
      .value("TRANSPOSE_COMM_MPI_A2A", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_MPI_A2A)
      .value("TRANSPOSE_COMM_NCCL", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_NCCL)
      .value("TRANSPOSE_COMM_NCCL_PL", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_NCCL_PL)
      .value("TRANSPOSE_COMM_NVSHMEM", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_NVSHMEM)
      .value("TRANSPOSE_COMM_NVSHMEM_PL", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL)
      .export_values();

  py::enum_<cudecompHaloCommBackend_t>(m, "HaloCommBackend")
      .value("HALO_COMM_MPI", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_MPI)
      .value("HALO_COMM_MPI_BLOCKING", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_MPI_BLOCKING)
      .value("HALO_COMM_NCCL", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_NCCL)
      .value("HALO_COMM_NVSHMEM", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_NVSHMEM)
      .value("HALO_COMM_NVSHMEM_BLOCKING", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING)
      .export_values();

  py::class_<jd::decompPencilInfo_t> pencil_info(m, "PencilInfo");
  pencil_info.def(py::init<>())
      .def_readonly("shape", &jd::decompPencilInfo_t::shape)
      .def_readonly("lo", &jd::decompPencilInfo_t::lo)
      .def_readonly("hi", &jd::decompPencilInfo_t::hi)
      .def_readonly("order", &jd::decompPencilInfo_t::order)
      .def_readonly("halo_extents", &jd::decompPencilInfo_t::halo_extents)
      .def_readonly("size", &jd::decompPencilInfo_t::size);

  py::class_<jaxdecomp::decompGridDescConfig_t> config(m, "GridConfig");
  config.def(py::init<>())
      .def_readwrite("gdims", &jd::decompGridDescConfig_t::gdims)
      .def_readwrite("pdims", &jd::decompGridDescConfig_t::pdims)
      .def_readwrite("transpose_comm_backend", &jd::decompGridDescConfig_t::transpose_comm_backend)
      .def_readwrite("halo_comm_backend", &jd::decompGridDescConfig_t::halo_comm_backend);
}
