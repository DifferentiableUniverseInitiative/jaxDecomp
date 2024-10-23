#include "jaxdecomp.h"
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
#include "helpers.h"
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

void transpose(cudaStream_t stream, void** buffers, const char* opaque, size_t opaque_len) {
  transposeDescriptor descriptor = *UnpackDescriptor<transposeDescriptor>(opaque, opaque_len);

  size_t work_size;
  cudecompHandle_t my_handle(jd::GridDescriptorManager::getInstance().getHandle());
  if (descriptor.double_precision) {

    auto executor = std::make_shared<jd::Transpose<double>>();
    jd::GridDescriptorManager::getInstance().createTransposeExecutor(descriptor, work_size, executor);
    executor->transpose(my_handle, descriptor, stream, buffers);

  } else {

    auto executor = std::make_shared<jd::Transpose<float>>();
    jd::GridDescriptorManager::getInstance().createTransposeExecutor(descriptor, work_size, executor);
    executor->transpose(my_handle, descriptor, stream, buffers);
  }
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
#else
void getAutotunedGridConfig() { print_error(); }
void getPencilInfo() { print_error(); }
void transpose() { print_error(); }
void pfft3d() { print_error(); }
void halo() { print_error(); }
#endif

// Utility to export ops to XLA
py::dict Registrations() {
  py::dict dict;
  dict["transpose"] = EncapsulateFunction(transpose);
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
  m.def("build_transpose_descriptor",
        [](jd::decompGridDescConfig_t config, jd::TransposeType type, bool double_precision, bool contiguous) {
#ifdef JD_CUDECOMP_BACKEND
          cudecompGridDescConfig_t cuconfig;
          cudecompGridDescConfigSet(&cuconfig, &config);
          size_t work_size;
          jd::transposeDescriptor desc(cuconfig, type, double_precision, contiguous);

          if (double_precision) {
            auto executor = std::make_shared<jd::Transpose<double>>();
            HRESULT hr = jd::GridDescriptorManager::getInstance().createTransposeExecutor(desc, work_size, executor);
          } else {
            auto executor = std::make_shared<jd::Transpose<float>>();
            HRESULT hr = jd::GridDescriptorManager::getInstance().createTransposeExecutor(desc, work_size, executor);
          }

          return std::pair<int64_t, pybind11::bytes>(work_size, PackDescriptor(desc));
#else
        print_error();
#endif
        });

  m.def("build_fft_descriptor", [](jd::decompGridDescConfig_t config, bool forward, bool double_precision, bool adjoint,
                                   bool contiguous, jd::Decomposition decomp) {
#ifdef JD_CUDECOMP_BACKEND
    // Create a real cuDecomp grid descriptor
    cudecompGridDescConfig_t cuconfig;
    cudecompGridDescConfigSet(&cuconfig, &config);

    size_t work_size;
    jd::fftDescriptor fftdesc(cuconfig, double_precision, forward, adjoint, contiguous, decomp);

    if (double_precision) {

      auto executor = std::make_shared<jd::FourierExecutor<double>>();
      HRESULT hr = jd::GridDescriptorManager::getInstance().createFFTExecutor(fftdesc, work_size, executor);

    } else {
      auto executor = std::make_shared<jd::FourierExecutor<float>>();
      HRESULT hr = jd::GridDescriptorManager::getInstance().createFFTExecutor(fftdesc, work_size, executor);
    }
    return std::pair<int64_t, pybind11::bytes>(work_size, PackDescriptor(fftdesc));
#else
        print_error();
#endif
  });

  m.def("build_halo_descriptor",
        [](jd::decompGridDescConfig_t config, bool double_precision, std::array<int32_t, 3> halo_extents,
           std::array<bool, 3> halo_periods, int axis = 0) {
#ifdef JD_CUDECOMP_BACKEND
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
#else
        print_error();
#endif
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

  py::enum_<jd::TransposeType>(m, "TransposeType")
      .value("TRANSPOSE_XY", jd::TransposeType::TRANSPOSE_XY)
      .value("TRANSPOSE_YZ", jd::TransposeType::TRANSPOSE_YZ)
      .value("TRANSPOSE_ZY", jd::TransposeType::TRANSPOSE_ZY)
      .value("TRANSPOSE_YX", jd::TransposeType::TRANSPOSE_YX)
      .value("TRANSPOSE_XZ", jd::TransposeType::TRANSPOSE_ZX)
      .value("TRANSPOSE_ZX", jd::TransposeType::TRANSPOSE_XZ)
      .export_values();

  py::enum_<jd::Decomposition>(m, "Decomposition")
      .value("NO_DECOMP", jd::Decomposition::no_decomp)
      .value("SLAB_XY", jd::Decomposition::slab_XY)
      .value("SLAB_YZ", jd::Decomposition::slab_YZ)
      .value("PENCILS", jd::Decomposition::pencil)
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
