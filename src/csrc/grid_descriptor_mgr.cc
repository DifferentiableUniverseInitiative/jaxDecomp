
#include "grid_descriptor_mgr.h"
#include "checks.h"
#include "fft.h"
#include "logger.hpp"
#include <cudecomp.h>
#include <ios>
#include <memory>
#include <mpi.h>
#include <ostream>
#include <unordered_map>

namespace jaxdecomp {

GridDescriptorManager::GridDescriptorManager() : m_Tracer("JAXDECOMP") {

  StartTraceInfo(m_Tracer) << "JaxDecomp start up" << std::endl;
  // Initialize MPI
  MPI_Comm mpi_comm = MPI_COMM_WORLD;

  // Check if MPI has already been initialized by the user (maybe with mpi4py)
  CHECK_MPI_EXIT(MPI_Initialized(&isMPIalreadyInitialized));
  if (!isMPIalreadyInitialized) { CHECK_MPI_EXIT(MPI_Init(nullptr, nullptr)); }
  // Initialize cuDecomp
  CHECK_CUDECOMP_EXIT(cudecompInit(&m_Handle, mpi_comm));
  isInitialized = true;
}

HRESULT GridDescriptorManager::createFFTExecutor(fftDescriptor& descriptor, size_t& work_size,
                                                 std::shared_ptr<FourierExecutor<double>>& executor) {

  HRESULT hr(E_FAIL);

  auto it = m_Descriptors64.find(descriptor);
  if (it != m_Descriptors64.end()) {

    executor = it->second;
    work_size = executor->m_WorkSize;
    hr = S_FALSE;
  }

  if (hr == E_FAIL) {
    hr = executor->Initialize(m_Handle, work_size, descriptor);
    if (SUCCEEDED(hr)) { m_Descriptors64[descriptor] = executor; }
  }
  return hr;
}

HRESULT GridDescriptorManager::createFFTExecutor(fftDescriptor& descriptor, size_t& work_size,
                                                 std::shared_ptr<FourierExecutor<float>>& executor) {

  HRESULT hr(E_FAIL);

  auto it = m_Descriptors32.find(descriptor);
  if (it != m_Descriptors32.end()) {
    work_size = it->second->m_WorkSize;
    executor = it->second;
    hr = S_FALSE;
  }

  if (hr == E_FAIL) {
    hr = executor->Initialize(m_Handle, work_size, descriptor);
    if (SUCCEEDED(hr)) { m_Descriptors32[descriptor] = executor; }
  }
  return hr;
}

HRESULT GridDescriptorManager::createHaloExecutor(haloDescriptor_t& descriptor, size_t& work_size,
                                                  std::shared_ptr<HaloExchange<float>>& executor) {

  HRESULT hr(E_FAIL);

  auto it = m_HaloDescriptors32.find(descriptor);
  if (it != m_HaloDescriptors32.end()) {
    work_size = it->second->m_WorkSize;
    executor = it->second;
    hr = S_FALSE;
  }

  if (hr == E_FAIL) {
    executor = std::make_shared<HaloExchange<float>>();
    hr = executor->get_halo_descriptor(m_Handle, work_size, descriptor);
    if (SUCCEEDED(hr)) { m_HaloDescriptors32[descriptor] = executor; }
  }
  return hr;
}

HRESULT GridDescriptorManager::createHaloExecutor(haloDescriptor_t& descriptor, size_t& work_size,
                                                  std::shared_ptr<HaloExchange<double>>& executor) {

  HRESULT hr(E_FAIL);

  auto it = m_HaloDescriptors64.find(descriptor);
  if (it != m_HaloDescriptors64.end()) {
    work_size = it->second->m_WorkSize;
    executor = it->second;
    hr = S_FALSE;
  }

  if (hr == E_FAIL) {
    executor = std::make_shared<HaloExchange<double>>();
    hr = executor->get_halo_descriptor(m_Handle, work_size, descriptor);
    if (SUCCEEDED(hr)) { m_HaloDescriptors64[descriptor] = executor; }
  }
  return hr;
}

HRESULT GridDescriptorManager::createTransposeExecutor(transposeDescriptor& descriptor, size_t& work_size,
                                                       std::shared_ptr<Transpose<double>>& executor) {
  HRESULT hr(E_FAIL);

  auto it = m_TransposeDescriptors64.find(descriptor);
  if (it != m_TransposeDescriptors64.end()) {
    work_size = it->second->m_WorkSize;
    executor = it->second;
    hr = S_FALSE;
  }

  if (hr == E_FAIL) {
    executor = std::make_shared<Transpose<double>>();
    hr = executor->get_transpose_descriptor(m_Handle, work_size, descriptor);
    if (SUCCEEDED(hr)) { m_TransposeDescriptors64[descriptor] = executor; }
  }
  return hr;
}

HRESULT GridDescriptorManager::createTransposeExecutor(transposeDescriptor& descriptor, size_t& work_size,
                                                       std::shared_ptr<Transpose<float>>& executor) {
  HRESULT hr(E_FAIL);

  auto it = m_TransposeDescriptors32.find(descriptor);
  if (it != m_TransposeDescriptors32.end()) {
    work_size = it->second->m_WorkSize;
    executor = it->second;
    hr = S_FALSE;
  }

  if (hr == E_FAIL) {
    executor = std::make_shared<Transpose<float>>();
    hr = executor->get_transpose_descriptor(m_Handle, work_size, descriptor);
    if (SUCCEEDED(hr)) { m_TransposeDescriptors32[descriptor] = executor; }
  }
  return hr;
}

// TODO(Wassim) : This can be cleanup using some polymorphism
void GridDescriptorManager::finalize() {
  if (!isInitialized) return;

  StartTraceInfo(m_Tracer) << "JaxDecomp shut down" << std::endl;
  // Destroy grid descriptors for FFTs
  for (auto& descriptor : m_Descriptors64) {
    auto& executor = descriptor.second;
    // TODO(wassim) : Cleanup cudecomp resources
    // CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc_c, work)); This can
    // be used instead of requesting XLA to allocate the memory
    cudecompResult_t err = cudecompGridDescDestroy(m_Handle, executor->m_GridConfig);
    // Do not throw exceptioin here, this called when the library is being
    // unloaded we should not throw exceptions here
    if (CUDECOMP_RESULT_SUCCESS != err) {
      StartTraceInfo(m_Tracer) << "CUDECOMP error.at exit " << err << ")" << std::endl;
    }
    executor->clearPlans();
  }

  for (auto& descriptor : m_Descriptors32) {
    auto& executor = descriptor.second;
    // Cleanup cudecomp resources
    // CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc_c, work)); This can
    // be used instead of requesting XLA to allocate the memory
    cudecompResult_t err = cudecompGridDescDestroy(m_Handle, executor->m_GridConfig);
    if (CUDECOMP_RESULT_SUCCESS != err) {
      StartTraceInfo(m_Tracer) << "CUDECOMP error.at exit " << err << ")" << std::endl;
    }
    executor->clearPlans();
  }

  // Destroy Halo descriptors
  for (auto& descriptor : m_HaloDescriptors64) {
    auto& executor = descriptor.second;
    // Cleanup cudecomp resources
    // CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc_c, work)); This can
    // be used instead of requesting XLA to allocate the memory
    cudecompResult_t err = cudecompGridDescDestroy(m_Handle, executor->m_GridConfig);
    if (CUDECOMP_RESULT_SUCCESS != err) {
      StartTraceInfo(m_Tracer) << "CUDECOMP error.at exit " << err << ")" << std::endl;
    }
    executor->cleanUp(m_Handle);
  }

  for (auto& descriptor : m_HaloDescriptors32) {
    auto& executor = descriptor.second;
    // Cleanup cudecomp resources
    // CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc_c, work)); This can
    // be used instead of requesting XLA to allocate the memory
    cudecompResult_t err = cudecompGridDescDestroy(m_Handle, executor->m_GridConfig);
    if (CUDECOMP_RESULT_SUCCESS != err) {
      StartTraceInfo(m_Tracer) << "CUDECOMP error.at exit " << err << ")" << std::endl;
    }
    executor->cleanUp(m_Handle);
  }

  // Destroy Transpose descriptors
  for (auto& descriptor : m_TransposeDescriptors64) {
    auto& executor = descriptor.second;
    // Cleanup cudecomp resources
    // CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc_c, work)); This can
    // be used instead of requesting XLA to allocate the memory
    cudecompResult_t err = cudecompGridDescDestroy(m_Handle, executor->m_GridConfig);
    if (CUDECOMP_RESULT_SUCCESS != err) {
      StartTraceInfo(m_Tracer) << "CUDECOMP error.at exit " << err << ")" << std::endl;
    }
  }

  for (auto& descriptor : m_TransposeDescriptors32) {
    auto& executor = descriptor.second;
    // Cleanup cudecomp resources
    // CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc_c, work)); This can
    // be used instead of requesting XLA to allocate the memory
    cudecompResult_t err = cudecompGridDescDestroy(m_Handle, executor->m_GridConfig);
    if (CUDECOMP_RESULT_SUCCESS != err) {
      StartTraceInfo(m_Tracer) << "CUDECOMP error.at exit " << err << ")" << std::endl;
    }
  }

  // TODO(wassim) : Cleanup cudecomp resources
  //  there is an issue with mpi4py calling finalize at py_exit before this
  cudecompFinalize(m_Handle);
  // Clean finish
  CHECK_CUDA_EXIT(cudaDeviceSynchronize());
  // If jaxDecomp initialized MPI finalize it
  // Otherwise mpi4py will finalize its own MPI WORLD Communicator
  if (!isMPIalreadyInitialized) { CHECK_MPI_EXIT(MPI_Finalize()); }
  isInitialized = false;
}

GridDescriptorManager::~GridDescriptorManager() {
  if (isInitialized) { finalize(); }
}
} // namespace jaxdecomp
