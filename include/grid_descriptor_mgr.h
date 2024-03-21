
#ifndef GRIDDESCRIPTORMANAGER_H
#define GRIDDESCRIPTORMANAGER_H

#include "logger.hpp"
#include "checks.h"
#include "fft.h"
#include <cstddef>
#include <cudecomp.h>
#include <unordered_map>

namespace jaxdecomp {

class GridDescriptorManager {

public:
  static GridDescriptorManager &getInstance() {
    static GridDescriptorManager instance; // Guaranteed to be destroyed.
                                           // Instantiated on first use.
    return instance;
  }

  HRESULT createFFTExecutor(fftDescriptor &descriptor, size_t &work_size,
                            std::shared_ptr<FourierExecutor<float>> &executor);

  HRESULT createFFTExecutor(fftDescriptor &descriptor, size_t &work_size,
                            std::shared_ptr<FourierExecutor<double>> &executor);

  inline cudecompHandle_t getHandle() const { return m_Handle; }

  void finalize();

  ~GridDescriptorManager();

private:
  GridDescriptorManager();

  AsyncLogger m_Tracer;
  bool isInitialized = false;

  cudecompHandle_t m_Handle;

  std::unordered_map<fftDescriptor, std::shared_ptr<FourierExecutor<double>>,
                     std::hash<fftDescriptor>, std::equal_to<>>
      m_Descriptors64;
  std::unordered_map<fftDescriptor, std::shared_ptr<FourierExecutor<float>>,
                     std::hash<fftDescriptor>, std::equal_to<>>
      m_Descriptors32;

public:
  GridDescriptorManager(GridDescriptorManager const &) = delete;
  void operator=(GridDescriptorManager const &) = delete;
};

} // namespace jaxdecomp

#endif // GRIDDESCRIPTORMANAGER_H
