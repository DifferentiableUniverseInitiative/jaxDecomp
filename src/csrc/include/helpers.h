#ifndef _JAX_DECOMP_HELPERS_H_
#define _JAX_DECOMP_HELPERS_H_

#include <nanobind/nanobind.h>
#include <type_traits>
#include "xla/ffi/api/c_api.h"

namespace nb = nanobind;

namespace jaxdecomp {

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From& src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to "
                "be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

// Helper to encapsulate FFI handler for registration with JAX
// Note: XLA_FFI_DEFINE_HANDLER_SYMBOL creates the correct handler type
template <typename T>
nb::capsule EncapsulateFfiHandler(T* fn) {
    return nb::capsule(reinterpret_cast<void*>(fn));
}

} // namespace jaxdecomp

#endif
