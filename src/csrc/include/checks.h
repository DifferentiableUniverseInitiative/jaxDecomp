#ifndef _JAX_DECOMP_CHECKS_H_
#define _JAX_DECOMP_CHECKS_H_

#include <cstdio>

typedef long HRESULT;
using namespace std;

#define SUCCEEDED(hr) (((HRESULT)(hr)) >= 0)
#define FAILED(hr) (((HRESULT)(hr)) < 0)

#define S_OK ((HRESULT)0x00000000L)
#define S_FALSE ((HRESULT)1L)

#define E_ABORT ((HRESULT)0x80004004L)
#define E_ACCESSDENIED ((HRESULT)0x80070005L)
#define E_FAIL ((HRESULT)0x80004005L)
#define E_HANDLE ((HRESULT)0x80070006L)
#define E_INVALIDARG ((HRESULT)0x80070057L)
#define E_NOINTERFACE ((HRESULT)0x80004002L)
#define E_NOTIMPL ((HRESULT)0x80004001L)
#define E_OUTOFMEMORY ((HRESULT)0x8007000EL)
#define E_POINTER ((HRESULT)0x80004003L)
#define E_UNEXPECTED ((HRESULT)0x8000FFFFL)
#define E_OUTOFMEMORY ((HRESULT)0x8007000EL)
#define E_NOTIMPL ((HRESULT)0x80004001L)

// Macro to check for CUDA errors
#define CHECK_CUDA_EXIT(call)                                                                                          \
  do {                                                                                                                 \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  } while (0)

// Error checking macros
#define CHECK_CUDECOMP_EXIT(call)                                                                                      \
  do {                                                                                                                 \
    cudecompResult_t err = call;                                                                                       \
    if (CUDECOMP_RESULT_SUCCESS != err) {                                                                              \
      fprintf(stderr, "%s:%d CUDECOMP error. (error code %d)\n", __FILE__, __LINE__, err);                             \
      throw exception();                                                                                               \
    }                                                                                                                  \
  } while (false)

#define CHECK_CUFFT_EXIT(call)                                                                                         \
  do {                                                                                                                 \
    cufftResult_t err = call;                                                                                          \
    if (CUFFT_SUCCESS != err) {                                                                                        \
      fprintf(stderr, "%s:%d CUFFT error. (error code %d)\n", __FILE__, __LINE__, err);                                \
      throw exception();                                                                                               \
    }                                                                                                                  \
  } while (false)

#define CHECK_MPI_EXIT(call)                                                                                           \
  {                                                                                                                    \
    int err = call;                                                                                                    \
    if (0 != err) {                                                                                                    \
      char error_str[MPI_MAX_ERROR_STRING];                                                                            \
      int len;                                                                                                         \
      MPI_Error_string(err, error_str, &len);                                                                          \
      if (error_str) {                                                                                                 \
        fprintf(stderr, "%s:%d MPI error. (%s)\n", __FILE__, __LINE__, error_str);                                     \
      } else {                                                                                                         \
        fprintf(stderr, "%s:%d MPI error. (error code %d)\n", __FILE__, __LINE__, err);                                \
      }                                                                                                                \
      exit(EXIT_FAILURE);                                                                                              \
    }                                                                                                                  \
  }                                                                                                                    \
  while (false)

#define HR2STR(hr)                                                                                                     \
  ((hr == S_OK)             ? "S_OK"                                                                                   \
   : (hr == S_FALSE)        ? "S_FALSE"                                                                                \
   : (hr == E_ABORT)        ? "E_ABORT"                                                                                \
   : (hr == E_ACCESSDENIED) ? "E_ACCESSDENIED"                                                                         \
   : (hr == E_FAIL)         ? "E_FAIL"                                                                                 \
   : (hr == E_HANDLE)       ? "E_HANDLE"                                                                               \
   : (hr == E_INVALIDARG)   ? "E_INVALIDARG"                                                                           \
   : (hr == E_NOINTERFACE)  ? "E_NOINTERFACE"                                                                          \
   : (hr == E_NOTIMPL)      ? "E_NOTIMPL"                                                                              \
   : (hr == E_OUTOFMEMORY)  ? "E_OUTOFMEMORY"                                                                          \
   : (hr == E_POINTER)      ? "E_POINTER"                                                                              \
   : (hr == E_UNEXPECTED)   ? "E_UNEXPECTED"                                                                           \
                            : "Unknown HRESULT")

#endif // _JAX_DECOMP_CHECKS_H_
