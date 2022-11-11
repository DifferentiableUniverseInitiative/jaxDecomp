#ifndef _JAX_DECOMP_CHECKS_H_
#define _JAX_DECOMP_CHECKS_H_


#include <cstdio>
using namespace std;

// Error checking macros
#define CHECK_CUDECOMP_EXIT(call)                                                                                      \
  do {                                                                                                                 \
    cudecompResult_t err = call;                                                                                       \
    if (CUDECOMP_RESULT_SUCCESS != err) {                                                                              \
      fprintf(stderr, "%s:%d CUDECOMP error. (error code %d)\n", __FILE__, __LINE__, err);                             \
      throw exception();                                               \
    }                                                                                                                  \
  } while (false)



#endif