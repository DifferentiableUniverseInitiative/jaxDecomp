#include <pybind11/pybind11.h>
#include <cudecomp.h>
#include <mpi.h>
#include "checks.h"

namespace py = pybind11;

namespace jaxdecomp{
    // Global handle for the decomposition operations
    cudecompHandle_t handle;

    void init(){
        CHECK_CUDECOMP_EXIT(cudecompInit(&handle, MPI_COMM_WORLD));
    };

    
    void finalize(){
        CHECK_CUDECOMP_EXIT(cudecompFinalize(handle));
    };
}


PYBIND11_MODULE(python_example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: python_example
        .. autosummary::
           :toctree: _generate
           init
           finalize
    )pbdoc";

    m.def("init", &jaxdecomp::init, R"pbdoc(
        Initialize the library.
    )pbdoc");

    m.def("finalize", &jaxdecomp::finalize, R"pbdoc(
        Finalize the library.
    )pbdoc");
}
