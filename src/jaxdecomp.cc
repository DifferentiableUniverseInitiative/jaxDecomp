#include <pybind11/pybind11.h>
#include <cudecomp.h>
#include <mpi.h>
#include "checks.h"
#include "helpers.h"

namespace py = pybind11;

namespace jaxdecomp {

    // Custom data types
    struct decompDescriptor_t {
        // gdim parameters
        std::int64_t nx;
        std::int64_t ny;
        std::int64_t nz;
    };

    // Global handle for the decomposition operations
    cudecompHandle_t handle;
    
    void init(){
        CHECK_CUDECOMP_EXIT(cudecompInit(&handle, MPI_COMM_WORLD));
    };
    
    void finalize(){
        CHECK_CUDECOMP_EXIT(cudecompFinalize(handle));
    };

    /// XLA interface ops

    /* Wrapper to cudecompTransposeXToY 
    Transpose data from X-axis aligned pencils to a Y-axis aligned pencils.
    */
    void transposeXtoY(cudaStream_t stream, void** buffers,
                       const char* opaque, size_t opaque_len){

        void* data_d = buffers[0];
        
        const decompDescriptor_t &desc = *UnpackDescriptor<decompDescriptor_t>(opaque, opaque_len);
        
        // Create cuDecomp grid descriptor
        cudecompGridDescConfig_t config;
        CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&config));

        config.pdims[0] = 2;//desc.p_cols; // P_rows
        config.pdims[1] = 1;//desc.p_rows; // P_cols

        config.gdims[0] = desc.nx; // X
        config.gdims[1] = desc.ny; // Y
        config.gdims[2] = desc.nz; // Z

        // Setting default communication backend
        config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_MPI_P2P;
        config.halo_comm_backend = CUDECOMP_HALO_COMM_MPI;

        config.transpose_axis_contiguous[0] = true;
        config.transpose_axis_contiguous[1] = true;
        config.transpose_axis_contiguous[2] = true;

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

        // Transpose from Y-pencils to Z-pencils.
        CHECK_CUDECOMP_EXIT(
            cudecompTransposeYToZ(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_FLOAT, nullptr, nullptr, 0));

        CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, transpose_work_d));

        CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));
    } 

    // Utility to export ops to XLA
    pybind11::dict Registrations() {
        pybind11::dict dict;
        dict["transpose_x_y"] = EncapsulateFunction(transposeXtoY);
        return dict;
    }
}

PYBIND11_MODULE(_jaxdecomp, m) {
    m.def("init", &jaxdecomp::init, R"pbdoc(
        Initialize the library.
    )pbdoc");

    m.def("finalize", &jaxdecomp::finalize, R"pbdoc(
        Finalize the library.
    )pbdoc");

    m.def("registrations", &jaxdecomp::Registrations,
         R"pbdoc(
        Runs a transpose operation
    )pbdoc");

    m.def("build_decomp_descriptor", 
    []( std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz        
    ) { return PackDescriptor(jaxdecomp::decompDescriptor_t{nx, ny, nz}); }); 
}
