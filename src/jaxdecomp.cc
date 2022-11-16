#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

    struct pencilInfo_t {
        std::array<int32_t, 3> shape;
        std::array<int32_t, 3> lo;
        std::array<int32_t, 3> hi;
        std::array<int32_t, 3> order;
        std::array<int32_t, 3> halo_extents;
        int64_t size;
    };

    struct decompConfig_t {
        int32_t pdims[2]; // Dimensions of the processor grid
        cudecompTransposeCommBackend_t transpose_comm_backend;
        cudecompHaloCommBackend_t halo_comm_backend;
    } global_config;

    // Global handle for the decomposition operations
    cudecompHandle_t handle;
    
    void init(int32_t pdim_x, int32_t pdim_y){
        // TODO: include parameters for the mesh and/or autotune
        CHECK_CUDECOMP_EXIT(cudecompInit(&handle, MPI_COMM_WORLD));
        global_config.pdims[0] = pdim_x;
        global_config.pdims[1] = pdim_y;
        global_config.transpose_comm_backend = CUDECOMP_TRANSPOSE_COMM_NCCL;
        global_config.halo_comm_backend = CUDECOMP_HALO_COMM_MPI;
    };
    
    void finalize(){
        CHECK_CUDECOMP_EXIT(cudecompFinalize(handle));
    };

    pencilInfo_t getPencilInfo(int64_t nx, int64_t ny, int64_t nz){

        // Create cuDecomp grid descriptor
        cudecompGridDescConfig_t config;
        CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&config));

        config.pdims[0] = global_config.pdims[0];
        config.pdims[1] = global_config.pdims[1];

        config.gdims[0] = nx; // X
        config.gdims[1] = ny; // Y
        config.gdims[2] = nz; // Z

        // config.transpose_axis_contiguous[0] = true;
        // config.transpose_axis_contiguous[1] = true;
        // config.transpose_axis_contiguous[2] = true;

        // Create the grid description
        cudecompGridDesc_t grid_desc;
        CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));

        cudecompPencilInfo_t pencil_info;
        CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pencil_info, 2, nullptr));
        
        pencilInfo_t result;
        for(int i=0; i<3; i++){
            result.hi[i] = pencil_info.hi[i];
            result.lo[i] = pencil_info.lo[i];
            result.halo_extents[i] = pencil_info.halo_extents[i];
            result.order[i] = pencil_info.order[i];
            result.shape[i] = pencil_info.shape[i];
        }
        result.size = pencil_info.size;

        return result;
    };

    /// XLA interface ops

    /* Wrapper to cudecompTransposeXToY 
    Transpose data from X-axis aligned pencils to a Y-axis aligned pencils.
    */
    void transposeXtoY(cudaStream_t stream, void** buffers,
                       const char* opaque, size_t opaque_len){

        void* data_d = buffers[0]; // In place operations, so only one buffer
        int32_t* gdims = (int32_t *) (buffers[1]);
        
        const decompDescriptor_t &desc = *UnpackDescriptor<decompDescriptor_t>(opaque, opaque_len);
        
        // Create cuDecomp grid descriptor
        cudecompGridDescConfig_t config;
        CHECK_CUDECOMP_EXIT(cudecompGridDescConfigSetDefaults(&config));

        config.pdims[0] = global_config.pdims[0];
        config.pdims[1] = global_config.pdims[1];

        config.gdims[0] = desc.nx; // X
        config.gdims[1] = desc.ny; // Y
        config.gdims[2] = desc.nz; // Z

        // Setting default communication backend
        config.transpose_comm_backend = global_config.transpose_comm_backend;
        config.halo_comm_backend = global_config.halo_comm_backend;

        // config.transpose_axis_contiguous[0] = true;
        // config.transpose_axis_contiguous[1] = true;
        // config.transpose_axis_contiguous[2] = true;

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

        CHECK_CUDECOMP_EXIT(
            cudecompTransposeXToY(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_FLOAT, nullptr, nullptr, 0));

        CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, transpose_work_d));

        CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));
    }

    // Utility to export ops to XLA
    py::dict Registrations() {
        py::dict dict;
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

    m.def("get_pencil_info", &jaxdecomp::getPencilInfo);

    m.def("registrations", &jaxdecomp::Registrations,
         R"pbdoc(
        Runs a transpose operation
    )pbdoc");

    m.def("build_decomp_descriptor", 
    []( std::int64_t nx,
        std::int64_t ny,
        std::int64_t nz        
    ) { return PackDescriptor(jaxdecomp::decompDescriptor_t{nx, ny, nz}); }); 

    py::class_<jaxdecomp::pencilInfo_t> pencil_info(m, "PencilInfo");
    pencil_info.def(py::init<>())
                .def_readonly("shape", &jaxdecomp::pencilInfo_t::shape)
                .def_readonly("lo", &jaxdecomp::pencilInfo_t::lo)
                .def_readonly("hi", &jaxdecomp::pencilInfo_t::hi)
                .def_readonly("order", &jaxdecomp::pencilInfo_t::order)
                .def_readonly("halo_extents", &jaxdecomp::pencilInfo_t::halo_extents)
                .def_readonly("size", &jaxdecomp::pencilInfo_t::size);
}
