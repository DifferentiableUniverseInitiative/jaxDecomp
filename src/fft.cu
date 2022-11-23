#include <algorithm>
#include <array>
#include <complex>
#include <numeric>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <cuda/std/complex>
#include <cudecomp.h>

#include "fft.h"
#include "checks.h"

namespace jaxdecomp{

    static cufftType get_cufft_type_r2c(float) { return CUFFT_R2C; }
    static cufftType get_cufft_type_c2r(float) { return CUFFT_C2R; }
    static cufftType get_cufft_type_c2c(float) { return CUFFT_C2C; }
    static cufftType get_cufft_type_r2c(double) { return CUFFT_D2Z; }
    static cufftType get_cufft_type_c2r(double) { return CUFFT_Z2D; }
    static cufftType get_cufft_type_c2c(double) { return CUFFT_Z2Z; }
    static cudecompDataType_t get_cudecomp_datatype(cuda::std::complex<float>) { return CUDECOMP_FLOAT_COMPLEX; }
    static cudecompDataType_t get_cudecomp_datatype(cuda::std::complex<double>) { return CUDECOMP_DOUBLE_COMPLEX; }

    // These functions are adapted from the cuDecomp benchmark code
    template<typename real_t> void fft3d(cudecompHandle_t handle,
              cudecompGridDescConfig_t config,
              void** buffers,
              bool forward,
              bool r2c){

        using complex_t = cuda::std::complex<real_t>;

        /* Setting up cuDecomp grid specifications
        */
        int gx = config.gdims[0];
        int gy = config.gdims[1];
        int gz = config.gdims[2];
        cudecompGridDesc_t grid_desc_c; // complex grid
        cudecompGridDesc_t grid_desc_r; // real grid (only used in r2c case)
        if(r2c){
            std::array<int32_t, 3> gdim_c{gx / 2 + 1, gy, gz};
            config.gdims[0] = gdim_c[0];
            config.gdims[1] = gdim_c[1];
            config.gdims[2] = gdim_c[2];
            CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc_c, &config, nullptr));

            std::array<int32_t, 3> gdim_r{(gx / 2 + 1) * 2, gy, gz}; // with padding
            config.gdims[0] = gdim_r[0];
            config.gdims[1] = gdim_r[1];
            config.gdims[2] = gdim_r[2];
            CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc_r, &config, nullptr));

        }else{
            CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc_c, &config, nullptr));
        }

        /* Extracting pencil information
        */
        cudecompPencilInfo_t pinfo_x_r;
        if (r2c){
            // Get x-pencil information (real)
            CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc_r, &pinfo_x_r, 0, nullptr));
        };

        // Get x-pencil information (complex)
        cudecompPencilInfo_t pinfo_x_c;
        CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_x_c, 0, nullptr));

        // Get y-pencil information (complex)
        cudecompPencilInfo_t pinfo_y_c;
        CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_y_c, 1, nullptr));

        // Get z-pencil information (complex)
        cudecompPencilInfo_t pinfo_z_c;
        CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc_c, &pinfo_z_c, 2, nullptr));

        // Get workspace size
        int64_t num_elements_work_c;
        CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc_c, &num_elements_work_c));

        /**
         * Setting up cuFFT
         * Starting with everything for the x axis
        */
        bool slab_xy = false;
        bool slab_yz = false;
        cufftHandle cufft_plan_r2c_x;
        cufftHandle cufft_plan_c2r_x;
        cufftHandle cufft_plan_c2c_x;
        size_t work_sz_r2c_x;
        size_t work_sz_c2r_x;
        size_t work_sz_c2c_x;
        
        if(r2c){
            // x-axis real-to-complex
            CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_r2c_x));
            CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_r2c_x, 0));
            

            // x-axis complex-to-real
            CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_c2r_x));
            CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_c2r_x, 0));


            if (config.pdims[0] == 1) {
                // x-y slab: use 2D FFT
                slab_xy = true;
                std::array<int, 2> n{gx, gy};
                CHECK_CUFFT_EXIT(cufftMakePlanMany(
                    cufft_plan_r2c_x, 2, n.data(), nullptr, 1, pinfo_x_r.shape[0] * pinfo_x_r.shape[1], nullptr, 1,
                    pinfo_x_c.shape[0] * pinfo_x_c.shape[1], get_cufft_type_r2c(real_t(0)), pinfo_x_r.shape[2], &work_sz_r2c_x));
                CHECK_CUFFT_EXIT(cufftMakePlanMany(
                    cufft_plan_c2r_x, 2, n.data(), nullptr, 1, pinfo_x_c.shape[0] * pinfo_x_c.shape[1], nullptr, 1,
                    pinfo_x_r.shape[0] * pinfo_x_r.shape[1], get_cufft_type_c2r(real_t(0)), pinfo_x_c.shape[2], &work_sz_c2r_x));
            } else {
                CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_r2c_x, gx, get_cufft_type_r2c(real_t(0)),
                                                pinfo_x_r.shape[1] * pinfo_x_r.shape[2], &work_sz_r2c_x));
                CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_c2r_x, gx, get_cufft_type_c2r(real_t(0)),
                                                pinfo_x_c.shape[1] * pinfo_x_c.shape[2], &work_sz_c2r_x));
            }
        }else{
            // x-axis complex-to-complex
            CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_c2c_x));
            CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_c2c_x, 0));

            if (config.pdims[0] == 1) {
                // x-y slab: use 2D FFT
                slab_xy = true;
                std::array<int, 2> n{gy, gx};
                CHECK_CUFFT_EXIT(cufftMakePlanMany(
                    cufft_plan_c2c_x, 2, n.data(), nullptr, 1, pinfo_x_c.shape[0] * pinfo_x_c.shape[1], nullptr, 1,
                    pinfo_x_c.shape[0] * pinfo_x_c.shape[1], get_cufft_type_c2c(real_t(0)), pinfo_x_c.shape[2], &work_sz_c2c_x));
            } else {
                CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_c2c_x, gx, get_cufft_type_c2c(real_t(0)),
                                                pinfo_x_c.shape[1] * pinfo_x_c.shape[2], &work_sz_c2c_x));
            }
        }

        // y-axis complex-to-complex
        cufftHandle cufft_plan_c2c_y;
        CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_c2c_y));
        CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_c2c_y, 0));
        size_t work_sz_c2c_y;
        if (config.pdims[1] == 1) {
            // y-z slab: use 2D FFT
            slab_yz = true;
            if (config.transpose_axis_contiguous[1]) {
            std::array<int, 2> n{gz, gy};
            CHECK_CUFFT_EXIT(cufftMakePlanMany(
                cufft_plan_c2c_y, 2, n.data(), nullptr, 1, pinfo_y_c.shape[0] * pinfo_y_c.shape[1], nullptr, 1,
                pinfo_y_c.shape[0] * pinfo_y_c.shape[1], get_cufft_type_c2c(real_t(0)), pinfo_y_c.shape[2], &work_sz_c2c_y));
            } else {
            // Note: In this case, both slab dimensions are strided, leading to slower performance using
            // 2D FFT. Run 1D + 1D instead.
            slab_yz = false;
            CHECK_CUFFT_EXIT(cufftMakePlanMany(cufft_plan_c2c_y, 1, &gy /* unused */, &gy, pinfo_y_c.shape[0], 1, &gy,
                                                pinfo_y_c.shape[0], 1, get_cufft_type_c2c(real_t(0)), pinfo_y_c.shape[0],
                                                &work_sz_c2c_y));
            }
        } else {
            if (config.transpose_axis_contiguous[1]) {
            CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_c2c_y, gy, get_cufft_type_c2c(real_t(0)),
                                            pinfo_y_c.shape[1] * pinfo_y_c.shape[2], &work_sz_c2c_y));
            } else {
            CHECK_CUFFT_EXIT(cufftMakePlanMany(cufft_plan_c2c_y, 1, &gy /* unused */, &gy, pinfo_y_c.shape[0], 1, &gy,
                                                pinfo_y_c.shape[0], 1, get_cufft_type_c2c(real_t(0)), pinfo_y_c.shape[0],
                                                &work_sz_c2c_y));
            }
        }

        // z-axis complex-to-complex
        cufftHandle cufft_plan_c2c_z;
        CHECK_CUFFT_EXIT(cufftCreate(&cufft_plan_c2c_z));
        CHECK_CUFFT_EXIT(cufftSetAutoAllocation(cufft_plan_c2c_z, 0));
        size_t work_sz_c2c_z;
        if (config.transpose_axis_contiguous[2]) {
            CHECK_CUFFT_EXIT(cufftMakePlan1d(cufft_plan_c2c_z, gz, get_cufft_type_c2c(real_t(0)),
                                            pinfo_z_c.shape[1] * pinfo_z_c.shape[2], &work_sz_c2c_z));
        } else {
            CHECK_CUFFT_EXIT(cufftMakePlanMany(cufft_plan_c2c_z, 1, &gz /* unused */, &gz,
                                            pinfo_z_c.shape[0] * pinfo_z_c.shape[1], 1, &gz,
                                            pinfo_z_c.shape[0] * pinfo_z_c.shape[1], 1, get_cufft_type_c2c(real_t(0)),
                                            pinfo_z_c.shape[0] * pinfo_z_c.shape[1], &work_sz_c2c_z));
        }

        // Allocate workspace
        int64_t work_sz_decomp, work_sz_cufft, work_sz;
        if(r2c){
            work_sz_decomp = 2 * num_elements_work_c * sizeof(real_t);
            work_sz_cufft = std::max(std::max(work_sz_r2c_x, std::max(work_sz_c2c_y, work_sz_c2c_z)), work_sz_c2r_x);
            work_sz = std::max(work_sz_decomp, work_sz_cufft);
        }else{
            work_sz_decomp = 2 * num_elements_work_c * sizeof(real_t);
            work_sz_cufft = std::max(work_sz_c2c_x, std::max(work_sz_c2c_y, work_sz_c2c_z));
            work_sz = std::max(work_sz_decomp, work_sz_cufft);
        }
        void* work = malloc(work_sz);
        void* work_d;
        CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc_c, reinterpret_cast<void**>(&work_d), work_sz));

        // Assign cuFFT work area
        complex_t* work_c_d = static_cast<complex_t*>(work_d);
        if(r2c){
            CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_r2c_x, work_d));
            CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_c2r_x, work_d));
        }else{
            CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_c2c_x, work_d));
        }
        CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_c2c_y, work_d));
        CHECK_CUFFT_EXIT(cufftSetWorkArea(cufft_plan_c2c_z, work_d));

        // Run 3D FFT sequence
        real_t* data_r_d;
        complex_t* data_c_d;
        // The logic below is to figure out which buffer to use depending on the fft case
        if(r2c){
            if(forward){
                data_r_d = static_cast<real_t*>(buffers[0]);
                data_c_d = static_cast<complex_t*>(buffers[1]);
            }else{
                data_r_d = static_cast<real_t*>(buffers[1]);
                data_c_d = static_cast<complex_t*>(buffers[0]);
            }
        }else{
            data_r_d = static_cast<real_t*>(buffers[0]);
            data_c_d = static_cast<complex_t*>(buffers[0]);
        }
        complex_t* input = data_c_d;
        complex_t* output = data_c_d;
        real_t* input_r = data_r_d;
        real_t* output_r = data_r_d;

        /*
         * Perform FFT along x and transpose array
         * It assumes that x is initially not distributed.
        */
        if(forward){
            if(r2c){
                CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_r2c_x, input_r, input, CUFFT_FORWARD));
            }else{
                CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_x, input, input, CUFFT_FORWARD));
            }
            CHECK_CUDECOMP_EXIT(cudecompTransposeXToY(handle, grid_desc_c, input, output, work_c_d,
                                        get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, 0));

            /* 
            * Perform FFT along y and transpose
            */
            if (!slab_xy) {
            if (config.transpose_axis_contiguous[1] || slab_yz) {
                CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_y, output, output, CUFFT_FORWARD));
            } else {
                for (int i = 0; i < pinfo_y_c.shape[2]; ++i) {
                CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_y, output + i * (pinfo_y_c.shape[0] * pinfo_y_c.shape[1]),
                                            output + i * (pinfo_y_c.shape[0] * pinfo_y_c.shape[1]), CUFFT_FORWARD));
                }
            }
            }
            // For y-z slab case, no need to perform yz transposes or z-axis FFT
            if (!slab_yz) {
            CHECK_CUDECOMP_EXIT(cudecompTransposeYToZ(handle, grid_desc_c, input, output, work_c_d,
                                        get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, 0));
            }

            /*
            * Perform FFT along z
            */
            CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_z, output, output, CUFFT_FORWARD));
        }else{
            /* Inverse FFT along z and transpose array
            */
            CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_z, input, output, CUFFT_INVERSE));

            if (!slab_yz) {
            CHECK_CUDECOMP_EXIT(cudecompTransposeZToY(handle, grid_desc_c, input, output, work_c_d,
                                                        get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, 0));
            }

            /* Inverse FFT along y and transpose array
            */
            if (!slab_xy) {
            if (config.transpose_axis_contiguous[1] || slab_yz) {
                CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_y, output, output, CUFFT_INVERSE));
            } else {
                for (int i = 0; i < pinfo_y_c.shape[2]; ++i) {
                CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_y, output + i * (pinfo_y_c.shape[0] * pinfo_y_c.shape[1]),
                                            output + i * (pinfo_y_c.shape[0] * pinfo_y_c.shape[1]), CUFFT_INVERSE));
                }
            }
            }

            CHECK_CUDECOMP_EXIT(cudecompTransposeYToX(handle, grid_desc_c, input, output, work_c_d,
                                              get_cudecomp_datatype(complex_t(0)), nullptr, nullptr, 0));

            /* Inverse FFT along x and we are back to the real world
            */
           if(r2c){
                CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2r_x, output, output_r, CUFFT_INVERSE));
           }else{
                CHECK_CUFFT_EXIT(cufftXtExec(cufft_plan_c2c_x, output, output, CUFFT_INVERSE));
           }
        }

        // TODO: Check if we need to clean the cufft stuff as well

        CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc_c, work_d));
        if(r2c){
            CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc_r));
        }
        CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc_c));
    };

    // Declare specialisations for float and double
    template void fft3d<float>(cudecompHandle_t handle,
              cudecompGridDescConfig_t config,
              void** buffers,
              bool forward,
              bool r2c);
    template void fft3d<double>(cudecompHandle_t handle,
              cudecompGridDescConfig_t config,
              void** buffers,
              bool forward,
              bool r2c);
};
