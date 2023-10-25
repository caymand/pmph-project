#include <cstdio>
#include <mma.h>
#include "matmul.cuh"
#include "helpers.h"
#include "goldenSeq.h"
#include "matmul-tensor.cuh"













template <typename elmT, int block_tile_size, int n_runs, int wmma_n, int wmma_m, int wmma_k, typename elmAccT = elmT>
unsigned benchmark_tiled_tensor_mmm(
        elmT *A_device,
        elmT *B_device,
        elmAccT *ResMat_device,
        int height_A,
        int width_B,
        int width_A)
{

    constexpr int threads_per_block = 256;
    constexpr int block_tiles_m = 2;
    constexpr int block_tiles_n = 2;
    constexpr int block_tiles_k = 4;

    constexpr unsigned int A_loc_m = block_tiles_m * wmma_m;
    constexpr unsigned int A_loc_k = wmma_k * (block_tiles_k + 1);

    // remapping (a slice of) B to shared memory
    constexpr unsigned int B_loc_k = block_tiles_k * wmma_k;
    constexpr unsigned int B_loc_n = wmma_n * (block_tiles_n + 1);

//    TODO: ensure A_loc_k and B_loc_n are multiples of warpSize

    constexpr int copies_per_thread_A = (A_loc_m * A_loc_k + threads_per_block) / threads_per_block;
    constexpr int copies_per_thread_B = (B_loc_k * B_loc_n + threads_per_block) / threads_per_block;

    int dimx = ceil( ((float) n * m)/(block_tiles_m * wmma_m * block_tiles_n * wmma_n));

    dim3 grid(dimx, 1, 1);
    dim3 block(threads_per_block, 1, 1);


    TimeMeasurement t;


    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiledTensor<elmAccT, elmT, wmma_m, wmma_n, wmma_k, block_tile_size><<<grid, block>>>(
                A_device, B_device, ResMat_device, height_A, width_B, width_B
        );
    }
    cudaDeviceSynchronize();
    t.stop();
    // Check if kernel launch was successfull
    if (!gpuAssert(cudaPeekAtLastError())) {
        return 0;
    }
    return t.elapsed();
}
template <typename elmT, int tile_size, int n_runs, int reg_size>
unsigned benchmark_tiled_mmm(
        elmT *A_device,
        elmT *B_device,
        elmT *ResMat_device,
        int height_A,
        int width_B,
        int width_A)
{
    int dimy = ceil( ((float) width_B)/(tile_size * reg_size));
    int dimx = ceil( ((float) height_A)/(tile_size * reg_size));
    TimeMeasurement t;
    dim3 grid(dimx, dimy, 1);
    dim3 block(16, 16, 1);

    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiled<elmT, tile_size, reg_size, tile_size, reg_size, tile_size><<<grid, block>>>(
                A_device, B_device, ResMat_device, height_A, width_B, width_A);
    }
    cudaDeviceSynchronize();
    t.stop();
    // Check if kernel launch was successfull
    if (!gpuAssert(cudaPeekAtLastError())) {
        return 0;
    }
    return t.elapsed();
}

// Expects A to have shape K x K and B to have K x N
template <typename elmT, int tile_size, int reg_size, int MatDim, int n_runs, bool use_tensor_cores>
//int reg_size, int n_runs = 1, int MatDim = 2, class accT = elmT>
RandomMatrix<elmT, MatDim>* run_mmm_kernel(
        int height_A,
        int width_B,
        int width_A,
        RandomMatrix<elmT, MatDim> &A,
        RandomMatrix<elmT, MatDim> &B)
{
    double total_ops = 2.0f * width_B * width_A * height_A;
    auto ResMat = new RandomMatrix<elmT, MatDim>;
    // This took me like 2 hours to fix...
    ResMat->template fill<float_range>(height_A, width_B);

    auto A_device = A.to_gpu();
    auto B_device = B.to_gpu();
    auto ResMat_device = ResMat->to_gpu();
    unsigned total_elapsed;
    if constexpr(use_tensor_cores) {
        total_elapsed = benchmark_tiled_tensor_mmm<elmT, tile_size, n_runs, 16, 16, 16>(
                A_device, B_device, ResMat_device, height_A, width_B, width_A
        );
    }
    else {
        total_elapsed = benchmark_tiled_mmm<elmT, tile_size, reg_size, n_runs>(
                A_device, B_device, ResMat_device, height_A, width_B, width_A
        );
    }

    cudaMemcpy(ResMat->to_cpu(), ResMat_device, ResMat->flatSize() * sizeof(elmT), cudaMemcpyDeviceToHost);
    cudaFree(A_device); cudaFree(B_device); cudaFree(ResMat_device);
    gpuAssert( cudaPeekAtLastError() );


    if (!total_elapsed) {
        printf("Kernel launch failed\n");
        memset(ResMat->to_cpu(), 0, height_A * width_B);
    } else {
        printGFlops(total_elapsed, total_ops * n_runs);
    }
    return ResMat;
}





int main(int argc, char * argv[]) {
    constexpr int width_A = 16 * 5 * 12;// Multiple of 8 to allign with frame leading dimension
    constexpr int height_A = 16 * 5 * 12;// Multiple of 8 to allign with frame leading dimension
    constexpr int width_B = 16 * 5 * 12;// Multiple of 8 to allign with frame leading dimension

    // Tiled GPU verion
    // TODO: this fails when the type is float since it is not supported for wmma
    // and the templated function is still created
    RandomMatrix<float, 2> A;
    RandomMatrix<float, 2> B;
    RandomMatrix<float, 2> CPU_res;
    TimeMeasurement t;
    A.fill<float_range>(height_A, width_A);
    B.fill<float_range>(width_A, width_B);
    CPU_res.fill<float_range>(height_A, width_B);
    std::cout << "Running CPU version" << std::endl;
    goldenSeq<float>(A.to_cpu(), B.to_cpu(), CPU_res.to_cpu(), height_A, width_B, width_A);
    // RandomMatrix<float, 2> *CPU = run_mmm_kernel<float, 16, 5, 2, 1, false>(
    //     height_A, width_B, width_A, A, B
    // );
    // RandomMatrix<float, 2> GPU_res_tiled;
    // GPU_res_tiled.fill_from(*GPU_res_tiled_half, height_A * width_B);

    // GPU version
    RandomMatrix<half, 2> A_half;
    RandomMatrix<half, 2> B_half;
    A_half.fill_from(A, height_A, width_A);
    B_half.fill_from(B, width_A, width_B);
    constexpr int block_tile_size = 5; // TODO: calculate based on amount of shared memory
    std::cout << "Running GPU tensor version" << std::endl;
    RandomMatrix<half, 2> *GPU_res_tensor_half = run_mmm_kernel<half, 5, 5, 2, 1, true>(
        height_A, width_A, width_B, A_half, B_half
    );

    RandomMatrix<float, 2> GPU_res_tensor;
    GPU_res_tensor.fill_from(*GPU_res_tensor_half, height_A, width_B);

    Validator<float> validator(CPU_res.to_cpu(), GPU_res_tensor.to_cpu(), height_A * width_B);
    // validator.setEps(0.000005);
    validator.setEps(0.05);
    validator.validate();
    delete GPU_res_tensor_half;
    // delete GPU_res_tiled_half;

    return 0;
}
