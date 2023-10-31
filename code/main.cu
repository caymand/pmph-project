#include <cstdio>
#include <mma.h>
#include "matmul.cuh"
#include "helpers.h"
#include "goldenSeq.h"
#include "matmul-tensor.cuh"

#define WARP_SIZE 32
/*
* block_tiles_m - how many elements to work on in the m direction
* block_tiles_n - how many elements to work on in the n direction
* block_tiles_k - how many elements to work on in the k direction
*/
template <typename elmT, int threads_per_block, int block_tiles_m, int block_tiles_n, int block_tiles_k, int wmma_n, int wmma_m, int wmma_k, typename elmAccT = elmT>
unsigned benchmark_tiled_tensor_mmm(
        elmT *A_device,
        elmT *B_device,
        elmAccT *ResMat_device,
        int m,
        int n,
        int k,
        unsigned n_runs)
{
    constexpr int copies_per_thread_A = 0, copies_per_thread_B = 0;

    // Let block work on block_tiles * wmma elements.
    // there are n elements on the x direction and we know each thread works on block_tiles_n
    int dimx = ceil(((float) n)/(wmma_n * block_tiles_n)); 
    int dimy = ceil( ((float) m)/(wmma_m * block_tiles_m));
    
    dim3 grid(dimx, dimy, 1);
    // dim3 block(threads_per_block, 1, 1); // 1D block of 256 elements
    /* Okay so what do we want? Each mm will be done by the entire warp and works warp level.
    So whatever we want to tile for should be multiple of the warp size.
    Here we say that the block should compute block_tiles_m x block_tiles_n tensor mm.

    This also works for the grid specification, since we tile so that each warp computes
    a wmma_m x wmma_n result, and we use block_tiles_m x block_tiles_n warps in the block.
    */
    dim3 block(block_tiles_n * WARP_SIZE, block_tiles_m, 1);

    TimeMeasurement t;

    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiledTensor<
            elmAccT, elmT, wmma_m, wmma_n, wmma_k, block_tiles_m, block_tiles_n, 
            block_tiles_k, copies_per_thread_A, copies_per_thread_B>
            <<<grid, block>>>(A_device, B_device, ResMat_device, m, n, n);
    }
    cudaDeviceSynchronize();
    t.stop();
    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());

    return t.elapsed();
}
template <typename elmT, int tile_size, int reg_size>
unsigned benchmark_tiled_mmm(
        elmT *A_device,
        elmT *B_device,
        elmT *ResMat_device,
        int height_A,
        int width_B,
        int width_A,
        unsigned n_runs)
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
template <typename elmT, int tile_size, int reg_size, int MatDim, bool use_tensor_cores, typename elmAccT = elmT>
//int reg_size, int n_runs = 1, int MatDim = 2, class accT = elmT>
RandomMatrix<elmAccT, MatDim>* run_mmm_kernel(
        int height_A,
        int width_B,
        int width_A,
        RandomMatrix<elmT, MatDim> &A,
        RandomMatrix<elmT, MatDim> &B,
        unsigned n_runs)
{
    double total_ops = 2.0f * width_B * width_A * height_A;
    auto ResMat = new RandomMatrix<elmAccT, MatDim>;
    // This took me like 2 hours to fix...
    ResMat->template fill<float_range>(height_A, width_B);

    constexpr int threads_per_block = 256;
    constexpr int block_tiles_m = 1;
    constexpr int block_tiles_n = 1;
    constexpr int block_tiles_k = 1;

    constexpr int wmma_m = 16;
    constexpr int wmma_n = 16;
    constexpr int wmma_k = 16;

    auto A_device = A.to_gpu();
    auto B_device = B.to_gpu();
    auto ResMat_device = ResMat->to_gpu();
    unsigned total_elapsed;
    if constexpr(use_tensor_cores) {
        total_elapsed = benchmark_tiled_tensor_mmm<
            elmT, threads_per_block, block_tiles_m, block_tiles_n, block_tiles_k, 
            wmma_m, wmma_n, wmma_k, elmAccT>
            (
                A_device, B_device, ResMat_device, height_A, width_B, width_A, n_runs
        );
    }
    else {
        total_elapsed = benchmark_tiled_mmm<elmT, tile_size, reg_size>(
                A_device, B_device, ResMat_device, height_A, width_B, width_A, n_runs
        );
    }
    cudaMemcpy(ResMat->to_cpu(), ResMat_device, ResMat->flatSize() * sizeof(elmAccT), cudaMemcpyDeviceToHost);
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
    constexpr int width_A = 16 * 256;// Multiple of 8 to allign with frame leading dimension
    constexpr int height_A = 16 * 256;// Multiple of 8 to allign with frame leading dimension
    constexpr int width_B = 16 * 256;// Multiple of 8 to allign with frame leading dimension
    unsigned n_runs = 100;
    // Tiled GPU verion
    // TODO: this fails when the type is float since it is not supported for wmma
    // and the templated function is still created
    RandomMatrix<float, 2> A;
    RandomMatrix<float, 2> B;
    TimeMeasurement t;
    A.fill<float_range>(height_A, width_A);
    B.fill<float_range>(width_A, width_B);

    std::cout << "-----" << std::endl;
    std::cout << "Running GPU register tiled version" << std::endl;
    std::cout << "Dry run" << std::endl;
    run_mmm_kernel<float, 16, 5, 2, false>(
        height_A, width_B, width_A, A, B, 1
    );
    std::cout << "Average run of: " << n_runs << std::endl;
    RandomMatrix<float, 2> *C = run_mmm_kernel<float, 16, 5, 2, false>(
        height_A, width_B, width_A, A, B, n_runs
    );
    RandomMatrix<float, 2> target_res;
    target_res.fill_from(*C, height_A * width_B);
    std::cout << "-----" << std::endl;

    // GPU version
    RandomMatrix<half, 2> A_half;
    RandomMatrix<half, 2> B_half;
    A_half.fill_from(A, height_A, width_A);
    B_half.fill_from(B, width_A, width_B);
    constexpr int block_tile_size = 5; // TODO: calculate based on amount of shared memory
    
    std::cout << "-----" << std::endl;
    std::cout << "Running GPU tensor version" << std::endl;
    std::cout << "Dry run" << std::endl;
    run_mmm_kernel<half, 16, 5, 2, true, float>(
        height_A, width_A, width_B, A_half, B_half, 1
    );
    std::cout << "Average run after: " << n_runs << " runs" << std::endl;
    RandomMatrix<float, 2> *GPU_res_tensor_half = run_mmm_kernel<half, 16, 5, 2, true, float>(
        height_A, width_A, width_B, A_half, B_half, n_runs
    );
    std::cout << "-----" << std::endl;

    RandomMatrix<float, 2> GPU_res_tensor;
    GPU_res_tensor.fill_from(*GPU_res_tensor_half, height_A, width_B);

    Validator<float> validator(target_res.to_cpu(), GPU_res_tensor.to_cpu(), height_A * width_B);
    // validator.setEps(0.000005); // original used by cosmin
    validator.setEps(0.0005);
    validator.validate();
    delete GPU_res_tensor_half;

    return 0;
}
