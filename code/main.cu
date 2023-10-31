#include <cstdio>
#include <mma.h>
#include "matmul.cuh"
#include "helpers.h"
#include "goldenSeq.h"
#include "matmul-tensor.cuh"
#include "cuda_fp16.h"
#include <cassert>


#define WARP_SIZE 32



template <typename elmT, typename elmAccT = elmT>
long int benchmark_tiled_tensor_mmm(
        int n_runs,
        elmT *A_device,
        elmT *B_device,
        elmAccT *ResMat_device,
        int m,
        int n,
        int k)
{
//    TODO: calculate maximum possible block_tiles_k based on shared memory size? similarly calculate warp_tiles_k based on availible registers
//    TODO: calculate m and n dimensions based on optimal block size?
//    constexpr int wmma_m = 16;
//    constexpr int wmma_n = 16;
//    constexpr int wmma_k = 16;
//
//    constexpr int warp_tiles_m = 1;
//    constexpr int warp_tiles_n = 1;
//    constexpr int warp_tiles_k = 1;
//
//    constexpr int block_tiles_m = 2;
//    constexpr int block_tiles_n = 2;
//    constexpr int block_tiles_k = 2;

// TODO: also set type using compiler options
// Set constants using compiler options
#ifdef WMMA_M
    constexpr int wmma_m = WMMA_M;
#else
    constexpr int wmma_m = 16;
#endif
#ifdef WMMA_N
    constexpr int wmma_n = WMMA_N;
#else
    constexpr int wmma_n = 16;
#endif
#ifdef WMMA_K
    constexpr int wmma_k = WMMA_K;
#else
    constexpr int wmma_k = 16;
#endif
#ifdef WARP_TILES_M
    constexpr int warp_tiles_m = WARP_TILES_M;
#else
    constexpr int warp_tiles_m = 2;
#endif
#ifdef WARP_TILES_N
    constexpr int warp_tiles_n = WARP_TILES_N;
#else
    constexpr int warp_tiles_n = 2;
#endif
#ifdef WARP_TILES_K
    constexpr int warp_tiles_k = WARP_TILES_K;
#else
    constexpr int warp_tiles_k = 2;
#endif
#ifdef BLOCK_TILES_M
    constexpr int block_tiles_m = BLOCK_TILES_M;
#else
    constexpr int block_tiles_m = 2;
#endif
#ifdef BLOCK_TILES_N
    constexpr int block_tiles_n = BLOCK_TILES_N;
#else
    constexpr int block_tiles_n = 2;
#endif
#ifdef BLOCK_TILES_K
    constexpr int block_tiles_k = BLOCK_TILES_K;
#else
    constexpr int block_tiles_k = 2;
#endif



    constexpr unsigned int threads_per_block = block_tiles_m * block_tiles_n * WARP_SIZE;
    printf("Threads used: %d\n", threads_per_block);
    assert(threads_per_block <= 1024);
    //    Assumes num_warps >= block_tiles_m * block_tiles_n, i.e. all block tiles are handled by a warp
    assert(threads_per_block / WARP_SIZE >= block_tiles_m * block_tiles_n);
//    TODO: try more than one tile per warp? would allow increasing sharing without increasing block size, but maybe this is already done by k dimension tiling?

    int dimx = ceil(((float) n)/(wmma_n * warp_tiles_n * block_tiles_n));
    int dimy = ceil(((float) m)/(wmma_m * warp_tiles_m * block_tiles_m));

    dim3 grid(dimx, dimy, 1);
    dim3 block(threads_per_block, 1, 1);

    TimeMeasurement t;

    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiledTensor<elmAccT, elmT, wmma_m, wmma_n, wmma_k, warp_tiles_m, warp_tiles_n, warp_tiles_k, block_tiles_m, block_tiles_n, block_tiles_k, threads_per_block><<<grid, block>>>(
                A_device, B_device, ResMat_device, m, n, k
        );
    }
    cudaDeviceSynchronize();
    t.stop();

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


template <typename elmT, int tile_size, int reg_size>
long int benchmark_tiled_mmm(
        int n_runs,
        elmT *A_device,
        elmT *B_device,
        elmT *ResMat_device,
        int m,
        int n,
        int k)
{
    int dimy = ceil( ((float) n)/(tile_size * reg_size));
    int dimx = ceil( ((float) m)/(tile_size * reg_size));
    TimeMeasurement t;
    dim3 grid(dimx, dimy, 1);
    dim3 block(16, 16, 1);

    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiled<elmT, tile_size, reg_size, tile_size, reg_size, tile_size><<<grid, block>>>(
                A_device, B_device, ResMat_device, m, n, k);
    }
    cudaDeviceSynchronize();
    t.stop();
    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}

// Expects A to have shape K x K and B to have K x N
template <typename elmT, int tile_size, int reg_size, int MatDim, bool use_tensor_cores, typename elmAccT = elmT>
//int reg_size, int n_runs = 1, int MatDim = 2, class accT = elmT>
RandomMatrix<elmAccT, MatDim>* run_mmm_kernel(
        int n_runs,
        int m,
        int n,
        int k,
        RandomMatrix<elmT, MatDim> &A,
        RandomMatrix<elmT, MatDim> &B)
{
    double total_ops = 2.0f * n * k * m;
    auto ResMat = new RandomMatrix<elmAccT, MatDim>;
    // This took me like 2 hours to fix...
    ResMat->fill(0, m, n);

    auto A_device = A.to_gpu();
    auto B_device = B.to_gpu();

    auto ResMat_device = ResMat->to_gpu();
    long int total_elapsed;
    if constexpr(use_tensor_cores) {
        total_elapsed = benchmark_tiled_tensor_mmm<elmT, elmAccT>(
                n_runs, A_device, B_device, ResMat_device, m, n, k
        );
    }
    else {
        total_elapsed = benchmark_tiled_mmm<elmT, tile_size, reg_size>(
                n_runs, A_device, B_device, ResMat_device, m, n, k
        );
    }

    cudaMemcpy(ResMat->to_cpu(), ResMat_device, ResMat->flatSize() * sizeof(elmAccT), cudaMemcpyDeviceToHost);
    cudaFree(A_device); cudaFree(B_device); cudaFree(ResMat_device);
    gpuAssert(cudaPeekAtLastError());


    if (!total_elapsed) {
        printf("Kernel launch failed\n");
        memset(ResMat->to_cpu(), 0, m * n);
    } else {
        printGFlops(total_elapsed, total_ops * n_runs);
    }
    return ResMat;
}





int main(int argc, char * argv[])
{
    constexpr int m = 16 * 256;
    constexpr int n = 16 * 256;
//    TODO: does not work if this is different, fix that
    constexpr int k = 16 * 256;

    int n_runs;

    if (argc == 2)
    {
        n_runs = atoi(argv[1]);
    } else {
        n_runs = 10;
    }

    // Tiled GPU verion
    // TODO: this fails when the type is float since it is not supported for wmma
    // and the templated function is still created
    RandomMatrix<float, 2> A;
    RandomMatrix<float, 2> B;
    TimeMeasurement t;
    A.fill_rand<float_range>(m, k);
    B.fill_rand<float_range>(k, n);

    std::cout << "-----" << std::endl;
    std::cout << "Running GPU register tiled version" << std::endl;
    std::cout << "Dry run" << std::endl;
    run_mmm_kernel<float, 16, 5, 2, false>(
            1, m, n, k, A, B
    );
    std::cout << "Average run of: " << n_runs << std::endl;
    RandomMatrix<float, 2> *C = run_mmm_kernel<float, 16, 5, 2, false>(
            n_runs, m, n, k, A, B
    );
    RandomMatrix<float, 2> target_res;
    target_res.fill_from(*C, m * n);
    std::cout << "-----" << std::endl;

    // GPU version
    RandomMatrix<half, 2> A_half;
    RandomMatrix<half, 2> B_half;
    A_half.fill_from(A, m, k);
    B_half.fill_from(B, k, n);

    constexpr int block_tile_size = 5; // TODO: calculate based on amount of shared memory

    std::cout << "-----" << std::endl;
    std::cout << "Running GPU tensor version" << std::endl;
    std::cout << "Dry run" << std::endl;
    RandomMatrix<float, 2> *GPU_res_tensor_half = run_mmm_kernel<half, 16, 5, 2, true, float>(
            1, m, n, k, A_half, B_half
    );

    RandomMatrix<float, 2> GPU_res_tensor;
    GPU_res_tensor.fill_from(*GPU_res_tensor_half, m, n);
    Validator<float> validator(target_res.to_cpu(), GPU_res_tensor.to_cpu(), m * n);
    // validator.setEps(0.000005); // original used by cosmin
    validator.setEps(0.0005);

    std::cout << "Average run after: " << n_runs << " runs"<< std::endl;
    run_mmm_kernel<half, 16, 5, 2, true, float>(
            n_runs, m, n, k, A_half, B_half
    );
    std::cout << "-----" << std::endl;

    validator.validate();

    delete GPU_res_tensor_half;

    return 0;



//    constexpr int k = 16 * 256;// Multiple of 8 to allign with frame leading dimension
//    constexpr int m = 16 * 256;// Multiple of 8 to allign with frame leading dimension
//    constexpr int n = 16 * 256;// Multiple of 8 to allign with frame leading dimension
//
//    // Tiled GPU verion
//    // TODO: this fails when the type is float since it is not supported for wmma
//    // and the templated function is still created
//    RandomMatrix<float, 2> A;
//    RandomMatrix<float, 2> B;
//    RandomMatrix<float, 2> CPU_res;
//    TimeMeasurement t;
//    A.fill_rand<float_range>(m, k);
//    B.fill_rand<float_range>(k, n);
//    CPU_res.fill_rand<float_range>(m, n);
//    std::cout << "Running GPU version" << std::endl;
//
//
//     RandomMatrix<float, 2> *CPU = run_mmm_kernel<float, 16, 5, 2, 1, false>(
//         m, n, k, A, B
//     );
//     RandomMatrix<float, 2> GPU_res_tiled;
//     GPU_res_tiled.fill_from(*GPU_res_tiled_half, m * n);
//
//    // GPU version
//    RandomMatrix<half, 2> A_half;
//    RandomMatrix<half, 2> B_half;
//    A_half.fill_from(A, m, k);
//    B_half.fill_from(B, k, n);
//    constexpr int block_tile_size = 5; // TODO: calculate based on amount of shared memory
//    std::cout << "Running GPU tensor version" << std::endl;
//
////    TODO: check arguments
//    RandomMatrix<half, 2> *GPU_res_tensor_half = run_mmm_kernel<half, 16, 5, 2, 1, true>(
//        m, k, n, A_half, B_half
//    );
//
//    RandomMatrix<float, 2> GPU_res_tensor;
//    GPU_res_tensor.fill_from(*GPU_res_tensor_half, m, n);
//
//    Validator<float> validator(CPU_res.to_cpu(), GPU_res_tensor.to_cpu(), m * n);
//
////    print C:
////    printf("C CPU:\n");
////    for (int i = 0; i < m; i++) {
////        for (int j = 0; j < n; j++) {
////            std::cout << CPU_res.to_cpu()[i * n + j] << " ";
////        }
////        std::cout << std::endl;
////    }
////
////    printf("C GPU:\n");
////    for (int i = 0; i < m; i++) {
////        for (int j = 0; j < n; j++) {
////            std::cout << GPU_res_tensor.to_cpu()[i * n + j] << " ";
////        }
////        std::cout << std::endl;
////    }
//
//    // validator.setEps(0.000005);
//    validator.setEps(0.05);
//    validator.validate();
//    delete GPU_res_tensor_half;
//    // delete GPU_res_tiled_half;
//
//    return 0;
}
