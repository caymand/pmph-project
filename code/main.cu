#include <cstdio>
#include <mma.h>
#include "matmul.cuh"
#include "helpers.h"
#include "goldenSeq.h"
#include "matmul-tensor.cuh"
#include "cuda_fp16.h"
#include <cassert>
//#include <cublas.h>
#include <cublas_v2.h>


#define WARP_SIZE 32
#define SHARED_MEM_SIZE 49152
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_REGISTERS_PER_BLOCK 65536

#define SHARED_PADDING 8


template <typename elmT, typename elmAccT = elmT>
long int benchmark_tiled_tensor_mmm(
        int n_runs,
        elmT *A_device,
        elmT *B_device,
        elmAccT *C_device,
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
    printf("Threads used: %d/%d\n", threads_per_block, MAX_THREADS_PER_BLOCK);
    assert(threads_per_block <= MAX_THREADS_PER_BLOCK);
    //    Assumes num_warps >= block_tiles_m * block_tiles_n, i.e. all block tiles are handled by a warp
    assert(threads_per_block / WARP_SIZE >= block_tiles_m * block_tiles_n);
//    TODO: try more than one tile per warp?
//    Would allow increasing sharing without increasing block size, but maybe this is already done by k dimension tiling?
//    Would maybe allow more concurrent blocks? Would mean less threads for copying memory

    int dimx = ceil(((float) n)/(wmma_n * warp_tiles_n * block_tiles_n));
    int dimy = ceil(((float) m)/(wmma_m * warp_tiles_m * block_tiles_m));

    dim3 grid(dimx, dimy, 1);
    dim3 block(threads_per_block, 1, 1);

    printf("Blocks used: %d x %d = %d\n", dimx, dimy, dimx * dimy);

    //  TODO: calculate register usage?
    printf("Available registers per thread: %d (%d per block)\n", MAX_REGISTERS_PER_BLOCK / threads_per_block, MAX_REGISTERS_PER_BLOCK);

    constexpr unsigned int shared_m = wmma_m * warp_tiles_m * block_tiles_m;
    constexpr unsigned int shared_n = wmma_n * warp_tiles_n * block_tiles_n;
    constexpr unsigned int shared_k = wmma_k * warp_tiles_k * block_tiles_k;

//    assert(m % wmma_m == 0 && m % wmma_m * warp_tiles_m == 0 && m % wmma_m * warp_tiles_m * block_tiles_m == 0);
//    assert(n % wmma_n == 0 && n % wmma_n * warp_tiles_n == 0 && n % wmma_n * warp_tiles_n * block_tiles_n == 0);
//    assert(k % wmma_k == 0 && k % wmma_k * warp_tiles_k == 0 && k % wmma_k * warp_tiles_k * block_tiles_k == 0);
//    printf("%d %d %d\n", m % wmma_m, m % wmma_m * warp_tiles_m, m % wmma_m * warp_tiles_m * block_tiles_m);
//    printf("%d %d %d\n", n % wmma_n, n % wmma_n * warp_tiles_n, n % wmma_n * warp_tiles_n * block_tiles_n);
//    printf("%d %d %d\n", k % wmma_k, k % wmma_k * warp_tiles_k, k % wmma_k * warp_tiles_k * block_tiles_k);


    constexpr unsigned int shared_memory_used_AB = shared_m * (shared_k + SHARED_PADDING) * sizeof(elmT) + shared_k * (shared_n + SHARED_PADDING) * sizeof(elmT);
#ifdef CACHE_C
//    Add space for caching C
    constexpr unsigned int shared_memory_used = shared_memory_used_AB + shared_m * (shared_n + SHARED_PADDING) * sizeof(elmAccT);
#else
    constexpr unsigned int shared_memory_used = shared_memory_used_AB;
#endif
    printf("Shared memory used: %d/%d bytes (%.0f%%)\n", shared_memory_used, SHARED_MEM_SIZE, (float) shared_memory_used / SHARED_MEM_SIZE * 100);


    TimeMeasurement t;

    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiledTensor<elmT, elmAccT, wmma_m, wmma_n, wmma_k, warp_tiles_m, warp_tiles_n, warp_tiles_k, block_tiles_m, block_tiles_n, block_tiles_k, threads_per_block><<<grid, block>>>(
                A_device, B_device, C_device, m, n, k
        );
    }
    cudaDeviceSynchronize();
    t.stop();

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


template <typename elmT, typename elmAccT>
long int benchmark_tiled_mmm(
        int n_runs,
        elmT *A_device,
        elmT *B_device,
        elmAccT *C_device,
        int m,
        int n,
        int k)
{
    constexpr int tile_size = 16;
    constexpr int reg_size = 5;

    int dimy = ceil( ((float) n)/(tile_size * reg_size));
    int dimx = ceil( ((float) m)/(tile_size * reg_size));
    TimeMeasurement t;
    dim3 grid(dimx, dimy, 1);
    dim3 block(16, 16, 1);

//    cublasHandle_t handle;
//    cublasStatus_t stat;
//    stat = cublasCreate(&handle);
//    half alpha = (half) 1.0;
//    half beta = (half) 0.0;
//    if (stat != CUBLAS_STATUS_SUCCESS) {
//        printf ("CUBLAS initialization failed\n");
//        return EXIT_FAILURE;
//    }

    t.start();
    for (int i = 0; i < n_runs; i++) {
//        cublasHgemm(
//                handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
//                &alpha,
//                (const half *) A_device, k,
//                (const half *) B_device, n,
//                &beta,
//                (half *) C_device, n);
        matMulTiled<elmT, elmAccT, tile_size, reg_size, tile_size, reg_size, tile_size><<<grid, block>>>(
                A_device, B_device, C_device, m, n, k);
    }
    cudaDeviceSynchronize();
    t.stop();
    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}

// Expects A to have shape K x K and B to have K x N
template <typename elmT, typename elmAccT, int MatDim, bool use_tensor_cores>
//int reg_size, int n_runs = 1, int MatDim = 2, class accT = elmT>
void run_mmm_kernel(
        int n_runs,
        int m,
        int n,
        int k,
        RandomMatrix<elmT, MatDim> &A,
        RandomMatrix<elmT, MatDim> &B,
        RandomMatrix<elmAccT, MatDim> &C)
{
    double total_ops = 2.0f * n * k * m;

    auto A_device = A.to_gpu();
    auto B_device = B.to_gpu();

    auto C_device = C.to_gpu();
    long int total_elapsed;
    if constexpr(use_tensor_cores) {
        total_elapsed = benchmark_tiled_tensor_mmm<elmT, elmAccT>(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    }
    else {
        total_elapsed = benchmark_tiled_mmm<elmT, elmAccT>(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    }

    cudaMemcpy(C.to_cpu(), C_device, C.flatSize() * sizeof(elmAccT), cudaMemcpyDeviceToHost);
    cudaFree(A_device); cudaFree(B_device); cudaFree(C_device);
    gpuAssert(cudaPeekAtLastError());


    if (!total_elapsed) {
        printf("Kernel launch failed\n");
        memset(C.to_cpu(), 0, m * n);
    } else {
        printGFlops(total_elapsed, total_ops * n_runs);
    }
}


#ifdef ELM_T
typedef ELM_T elmT;
#else
typedef half elmT;
#endif

#ifdef ACC_T
typedef ACC_T accT;
#else
typedef float accT;
#endif


int main(int argc, char * argv[])
{
    int m = 16 * 256;
    int n = 16 * 256;
//    TODO: does not work if this is different, fix that
    int k = 16 * 256;

    int n_runs = 10;

    if (argc >= 2)
    {
        n_runs = atoi(argv[1]);
    }
    if (argc == 3)
    {
        int input_int = atoi(argv[2]);
        m = input_int;
        n = input_int;
        k = input_int;
    } else if (argc == 4)
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    } else if (argc == 5)
    {
        n_runs = atoi(argv[1]);
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        k = atoi(argv[4]);
    }


    TimeMeasurement t;

    //  Define matrices
    RandomMatrix<elmT, 2> A;
    RandomMatrix<elmT, 2> B;
    RandomMatrix<accT, 2> A_accT;
    RandomMatrix<accT, 2> B_accT;
    RandomMatrix<accT, 2> C;
    RandomMatrix<accT, 2> C_target;
    RandomMatrix<accT, 2> C_actual;

    //  Initialize matrices
    A.fill_rand<float_range>(m, k);
    B.fill_rand<float_range>(k, n);
    C.fill(0, m, n);
    C_target.fill(0, m, n);
    A_accT.fill_from(A, m, k);
    B_accT.fill_from(B, k, n);

    // Tiled GPU verion
    std::cout << "-----" << std::endl;
    std::cout << "Running GPU register tiled version" << std::endl;
    std::cout << "Dry run" << std::endl;
    run_mmm_kernel<accT, accT, 2, false>(
            1, m, n, k, A_accT, B_accT, C_target
    );
    std::cout << "Average run of: " << n_runs << std::endl;
    run_mmm_kernel<accT , accT, 2, false>(
            n_runs, m, n, k, A_accT, B_accT, C_target
    );

    std::cout << "-----" << std::endl;

    // GPU version
    std::cout << "-----" << std::endl;
    std::cout << "Running GPU tensor version" << std::endl;
    std::cout << "Dry run" << std::endl;
    run_mmm_kernel<elmT, accT, 2, true>(
            1, m, n, k, A, B, C
    );

    C_actual.fill_from(C, m, n);

    std::cout << "Average run after: " << n_runs << " runs"<< std::endl;
    run_mmm_kernel<elmT, accT, 2, true>(
            n_runs, m, n, k, A, B, C
    );
    std::cout << "-----" << std::endl;

    Validator<accT> validator(C_target.to_cpu(), C_actual.to_cpu(), m * n);
//    validator.setEps(0.000005); // original used by cosmin
    validator.setEps(0.0005);
//    Check if something is wrong, or we really need this eps
//    validator.setEps((accT) 0.1);

    validator.validate();

    cudaFree(A.to_gpu());
    cudaFree(B.to_gpu());
    cudaFree(C.to_gpu());
    cudaFree(C_target.to_gpu());
    cudaFree(C_actual.to_gpu());
    cudaFree(A_accT.to_gpu());
    cudaFree(B_accT.to_gpu());

    return 0;
}

// TODO: try half-half, half-float, (double-double, tf32-tf32)
// Find best parameters for tiled and tensor for each case
// TODO: graphs

