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

enum mm_kernel {
    register_tiled,
    tensor_naive,
    tensor_optimized,
    cublas
};


template <typename elmT, typename elmAccT = elmT>
long int benchmark_optimized_tensor_mmm(
        int n_runs,
        elmT *A_device,
        elmT *B_device,
        elmAccT *C_device,
        int m,
        int n,
        int k)
{

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
    printf("    Threads used: %d/%d\n", threads_per_block, MAX_THREADS_PER_BLOCK);
    assert(threads_per_block <= MAX_THREADS_PER_BLOCK);
    // Assumes num_warps >= block_tiles_m * block_tiles_n, i.e. all block tiles are handled by a warp
    assert(threads_per_block / WARP_SIZE >= block_tiles_m * block_tiles_n);

    int dimx = ceil(((float) n)/(wmma_n * warp_tiles_n * block_tiles_n));
    int dimy = ceil(((float) m)/(wmma_m * warp_tiles_m * block_tiles_m));

    dim3 grid(dimx, dimy, 1);
    dim3 block(threads_per_block, 1, 1);

    printf("    Blocks used: %d x %d = %d\n", dimx, dimy, dimx * dimy);

    printf("    Available registers per thread: %d (%d per block)\n", MAX_REGISTERS_PER_BLOCK / threads_per_block, MAX_REGISTERS_PER_BLOCK);

    constexpr unsigned int shared_m = wmma_m * warp_tiles_m * block_tiles_m;
    constexpr unsigned int shared_n = wmma_n * warp_tiles_n * block_tiles_n;
    constexpr unsigned int shared_k = wmma_k * block_tiles_k;

    constexpr unsigned int shared_memory_used = (shared_m * (shared_k + SHARED_PADDING)+ shared_k * (shared_n + SHARED_PADDING)) * sizeof(elmT) * 2;

    printf("    Shared memory used: %d/%d bytes (%.0f%%)\n", shared_memory_used, SHARED_MEM_SIZE, (float) shared_memory_used / SHARED_MEM_SIZE * 100);


    TimeMeasurement t;

    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiledTensor<elmT, elmAccT, wmma_m, wmma_n, wmma_k, warp_tiles_m, warp_tiles_n, block_tiles_m, block_tiles_n, block_tiles_k, threads_per_block><<<grid, block>>>(
                A_device, B_device, C_device, m, n, k
        );
    }
    cudaDeviceSynchronize();
    t.stop();

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


template <typename elmT, typename elmAccT = elmT>
unsigned benchmark_naive_tensor_mmm(
        unsigned n_runs,
        elmT *A_device,
        elmT *B_device,
        elmAccT *ResMat_device,
        int m,
        int n,
        int k)
{
    constexpr int block_tiles_m = 8;
    constexpr int block_tiles_n = 4;
    constexpr int block_tiles_k = 4;
    constexpr int wmma_n = 16;
    constexpr int wmma_m = 16;
    constexpr int wmma_k = 16;


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
        matMulTiledTensorNaive<
            elmAccT, elmT, wmma_m, wmma_n, wmma_k, block_tiles_m, block_tiles_n, block_tiles_k>
            <<<grid, block>>>(A_device, B_device, ResMat_device, m, n, n);
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

    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiled<elmT, elmAccT, tile_size, reg_size, tile_size, reg_size, tile_size><<<grid, block>>>(
                A_device, B_device, C_device, m, n, k);
    }
    cudaDeviceSynchronize();
    t.stop();
    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


template <typename elmT, typename elmAccT>
cublasStatus_t cublas_wrapper(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const elmAccT *alpha,
        const elmT *A, int lda,
        const elmT *B, int ldb,
        const elmAccT *beta,
        elmAccT *C, int ldc
);

template <>
cublasStatus_t cublas_wrapper<half, half>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const half *alpha,
        const half *A, int lda,
        const half *B, int ldb,
        const half *beta,
        half *C, int ldc
) {
    return cublasGemmEx(
            handle,
            transa, transb,
            m, n, k,
            alpha,
            A, CUDA_R_16F, lda,
            B, CUDA_R_16F, ldb,
            beta,
            C, CUDA_R_16F, ldc,
            CUDA_R_16F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}

template <>
cublasStatus_t cublas_wrapper<float, float>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float *alpha,
        const float *A, int lda,
        const float *B, int ldb,
        const float *beta,
        float *C, int ldc
) {
    return cublasGemmEx(
            handle,
            transa, transb,
            m, n, k,
            alpha,
            A, CUDA_R_32F, lda,
            B, CUDA_R_32F, ldb,
            beta,
            C, CUDA_R_32F, ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}


template <>
cublasStatus_t cublas_wrapper<half, float>(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float *alpha,
        const half *A, int lda,
        const half *B, int ldb,
        const float *beta,
        float *C, int ldc
) {
    return cublasGemmEx(
            handle,
            transa, transb,
            m, n, k,
            alpha,
            A, CUDA_R_16F, lda,
            B, CUDA_R_16F, ldb,
            beta,
            C, CUDA_R_32F, ldc,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
}


template <typename elmT, typename elmAccT>
long int benchmark_cublas(
        int n_runs,
        elmT *A_device,
        elmT *B_device,
        elmAccT *C_device,
        int m,
        int n,
        int k)
{
    TimeMeasurement t;

    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    elmAccT alpha = (elmAccT) 1.0;
    elmAccT beta = (elmAccT) 0.0;
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    t.start();
    for (int i = 0; i < n_runs; i++) {
        stat = cublas_wrapper<elmT, elmAccT>(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
            &alpha,
            // Cublas uses column major, so we need to swap A and B, since B^T @ A^T = (A @ B)^T = C^T
            B_device, n,
            A_device, k,
            &beta,
            C_device, n
        );
    }
    cudaDeviceSynchronize();
    t.stop();

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS error\n");
        printf("%s\n", cublasGetStatusName(stat));
        printf("%s\n", cublasGetStatusString(stat));
        exit(1);
    }

    // Check if kernel launch was successfull
    gpuAssert(cudaPeekAtLastError());
    return t.elapsed();
}


// Expects A to have shape K x K and B to have K x N
template <typename elmT, typename elmAccT, int MatDim, mm_kernel kernel_type>
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

    if constexpr (kernel_type == mm_kernel::tensor_optimized) {
        total_elapsed = benchmark_optimized_tensor_mmm<elmT, elmAccT>(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    }
    else if constexpr (kernel_type == mm_kernel::tensor_naive) {
        total_elapsed = benchmark_naive_tensor_mmm<elmT, elmAccT>(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    }
    else if constexpr (kernel_type == mm_kernel::cublas) {
        total_elapsed = benchmark_cublas<elmT, elmAccT>(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    }
    else {
        total_elapsed = benchmark_tiled_mmm<elmT, elmAccT>(
                n_runs, A_device, B_device, C_device, m, n, k
        );
    }

    cudaMemcpy(C.to_cpu(), C_device, C.flatSize() * sizeof(elmAccT), cudaMemcpyDeviceToHost);


    if (!total_elapsed) {
        printf("Kernel launch failed\n");
        memset(C.to_cpu(), 0, m * n);
    } else {
        printGFlops(total_elapsed, total_ops * n_runs);
    }
}


// Expects A to have shape K x K and B to have K x N
template <typename elmT, typename accT, int MatDim, mm_kernel kernel_type, bool validate>
void benchmark_kernel(
        int n_runs,
        int m,
        int n,
        int k,
        RandomMatrix<elmT, MatDim> &A,
        RandomMatrix<elmT, MatDim> &B,
        RandomMatrix<accT, MatDim> &C,
        RandomMatrix<accT, MatDim> &C_target,
        std::string kernel_name
    ) {
    C.fill_zeros(m, n);

    std::cout << "-----" << std::endl;
    std::cout << "Running " << kernel_name << std::endl;
    std::cout << "Dry run" << std::endl;
    run_mmm_kernel<elmT, accT, MatDim, kernel_type>(
            1, m, n, k, A, B, C
    );

    RandomMatrix<accT, MatDim> C_actual;

    if constexpr (validate) {
        C_actual.fill_from(C, m, n);
    }

    std::cout << "Average run after: " << n_runs << " runs"<< std::endl;
    run_mmm_kernel<elmT, accT, MatDim, kernel_type>(
            n_runs, m, n, k, A, B, C
    );
    std::cout << "-----" << std::endl;

    if constexpr (validate)
    {
        Validator<accT> validator(C_target.to_cpu(), C_actual.to_cpu(), m * n);
        // validator.setEps(0.000005); // original used by cosmin
        validator.setEps(0.0005);

        validator.validate();
    }
}


#ifdef ELM_T
typedef ELM_T element_type;
#else
typedef half element_type;
#endif

#ifdef ACC_T
typedef ACC_T acc_type;
#else
typedef float acc_type;
#endif


int main(int argc, char * argv[])
{
    int m = 16 * 256;
    int n = 16 * 256;
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

    // Define matrices
    RandomMatrix<element_type, 2> A;
    RandomMatrix<element_type, 2> B;
    RandomMatrix<acc_type, 2> A_accT;
    RandomMatrix<acc_type, 2> B_accT;
    RandomMatrix<acc_type, 2> C;
    RandomMatrix<acc_type, 2> C_target;

    // Initialize matrices
    A.fill_rand<float_range>(m, k);
    B.fill_rand<float_range>(k, n);
    A_accT.fill_from(A, m, k);
    B_accT.fill_from(B, k, n);

    benchmark_kernel<acc_type, acc_type, 2, mm_kernel::register_tiled, false>(
        n_runs, m, n, k, A_accT, B_accT, C_target, C_target, std::string("GPU register tiled")
    );

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::tensor_naive, true>(
        n_runs, m, n, k, A, B, C, C_target, std::string("GPU tensor naive")
    );

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::tensor_optimized, true>(
        n_runs, m, n, k, A, B, C, C_target, std::string("GPU tensor optimized")
    );

    benchmark_kernel<element_type, acc_type, 2, mm_kernel::cublas, true>(
        n_runs, m, n, k, A, B, C, C_target, std::string("cublas")
    );

    cudaFree(A.to_gpu());
    cudaFree(B.to_gpu());
    cudaFree(C.to_gpu());
    cudaFree(C_target.to_gpu());
    cudaFree(A_accT.to_gpu());
    cudaFree(B_accT.to_gpu());

    return 0;
}
