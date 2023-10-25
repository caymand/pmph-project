#include <cstdio>
#include <mma.h>
#include "matmul.cuh"
#include "helpers.h"
#include "goldenSeq.h"
#include "matmul-tensor.cuh"

// template <typename elmT, int n_runs>
// unsigned benchmark_tiled_mmm(
//     int tile_size, 
//     int reg_size,
//     elmT *A_device, 
//     elmT *B_device, 
//     elmT *ResMat_device, 
//     int height_A, 
//     int width_B, 
//     int width_A) 
// {
//     int dimy = ceil( ((float) width_B)/(tile_size * reg_size));
//     int dimx = ceil( ((float) height_A)/(tile_size * reg_size));
//     TimeMeasurement t;
//     dim3 grid(dimx, dimy, 1);
//     dim3 block(16, 16, 1);
    
//     t.start();
//     for (int i = 0; i < n_runs; i++) {
//         matMulTiled<elmT, tile_size, reg_size, tile_size, reg_size, tile_size><<<grid, block>>>(
//                 A_device, B_device, ResMat_device, height_A, width_B, width_A);
//     }
//     cudaDeviceSynchronize();
//     t.stop();
//     // Check if kernel launch was successfull
//     if (!gpuAssert(cudaPeekAtLastError())) {
//         return 0;
//     }
//     return t.elapsed();
// }

// template <typename elmT, int n_runs, typename elmAccT = elmT>
// unsigned benchmark_tiled_tensor_mmm(
//     int block_tile_size,       
//     elmT *A_device, 
//     elmT *B_device, 
//     elmAccT *ResMat_device, 
//     int height_A, 
//     int width_B, 
//     int width_A) 
// {
//     constexpr int wmma_c = 16;
//     int dimy = ceil( ((float) width_B)/(block_tile_size * wmma_c));
//     int dimx = ceil( ((float) height_A)/(block_tile_size * wmma_c));
//     TimeMeasurement t;
//     dim3 grid(dimx, dimy, 1);
//     dim3 block(16, 16, 1);
    
//     t.start();
//     for (int i = 0; i < n_runs; i++) {
//         matMulTiledTensor<elmAccT, elmT, wmma_c, wmma_c, wmma_c, block_tile_size><<<grid, block>>>(
//             A_device, B_device, ResMat_device, height_A, width_B, width_B
//         );
//     }
//     cudaDeviceSynchronize();
//     t.stop();
//     // Check if kernel launch was successfull
//     if (!gpuAssert(cudaPeekAtLastError())) {
//         return 0;
//     }
//     return t.elapsed();
// }

template <typename elmT, int block_tile_size, int n_runs, int wmma_n, int wmma_m, int wmma_k, typename elmAccT>
unsigned benchmark_mmm(    
    elmT *A_device, 
    elmT *B_device, 
    elmAccT *ResMat_device, 
    int height_A, 
    int width_B, 
    int width_A) 
{
    constexpr int wmma_c = 16;
    int dimy = ceil( ((float) width_B)/(block_tile_size * wmma_c));
    int dimx = ceil( ((float) height_A)/(block_tile_size * wmma_c));
    TimeMeasurement t;
    dim3 grid(dimx, dimy, 1);
    dim3 block(16, 16, 1);
    
    t.start();
    for (int i = 0; i < n_runs; i++) {
        matMulTiledTensor<elmAccT, elmT, wmma_c, wmma_c, wmma_c, block_tile_size><<<grid, block>>>(
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
unsigned benchmark_mmm(
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
    // unsigned total_elapsed = benchmark_mmm<elmT, tile_size, n_runs, Args>(A_device, B_device, ResMat_device, height_A, width_B, width_A);
    if constexpr(use_tensor_cores) {       
        total_elapsed = benchmark_mmm<elmT, elmT, tile_size, 16, 16, 16, n_runs>(
            A_device, B_device, ResMat_device, height_A, width_B, width_A
        );
        // benchmark_tiled_tensor_mmm<elmT, n_runs, elmT>(
        //     tile_size , A_device, B_device, ResMat_device, height_A, width_B, width_A);
    }
    else {
        // total_elapsed = benchmark_tiled_mmm<elmT, tile_size, reg_size, n_runs>(
        //     tile_size, 5, A_device, B_device, ResMat_device, height_A, width_B, width_A);
        total_elapsed = benchmark_mmm<elmT, tile_size, reg_size, n_runs>(
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
//     constexpr int float_range = RAND_MAX / 10;
//     constexpr int n = 16 * 5 * 24;// Multiple of 8 to allign with frame leading dimension
//     constexpr int m = 16 * 5 * 24;// Multiple of 8 to allign with frame leading dimension
//     constexpr int k = 16 * 5 * 24;// Multiple of 8 to allign with frame leading dimension

//     constexpr int tile_size2 = 16;
//     constexpr int reg_size2 = 5;

//     int dimy2 = ceil( ((float) m)/(tile_size2 * reg_size2));
//     int dimx2 = ceil( ((float) n)/(tile_size2 * reg_size2));

//     dim3 grid2(dimx2, dimy2, 1);
//     dim3 block2(16, 16, 1);

//     constexpr int n_runs = 1;
//     constexpr double total_ops = 2.0f * m * k * n;

//     constexpr int wmma_m = 16;
//     constexpr int wmma_n = 16;
//     constexpr int wmma_k = 16;


// //    TODO: calculate based on amount of shared memory
//     constexpr int block_tile_size = 5;

//     int dimy = ceil( ((float) m)/(block_tile_size * wmma_m));
//     int dimx = ceil( ((float) n)/(block_tile_size * wmma_n));

//     dim3 grid(dimx, dimy, 1);
//     dim3 block(16, 16, 1);
    
//     // Allocate 3 matrices with random data
//     RandomMatrix<half, 2> Ahost;
//     RandomMatrix<half, 2> Bhost;
//     RandomMatrix<half, 2> Chost;
//     RandomMatrix<half, 2> Dhost;
    
//     Ahost.fill<float_range>(m, k);
//     Bhost.fill<float_range>(k, n);
//     Chost.fill<float_range>(m, n);
//     Dhost.fill<float_range>(m, n);

//     TimeMeasurement t;

// //    std::cout << "Running on CPU" << std::endl;
// //    t.start();
// //    goldenSeq<float>(Ahost.to_cpu(), Bhost.to_cpu(), Chost.to_cpu(), n, k, m);
// //    t.stop();

// //    printGFlops(t.elapsed(), total_ops);

//     std::cout << "Running on GPU:" << std::endl;
//     auto Adevice = Ahost.to_gpu();
//     auto Bdevice = Bhost.to_gpu();
//     auto Cdevice = Chost.to_gpu();
//     auto Ddevice = Dhost.to_gpu();

//     t.start();
//     {
//         for (int i = 0; i < n_runs; i++) {
//             matMulTiled<half, tile_size2, reg_size2, tile_size2, reg_size2, tile_size2><<<grid, block>>>(
//                     Adevice, Bdevice, Cdevice, m, n, k);
//         }
//         cudaDeviceSynchronize();
//     }
//     t.stop();
//     cudaMemcpy(Chost.to_cpu(), Cdevice, Chost.flatSize() * sizeof(float), cudaMemcpyDeviceToHost);
// //    cudaFree(Adevice); cudaFree(Bdevice); cudaFree(Ddevice);
//     gpuAssert( cudaPeekAtLastError() );

//     printGFlops(t.elapsed(), total_ops * n_runs);
    
//     t.start();
//     {
//         for (int i = 0; i < n_runs; i++) {
//             matMulTiledTensor<half, half, wmma_m, wmma_n, wmma_k, block_tile_size><<<grid, block>>>(
//                 Adevice, Bdevice, Ddevice, m, n, k);
//         }
//         cudaDeviceSynchronize();
//     }
//     t.stop();
//     cudaMemcpy(Dhost.to_cpu(), Ddevice, Dhost.flatSize() * sizeof(float), cudaMemcpyDeviceToHost);
// //    cudaFree(Adevice); cudaFree(Bdevice); cudaFree(Ddevice);
//     gpuAssert( cudaPeekAtLastError() );

//     printGFlops(t.elapsed(), total_ops * n_runs);

    
    constexpr int width_A = 16 * 5 * 12;// Multiple of 8 to allign with frame leading dimension
    constexpr int height_A = 16 * 5 * 12;// Multiple of 8 to allign with frame leading dimension
    constexpr int width_B = 16 * 5 * 12;// Multiple of 8 to allign with frame leading dimension
    
    // Tiled GPU verion
    // TODO: this fails when the type is float since it is not supported for wmma
    // and the templated function is still created
    RandomMatrix<half, 2> A;
    RandomMatrix<half, 2> B;
    TimeMeasurement t;
    A.fill<float_range>(height_A, width_A);
    B.fill<float_range>(height_A, width_B);
    std::cout << "Running GPU tiled version" << std::endl;
    RandomMatrix<half, 2> *GPU_res_tiled_half = run_mmm_kernel<half, 16, 5, 2, 1, false>(
        height_A, width_B, width_A, A, B
    );
    RandomMatrix<float, 2> GPU_res_tiled;
    GPU_res_tiled.fill_from(*GPU_res_tiled_half, height_A * width_B);

    // GPU version    
    RandomMatrix<half, 2> A_half;
    RandomMatrix<half, 2> B_half;
    A_half.fill_from(A, height_A, width_A);
    B_half.fill_from(B, width_A, width_B);
    constexpr int block_tile_size = 5; // TODO: calculate based on amount of shared memory
    std::cout << "Running GPU tensor version" << std::endl;
    RandomMatrix<half, 2> *GPU_res_tensor_half = run_mmm_kernel<half, 16, 5, 2, 1, false>(
        height_A, width_A, width_B, A_half, B_half
    );
    
    RandomMatrix<float, 2> GPU_res_tensor;
    GPU_res_tensor.fill_from(*GPU_res_tensor_half, height_A, width_B);

    Validator<float> validator(GPU_res_tiled.to_cpu(), GPU_res_tensor.to_cpu(), height_A * width_B);
    // validator.setEps(0.000005);
    validator.setEps(0.05);
    validator.validate();
    delete GPU_res_tensor_half;
    delete GPU_res_tiled_half;

    return 0;
}
