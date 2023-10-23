#include <cstdio>
#include <mma.h>
#include "matmul.cuh"
#include "matmul-tensor.cu"
#include "helpers.h"
#include "goldenSeq.h"
#include "matmul-tensor.cuh"

//using namespace nvcuda;


int main(int argc, char * argv[]) {
    constexpr int float_range = RAND_MAX / 10;
    constexpr int n = 16 * 5 * 24;// Multiple of 8 to allign with frame leading dimension
    constexpr int m = 16 * 5 * 24;// Multiple of 8 to allign with frame leading dimension
    constexpr int k = 16 * 5 * 24;// Multiple of 8 to allign with frame leading dimension

    constexpr int tile_size2 = 16;
    constexpr int reg_size2 = 5;

    int dimy2 = ceil( ((float) m)/(tile_size2 * reg_size2));
    int dimx2 = ceil( ((float) n)/(tile_size2 * reg_size2));

    dim3 grid2(dimx2, dimy2, 1);
    dim3 block2(16, 16, 1);


    constexpr int n_runs = 1;
    constexpr double total_ops = 2.0f * m * k * n;

    constexpr int wmma_m = 16;
    constexpr int wmma_n = 16;
    constexpr int wmma_k = 16;


//    TODO: calculate based on amount of shared memory
    constexpr int block_tile_size = 5;

    int dimy = ceil( ((float) m)/(block_tile_size * wmma_m));
    int dimx = ceil( ((float) n)/(block_tile_size * wmma_n));

    dim3 grid(dimx, dimy, 1);
    dim3 block(16, 16, 1);
    
    // Allocate 3 matrices with random data
    RandomMatrix<half, 2> Ahost;
    RandomMatrix<half, 2> Bhost;
    RandomMatrix<half, 2> Chost;
    RandomMatrix<half, 2> Dhost;
    
    Ahost.fill<float_range>(m, k);
    Bhost.fill<float_range>(k, n);
    Chost.fill<float_range>(m, n);
    Dhost.fill<float_range>(m, n);

    TimeMeasurement t;

//    std::cout << "Running on CPU" << std::endl;
//    t.start();
//    goldenSeq<float>(Ahost.to_cpu(), Bhost.to_cpu(), Chost.to_cpu(), n, k, m);
//    t.stop();

//    printGFlops(t.elapsed(), total_ops);

    std::cout << "Running on GPU:" << std::endl;
    auto Adevice = Ahost.to_gpu();
    auto Bdevice = Bhost.to_gpu();
    auto Cdevice = Chost.to_gpu();
    auto Ddevice = Dhost.to_gpu();

    t.start();
    {
        for (int i = 0; i < n_runs; i++) {
            matMulTiled<half, tile_size2, reg_size2, tile_size2, reg_size2, tile_size2><<<grid, block>>>(
                    Adevice, Bdevice, Cdevice, m, n, k);
        }
        cudaDeviceSynchronize();
    }
    t.stop();
    cudaMemcpy(Chost.to_cpu(), Cdevice, Chost.flatSize() * sizeof(float), cudaMemcpyDeviceToHost);
//    cudaFree(Adevice); cudaFree(Bdevice); cudaFree(Ddevice);
    gpuAssert( cudaPeekAtLastError() );

    printGFlops(t.elapsed(), total_ops * n_runs);
    
    t.start();
    {
        for (int i = 0; i < n_runs; i++) {
            matMulTiledTensor<half, half, wmma_m, wmma_n, wmma_k, block_tile_size><<<grid, block>>>(
                Adevice, Bdevice, Ddevice, m, n, k);
        }
        cudaDeviceSynchronize();
    }
    t.stop();
    cudaMemcpy(Dhost.to_cpu(), Ddevice, Dhost.flatSize() * sizeof(float), cudaMemcpyDeviceToHost);
//    cudaFree(Adevice); cudaFree(Bdevice); cudaFree(Ddevice);
    gpuAssert( cudaPeekAtLastError() );

    printGFlops(t.elapsed(), total_ops * n_runs);

    Validator<float> validator(reinterpret_cast<float *>(Chost.to_cpu()), reinterpret_cast<float *>(Dhost.to_cpu()), n * k);
    validator.setEps(0.000005);
    validator.validate();

    return 0;
}
