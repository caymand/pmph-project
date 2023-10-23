#include <cstdio>
#include <mma.h>
#include "matmul.cuh"
#include "helpers.h"
#include "goldenSeq.h"

//using namespace nvcuda;


int main(int argc, char * argv[]) {
    constexpr int float_range = RAND_MAX / 10;
    constexpr int n = 16 * 5 * 24;// Multiple of 8 to allign with frame leading dimension
    constexpr int m = 16 * 5 * 24;// Multiple of 8 to allign with frame leading dimension
    constexpr int k = 16 * 5 * 24;// Multiple of 8 to allign with frame leading dimension
    constexpr int tile_size = 16;
    constexpr int reg_size = 5;
    constexpr int n_runs = 1;
    constexpr double total_ops = 2.0f * m * k * n;

    int dimy = ceil( ((float)n)/(tile_size * reg_size) ); 
    int dimx = ceil( ((float) k)/(tile_size * reg_size) );
    
    // Allocate 3 matrices with random data
    RandomMatrix<float, 2> Ahost;
    RandomMatrix<float, 2> Bhost;
    RandomMatrix<float, 2> Chost;
    RandomMatrix<float, 2> Dhost;
    
    Ahost.fill<float_range>(n, m);
    Bhost.fill<float_range>(m, k);
    Chost.fill<float_range>(n, k);
    Dhost.fill<float_range>(n, k);

    TimeMeasurement t;

    std::cout << "Running on CPU" << std::endl;
    t.start();
    goldenSeq<float>(Ahost.to_cpu(), Bhost.to_cpu(), Chost.to_cpu(), n, k, m);
    t.stop();

    printGFlops(t.elapsed(), total_ops);

    std::cout << "Running on GPU:" << std::endl;
    auto Adevice = Ahost.to_gpu();
    auto Bdevice = Bhost.to_gpu();
    auto Ddevice = Dhost.to_gpu();

    dim3 grid(dimx, dimy, 1);
    dim3 block(tile_size, tile_size, 1);
    
    t.start();
    {
        for (int i = 0; i < n_runs; i++) {
            matMulTiled<float, tile_size, reg_size, tile_size, reg_size, tile_size><<<grid, block>>>(
                Adevice, Bdevice, Ddevice, n, k, m);
        }
        cudaDeviceSynchronize();
    }
    t.stop();
    cudaMemcpy(Dhost.to_cpu(), Ddevice, Dhost.flatSize() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(Adevice); cudaFree(Bdevice); cudaFree(Ddevice);
    gpuAssert( cudaPeekAtLastError() );

    printGFlops(t.elapsed(), total_ops * n_runs);

    Validator<float> validator(Chost.to_cpu(), Dhost.to_cpu(), n * k);
    validator.setEps(0.000005);
    validator.validate();

    return 0;
}
