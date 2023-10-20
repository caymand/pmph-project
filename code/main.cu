#include <cstdio>
#include <mma.h>
#include "matmul.cuh"
#include "helpers.h"
#include "goldenSeq.h"

//using namespace nvcuda;


int main(int argc, char * argv[]) {
    constexpr int float_range = RAND_MAX / 10;
    constexpr int n = 2048;
    constexpr int m = 4096;
    constexpr int k = 2048;
    int dimy = ceil( ((float)n)/(32 * 5) ); 
    int dimx = ceil( ((float) k)/(32 * 5) );
    
    double total_ops = 2.0f * m * k * n;

    // Allocate 3 matrices with random data
    RandomMatrix<float, 2> Ahost;
    RandomMatrix<float, 2> Bhost;
    RandomMatrix<float, 2> Chost;
    RandomMatrix<float, 2> Cdevice;
    
    Ahost.fill<float_range>(n, m);
    Bhost.fill<float_range>(m, k);
    Chost.fill<float_range>(n, k);
    Cdevice.fill<float_range>(n, k);

    TimeMeasurement t;

    std::cout << "Running on CPU" << std::endl;
    t.start();
    goldenSeq<float>(Ahost.to_cpu(), Bhost.to_cpu(), Chost.to_cpu(), n, k, m);
    t.stop();

    unsigned int elapsed = t.elapsed();
    printGFlops(elapsed, total_ops);

    std::cout << "Running on GPU:" << std::endl;
    auto Adevice = Ahost.to_gpu();
    auto Bdevice = Bhost.to_gpu();
    auto Cdev = Chost.to_gpu();

    dim3 grid(dimx, dimy, 1);
    dim3 block(32, 32, 1);

    t.start();
    {
        matMulTiled<float, 32, 5, 32, 5, 32><<<grid, block>>>(
            Adevice, Bdevice, Cdev, n, k, m);
        cudaDeviceSynchronize();
    }
    t.stop();

    cudaMemcpy(Cdevice.to_cpu(), Cdev, Cdevice.flatSize() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(Adevice); cudaFree(Bdevice); cudaFree(Cdev);
    gpuAssert( cudaPeekAtLastError() );

    elapsed = t.elapsed();
    printGFlops(elapsed, total_ops);

    Validator<float> validator(Chost.to_cpu(), Cdevice.to_cpu(), n * k);
    validator.setEps(0.000005);
    validator.validate();

    return 0;
}
