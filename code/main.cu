#include <cstdio>
#include <mma.h>
#include "matmul.cuh"
#include "helpers.h"

//using namespace nvcuda;


int main(int argc, char * argv[]) {
    constexpr int float_range = RAND_MAX / 10;
    constexpr int n = 1000;
    constexpr int m = 1000;
    constexpr int k = 1000;
    int dimy = ceil( ((float)n)/(32 * 5) ); 
    int dimx = ceil( ((float) k)/(32 * 5) );
    
    double total_ops = 2.0f * m * k * n;

    // Allocate 3 matrices with random data
    RandomMatrix<float, 2> Ahost;
    RandomMatrix<float, 2> Bhost;
    RandomMatrix<float, 2> Chost;
    Ahost.fill<float_range>(n, m);
    Bhost.fill<float_range>(m, k);
    Chost.fill<float_range>(n, k);

    TimeMeasurement t;

    std::cout << "Running on GPU:" << std::endl;
    {
        t.start();
        matMulTiled<float, 32, 5, 32, 5, 32><<<(32, 32, 1), (dimx, dimy, 1)>>>(
            Ahost.to_gpu(), Bhost.to_gpu(), Chost.to_gpu(), n, k, m);
        cudaDeviceSynchronize();
        t.stop();
        gpuAssert( cudaPeekAtLastError() );
    }

    unsigned int elapsed = t.elapsed();
    printGFlops(elapsed, total_ops);

    return 0;
}
