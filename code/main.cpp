#include <iostream>
#include "helpers.h"
#include "goldenSeq.h"
#include <vector>


int main() 
{
    constexpr int float_range = RAND_MAX / 10;
    constexpr int n = 1000;
    constexpr int m = 1000;
    constexpr int k = 1000;
    
    double total_ops = 2.0f * m * k * n;

    RandomMatrix<float, 2> A;
    RandomMatrix<float, 2> B;
    A.fill<float_range>(n, m);
    B.fill<float_range>(m, k);
    std::vector<float> C(n * k);
    
    TimeMeasurement t;
    
    t.start();
    goldenSeq<float>(A.to_cpu(), B.to_cpu(), C.data(), n, k, m);
    t.stop();
    
    if (C[0] == 0.0f)  {
        std::cout << "Should not be the case" << std::endl;
    }

    float us = t.elapsed() ;
    std::cout << "Ran in " << us << " microseconds" << std::endl;
    if (printGFlops(us, total_ops)) {
        std::cout << "ERROR" << std::endl;
    }
}   