#include <iostream>
#include "helpers.h"

int main() 
{
    constexpr int N = 10;
    int r1[N];
    int r2[N];
    std::fill_n(r1, N, 1);
    std::fill_n(r2, N, 2);
    
    RandomMatrix<float, 3> A; // Float matrix with 3 dimensions
    // For int use RANDMAX = 10
    constexpr int RANDMAX = RAND_MAX / 10; // [0, 10) range]
    A.fill<RANDMAX>(2, 3, 4); // fill a 2 x 3 x 4 matrix

    RandomMatrix<int, 1> B(r1, N);
    RandomMatrix<int, 1> C;
    A.fill<10>(N);

    Validator<int> validator(r1, r2, N);    
    validator.validate();


}