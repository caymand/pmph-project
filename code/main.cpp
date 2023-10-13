#include <iostream>
#include "helpers.h"

int main() 
{
    constexpr int N = 10;
    int r1[N];
    int r2[N];
    std::fill_n(r1, N, 1);
    std::fill_n(r2, N, 2);
    
    Validator<int> validator(r1, r2, N);    
    validator.validate();
}