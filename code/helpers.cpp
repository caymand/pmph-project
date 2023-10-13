#include "helpers.h"
#include <iostream>
#include <vector>
#include <algorithm>

template <typename T>
Validator<T>::Validator(T *res1, T* res2, int N) 
{
    this->res1.insert(this->res1.begin(), res1, res1 + N);
    this->res2.insert(this->res2.begin(), res2, res2 + N);
    this->eps = 0.0000000001; // e-10
}
template <typename T>
void Validator<T>::setEps(int eps) {
    this->eps = eps;
}

template <typename T>
void Validator<T>::validate()
{    
    bool errValue = false;
    std::vector<bool> flatMask(this->res1.capacity());
    std::transform(
        this->res1.begin(), this->res1.end(), this->res2.begin(), flatMask.begin(),  
        [this] (T v1, T v2) {
            float err = (float)std::abs(v1 - v2) / std::max(v1, v2);            
            return err < this->eps;
        }
    );    
    auto errorCount = std::count(flatMask.begin(), flatMask.end(), errValue);
    auto firstError = std::find(flatMask.begin(), flatMask.end(), errValue) - flatMask.begin();
    if (errorCount <= 0) {
        std::cout << "VALID" << std::endl;
        std::cout << "-----" << std::endl;    
        return;
    }
    std::cout << "INVALID" << std::endl;
    std::cout << "-------" << std::endl;
    std::cout << "Found: " << errorCount << " ";
    std::cout << "wrong elements." << std::endl;
    std::cout << "First at flat index: " << firstError << std::endl;
    std::cout << "Expected: " << this->res1[firstError] << " ";
    std::cout << "got: " << this->res2[firstError] << std::endl;
}

template class Validator<int>;
