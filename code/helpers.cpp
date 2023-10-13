#include "helpers.h"
#include <iostream>
#include <vector>
#include <algorithm>

template <typename T>
Validator<T>::Validator(T *flatMat1, T* flatMat2, int N) 
{
    this->flatMat1.insert(this->flatMat1.begin(), flatMat1, flatMat1 + N);
    this->flatMat2.insert(this->flatMat2.begin(), flatMat2, flatMat2 + N);
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
    std::vector<bool> flatMatrixMask(this->flatMat1.capacity());
    std::transform(
        this->flatMat1.begin(), this->flatMat1.end(), this->flatMat2.begin(), flatMatrixMask.begin(),  
        [this] (T v1, T v2) {
            float err = (float)std::abs(v1 - v2) / std::max(v1, v2);            
            return err < this->eps;
        }
    );    
    auto errorCount = std::count(flatMatrixMask.begin(), flatMatrixMask.end(), errValue);
    auto firstError = std::find(flatMatrixMask.begin(), flatMatrixMask.end(), errValue) - flatMatrixMask.begin();
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
    std::cout << "Expected: " << this->flatMat1[firstError] << " ";
    std::cout << "got: " << this->flatMat2[firstError] << std::endl;
}


template class Validator<int>;
