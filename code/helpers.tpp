#pragma once
#include <sys/time.h>
#include <algorithm>
#include <cstdarg>
//#include <cuda_runtime.h>

// Validator
template <typename T>
Validator<T>::Validator(T *flatMat1, T* flatMat2, int N) 
{
    this->flatMat1.insert(this->flatMat1.begin(), flatMat1, flatMat1 + N);
    this->flatMat2.insert(this->flatMat2.begin(), flatMat2, flatMat2 + N);
    this->eps = 0.0000000001; // e-10
}
template <typename T>
void Validator<T>::setEps(float eps) {
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
    std::printf("Found: %d wrong elements\n", errorCount);
    std::cout << "First at flat index: " << firstError << std::endl;
    std::printf("Expected %.5f, got %.5f\n", this->flatMat1[firstError], this->flatMat2[firstError]);
}

// RandomMatrix

//Private 
template <typename T, int N>
unsigned RandomMatrix<T, N>::flatSize() 
{
    return std::accumulate(
        this->dimensions.begin(), this->dimensions.end(), 
        1, 
        std::multiplies<T>()
    );
}
template <typename T, int N>
void RandomMatrix<T, N>::setDimensions(const unsigned dimensions, ...) 
{
    va_list dims; va_start(dims, dimensions);                
    this->dimensions.push_back(dimensions);                        
    for (int i = 0; i < N - 1; i++) 
    {
        T dim = va_arg(dims, unsigned);
        this->dimensions.push_back(dim);                
    }                            
}
template <typename T, int N>
RandomMatrix<T, N>::RandomMatrix()
{
    this->setSeed(37);
}

template <typename T, int N>
RandomMatrix<T, N>::RandomMatrix(T *flatMat, const unsigned dimensions, ...) 
{
    this->setDimensions(dimensions);        
    this->flatMat.resize(this->flatSize());
}
template <typename T, int N>
T* RandomMatrix<T, N>::to_cpu()
{
    return this->flatMat.data();
}

template <typename T, int N>
T* RandomMatrix<T, N>::to_gpu() 
{
    void *gpu_mem;
    if (!gpuAssert(cudaMalloc(&gpu_mem, this->flatSize() * sizeof(T)))) 
    {
        std::cout << "GPU memory allocation error" << std::endl;     
    }
    gpuAssert(cudaMemcpy(gpu_mem, this->to_cpu(), sizeof(T) * this->flatSize(), cudaMemcpyHostToDevice));
    return (T *) gpu_mem;
}

template <typename T, int N>
RandomMatrix<T, N>& RandomMatrix<T, N>::setSeed(unsigned s)
{
    srand(s);
    return *this;
}    
template <typename T, int N>
template <int RANDMAX> void 
RandomMatrix<T, N>::fill(const unsigned dimensions, ...) 
{              
    this->setDimensions(dimensions);
    unsigned flatSize = this->flatSize();
    // Why is this UB without the print?
    this->flatMat.resize(this->flatSize());
    std::cout << "Capacity: " << this->flatSize()  << std::endl;
    std::generate(this->flatMat.begin(), this->flatMat.end(), [](){                
        return rand() / ((T) RANDMAX);
    });            
    
}

// TimeMeasurment
TimeMeasurement::TimeMeasurement(int resolution) {
    this ->resolution = resolution;
}
int TimeMeasurement::start() 
{    
    return gettimeofday(&this->tstart, NULL);
}
int TimeMeasurement::stop() 
{
    return gettimeofday(&this->tend, NULL);
}
long int TimeMeasurement::elapsed() 
{
    struct timeval &t2 = this->tend;
    struct timeval &t1 = this->tstart;
    long int diff = (
        (t2.tv_usec + this->resolution * t2.tv_sec) - 
        (t1.tv_usec + this->resolution * t1.tv_sec)
    );
    return diff > 0 ? diff : -1;
}
