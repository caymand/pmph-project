#pragma once
#include <sys/time.h>
#include <algorithm>
#include <cstdarg>

// Validator
template <typename T>
Validator<T>::Validator(T *flatMat1, T* flatMat2, int N) 
{
    this->flatMat1.insert(this->flatMat1.begin(), flatMat1, flatMat1 + N);
    this->flatMat2.insert(this->flatMat2.begin(), flatMat2, flatMat2 + N);
    this->eps = 0.0000000001; // e-10
}
template <typename T>
void Validator<T>::setEps(T eps) {
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
//            float err = (float)std::abs(v1 - v2) / std::max(v1, v2);
//            return err < this->eps;

//            Calculate err without using abs
            T err = (v1 - v2) / std::max(v1, v2);

//            printf("v1: %.5f, v2: %.5f, err: %.5f\n", (float) v1, (float) v2, (float) err);

            return err < (T) 0.0 ? -err < this->eps : err < this->eps;
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
    std::printf("Expected %.5f, got %.5f\n", (float) this->flatMat1[firstError], (float) this->flatMat2[firstError]);
}

// RandomMatrix

//Private 
template <typename T, int N>
unsigned RandomMatrix<T, N>::flatSize() 
{
    unsigned acc = 1;

    for (auto dim : this->dimensions) {
        acc *= dim;
    }
    return acc;
}
template <typename T, int N>
void RandomMatrix<T, N>::setDimensions(unsigned first_dim, va_list dimensions) 
{
    // va_list dims; va_start(dims, dimensions);                
    this->dimensions.push_back(first_dim);                        
    for (int i = 0; i < N - 1; i++) 
    {
        unsigned dim = va_arg(dimensions, unsigned);        
        this->dimensions.push_back(dim);                
    }      
}
template <typename T, int N>
RandomMatrix<T, N>::RandomMatrix()
{
    this->setSeed(37);
}

template <typename T, int N>
template <typename U>
void RandomMatrix<T, N>::fill_from(RandomMatrix<U, N> &other, const unsigned first_dim, ...) 
{
    va_list remaining_dims; va_start(remaining_dims, first_dim);      
    this->setDimensions(first_dim, remaining_dims);            
    U *other_flat_mat = other.to_cpu();
    for (int i = 0; i < other.flatSize(); i++) {
        U v = other_flat_mat[i];
        this->flatMat.push_back((T) v);
    }
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
    size_t n_bytes = this->flatSize() * sizeof(T);
    if (!gpuAssert(cudaMalloc(&gpu_mem, n_bytes))) 
    {
        std::cout << "GPU memory allocation error" << std::endl;     
    }
    gpuAssert(cudaMemcpy(gpu_mem, this->to_cpu(), n_bytes, cudaMemcpyHostToDevice));
    return (T *) gpu_mem;
}

template <typename T, int N>
RandomMatrix<T, N>& RandomMatrix<T, N>::setSeed(unsigned s)
{
    srand(s);
    return *this;
}    
template <typename T, int N>
template <int RANDMAX> 
void RandomMatrix<T, N>::fill_rand(const unsigned first_dim, ...)
{              
    va_list remaining_dims; va_start(remaining_dims, first_dim);      
    this->setDimensions(first_dim, remaining_dims);
    this->flatMat.resize(this->flatSize());
    std::cout << "Capacity: " << this->flatSize()  << std::endl;
    std::generate(this->flatMat.begin(), this->flatMat.end(), [](){                
        return (T) (rand() / ((float) RANDMAX));
    });
}

template <typename T, int N>
void RandomMatrix<T, N>::fill(T value, const unsigned first_dim, ...)
{
    va_list remaining_dims; va_start(remaining_dims, first_dim);      
    this->setDimensions(first_dim, remaining_dims);
    std::cout << "Capacity: " << this->flatSize()  << std::endl;
    this->flatMat.resize(this->flatSize());
    std::fill(this->flatMat.begin(), this->flatMat.end(), 0);
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
