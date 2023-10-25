#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <sys/time.h>
#include <cuda_runtime.h>

template <typename T, int N>
class RandomMatrix
{
    private:
        unsigned seed;
        std::vector<T> flatMat;
        std::vector<unsigned> dimensions;        
        void setDimensions(const unsigned dimensions, ...);

    public:                
        RandomMatrix();
        RandomMatrix(T *flatMat, const unsigned dimensions, ...);
        T* to_cpu();
        T* to_gpu();
        unsigned flatSize();
        RandomMatrix<T, N>& setSeed(unsigned s);        
        template <int RANDMAX> void fill(const unsigned dimensions, ...);                
};


template <typename T>
class Validator
{
    private:
        std::vector<T> flatMat1;
        std::vector<T> flatMat2;
        float eps;
    public:
        Validator(T *flatMat1, T *flatMat2, int N);
        void setEps(float);
        void validate();
};

class TimeMeasurement
{
    private:
        int resolution;
        timeval tstart;
        timeval tend;        
    public:
        TimeMeasurement(int resolution = 1000000);
        int start();
        int stop();
        long int elapsed();
        
};

int printPerformanceMetric(long int elapsed_time_us, unsigned long total, const char *metric_name) 
{
    double elapsed_sec = elapsed_time_us * 1e-6f;
    double metric = (total / elapsed_sec) * 1.0e-9f; // make it in Giga scale
    std::cout << metric_name << " : " << metric << std::endl;
    return metric < 0;
}
int printGBSec(long int elapsed_time_us, unsigned long total_memsize) 
{
    return printPerformanceMetric(elapsed_time_us, total_memsize, "GB/sec");
}

int printGFlops(long int elapsed_time_us, unsigned long total_flops) 
{
    return printPerformanceMetric(elapsed_time_us, total_flops, "GFlops/sec");
}

int gpuAssert(cudaError_t code) 
{
    if(code != cudaSuccess) {
        printf("GPU Error: %s\n", cudaGetErrorString(code));
        exit(1);
    }
    return 1;
}


#include "helpers.tpp"