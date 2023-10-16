#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

template <typename T, int N>
class RandomMatrix
{
    private:
        unsigned seed;
        std::vector<T> flatMat;
        std::vector<unsigned> dimensions;
        unsigned flatSize();
        void setDimensions(const unsigned dimensions, ...);

    public:        
        RandomMatrix();
        RandomMatrix(T *flatMat, const unsigned dimensions, ...);
        T* rawData();
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
        void setEps(int);
        void validate();
};

#include "helpers.tpp"