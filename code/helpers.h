#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

// template <typename T, int N>
// class RandomMatrix
// {
//     private:
//         unsigned seed; // Seed used to generate random matrix
//         std::vector<T> flatMat;
//         std::vector<unsigned> dimensions;
//         unsigned flatSize() 
//         {
//             return std::reduce(
//                 this->dimensions.begin(), this->dimensions.end(), 
//                 1, 
//                 std::multiplies<T>()
//             );
//         }
//         // Has to be implemented here since it used non-type type parameter
//         void setDimensions(const unsigned dimensions, ...) 
//         {
//             va_list dims; va_start(dims, dimensions);                
//             this->dimensions.push_back(dimensions);                        
//             for (int i = 0; i < N - 1; i++) 
//             {
//                 T dim = va_arg(dims, unsigned);
//                 this->dimensions.push_back(dim);                
//             }                            
//         }

//     public:        
//         RandomMatrix()
//         {
//             this->setSeed(37);
//         }

//         RandomMatrix(T *flatMat, const unsigned dimensions, ...) 
//         {
//             this->setDimensions(dimensions);          
//             this->flatMat.insert(this->flatMat.begin(), flatMat, flatMat + this->flatSize());            
//         }

//         T* rawData()
//         {
//             return this->flatMat.data();
//         }

//         RandomMatrix<T, N>& setSeed(unsigned s)
//         {
//             srand(s);
//             return *this;
//         }    

//         template <int RANDMAX> void fill(const unsigned dimensions, ...) 
//         {              
//             this->setDimensions(dimensions);
//             std::cout << "Flat size: " << this->flatSize() << std::endl;
//             this->flatMat.resize(this->flatSize());
//             std::generate(this->flatMat.begin(), this->flatMat.end(), [](){                
//                 return rand() / ((T) RANDMAX);
//             });            
            
//         }
// };
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