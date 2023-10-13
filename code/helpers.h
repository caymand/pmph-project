#include <vector>

template <typename T>
class Validator
{
    private:
        std::vector<T> res1;
        std::vector<T> res2;
        float eps;
    public:
        Validator(T *res1, T *res2, int N);
        void setEps(int);
        void validate();
};