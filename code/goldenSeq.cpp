
#include "goldenSeq.h"

template<class ElTp>
void goldenSeq(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {
    #pragma omp parallel for schedule(static) collapse(2)
    for(int i=0; i<heightA; i++) {
        for(int j=0; j<widthB; j++) {
            ElTp c = 0;
            for(int k=0; k<widthA; k++) {
                c += A[i*widthA +k] * B[k*widthB + j];
            }
            C[i*widthB + j] = c;
        }
    }
}

template void goldenSeq(float* A, float* B, float* C, int heightA, int widthB, int widthA);