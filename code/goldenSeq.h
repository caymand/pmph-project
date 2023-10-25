#ifndef GOLDEN
#define GOLDEN

/**
 * Computes matrix multiplication C = A*B
 * Semantically the matrix sizes are:
 *    A : [heightA][widthA]ElTp
 *    B : [ widthA][widthB]ElTp
 *    C : [heightA][widthB]ElTp
 *  for some numeric type ElTp.
 **/
template<class ElTp>
void goldenSeq(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA);

// template void goldenSeq(float* A, float* B, float* C, int heightA, int widthB, int widthA);

#endif
