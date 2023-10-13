template <class ElTp, int Ty, int Ry, int Tx, int Rx, int Tk>
__global__ void matMulTiled(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA);
