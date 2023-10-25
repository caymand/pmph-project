
#ifndef MULT_KERNELS
#define MULT_KERNELS
#include <stdint.h>
#include <mma.h>
using namespace nvcuda;

template <class ElTp, int Ty, int Ry, int Tx, int Rx, int Tk>
__global__ void matMulTiled(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {

  int gid = threadIdx.x + threadIdx.y * blockDim.x;
  if (gid >= widthA * heightA || gid >= widthA * widthB) { return; }

  // remapping (a slice of) A to shared memory
  __shared__ ElTp Aloc[Ty*Ry][Tk+1];

  // remapping (a slice of) B to shared memory
  __shared__ ElTp Bloc[Tk][Tx*Rx+1]; 

  // the thread result is computed in register memory
  // and the global-memory array C is updated at the end.
  ElTp css[Ry][Rx];

  unsigned int iii = blockIdx.y * Ty * Ry;
  unsigned int jjj = blockIdx.x * Tx * Rx;

  // initialize the result with zero
  // (the neutral element for addition)
  #pragma unroll
  for(int i=0; i<Ry; i++)
      #pragma unroll
      for(int j=0; j<Rx; j++)
          css[i][j] = 0.0;

  for(int kk = 0; kk < widthA; kk += Tk) {
      #pragma unroll
      for (uint32_t r = 0; r < Ry; r++) {
        // Stack R blocks of size Ty x Tx on top of each other          
        uint32_t local_x = threadIdx.x;
        uint32_t local_y = threadIdx.y + Ty * r; // stack Ry blocks on top of each other          

        uint32_t slice_y = iii + local_y; // this gives [iii : iii + Ty*Ry]
        uint32_t slice_x = kk + threadIdx.x;// This is [kk: kk + Tk] 

        bool insideBounds = (slice_y < heightA) && (slice_x < widthA);
        Aloc[local_y][local_x] = insideBounds ? A[slice_y * widthA + slice_x] : (ElTp) 0.0;
      }
      
      #pragma unroll
      for (uint32_t r = 0; r < Rx; r++) {          
          uint32_t local_y = threadIdx.y;
          uint32_t local_x = threadIdx.x + Tx*r; // stack Rx blocks next to each other

          uint32_t slice_y = kk + threadIdx.y;// [kk : kk + Tk] 
          uint32_t slice_x = jjj + local_x; // [jjj : jjj + Tx*Rx] 
          
          bool insideBounds = (slice_y < widthA) && (slice_x < widthB);
          Bloc[local_y][local_x] = insideBounds ? B[slice_y * widthB + slice_x] : (ElTp) 0.0;
      }

      __syncthreads();

      // compute the per-thread result css:
      for(int k = 0; k < Tk; k++) {
          #pragma unroll
          for(int i=0; i<Ry; i++) {
              #pragma unroll
              for(int j=0; j<Rx; j++) {
                float Aik = Aloc[threadIdx.y * Ry + i][k];
                float Bkj = Bloc[k][threadIdx.x * Rx + j];
                css[i][j] += Aik * Bkj;                

              }
          }
      }
      __syncthreads();
  }

  unsigned int indy = iii + threadIdx.y * Ry;
  unsigned int indx = jjj + threadIdx.x * Rx;

  // Update C in global memory with the per-thread result css.
  #pragma unroll
  for(int i=0; i<Ry; i++) {
    #pragma unroll
    for(int j=0; j<Rx; j++) {
      if( (indy+i < heightA) && (indx+j < widthB) )
        C[(indy+i)*widthB + (indx+j)] = css[i][j];
    }
  }
}

template <class ElTp, class ElHalfTp, int Ty, int Ry, int Tx, int Rx, int Tk>
__global__ void matMulTensor(ElTp* A, ElTp* B, ElTp* C, int heightA, int widthB, int widthA) {

  int gid = threadIdx.x + threadIdx.y * blockDim.x;
  if (gid >= widthA * heightA || gid >= widthA * widthB) { return; }
  // These are the fragment sizes
  constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
  // Leading axis size for matrix A, B and C 
  constexpr int lda = WMMA_M, ldb = WMMA_K, ldc = WMMA_K;
  // remapping (a slice of) A to shared memory
  __shared__ ElHalfTp Aloc[Ty*Ry][Tk+1];

  // remapping (a slice of) B to shared memory
  __shared__ ElHalfTp Bloc[Tk][Tx*Rx+1]; 

  // Fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, ElHalfTp, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, ElHalfTp, wmma::row_major> b_frag;  
  // We need it to be in an array because we do "warp tiling" to increase register usage
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, ElTp> c_frag[Ry][Rx];
  // the thread result is computed in register memory
  // and the global-memory array C is updated at the end.
  // ElTp css[Ry][Rx];

  unsigned int iii = blockIdx.y * Ty * Ry;
  unsigned int jjj = blockIdx.x * Tx * Rx;

  // initialize the result with zero
  // (the neutral element for addition)
  // #pragma unroll
  // for(int i=0; i<Ry; i++)
  //     #pragma unroll
  //     for(int j=0; j<Rx; j++)
  //         css[i][j] = 0.0;
  for (int i = 0; i < Ry; i++) {
    for (int j = 0; j < Rx; j++) {
      wmma::fill_fragment(c_frag[i][j], 0.0f);
    }
  }
  // Ensure all accumulator fragments have 0.0f
  __syncthreads();

  for(int kk = 0; kk < widthA; kk += Tk) {
      #pragma unroll
      for (uint32_t r = 0; r < Ry; r++) {
        // Stack R blocks of size Ty x Tx on top of each other          
        uint32_t local_x = threadIdx.x;
        uint32_t local_y = threadIdx.y + Ty * r; // stack Ry blocks on top of each other          

        uint32_t slice_y = iii + local_y; // this gives [iii : iii + Ty*Ry]
        uint32_t slice_x = kk + threadIdx.x;// This is [kk: kk + Tk] 

        bool insideBounds = (slice_y < heightA) && (slice_x < widthA);
        Aloc[local_y][local_x] = (ElHalfTp)(insideBounds ? A[slice_y * widthA + slice_x] : 0.0f);
      }
      
      #pragma unroll
      for (uint32_t r = 0; r < Rx; r++) {          
          uint32_t local_y = threadIdx.y;
          uint32_t local_x = threadIdx.x + Tx*r; // stack Rx blocks next to each other

          uint32_t slice_y = kk + threadIdx.y;// [kk : kk + Tk] 
          uint32_t slice_x = jjj + local_x; // [jjj : jjj + Tx*Rx] 
          
          bool insideBounds = (slice_y < widthA) && (slice_x < widthB);
          Bloc[local_y][local_x] = (ElHalfTp)(insideBounds ? B[slice_y * widthB + slice_x] : 0.0f);  
      }

      __syncthreads();

      // compute the per-thread result css:
      for(int k = 0; k < Tk; k++) {
          #pragma unroll
          for(int i=0; i<Ry; i++) {
              // Place it here so we do not load into registers at each "j" loop iterations
              ElHalfTp *a_ptr = &Aloc[threadIdx.y * Ry + i][k];
              wmma::load_matrix_sync(a_frag, a_ptr, lda);
              #pragma unroll
              for(int j=0; j<Rx; j++) {
                ElHalfTp *b_ptr = &Bloc[k][threadIdx.x * Rx + j];
                wmma::load_matrix_sync(a_frag, a_ptr, lda); 
                // This computes D = A*B + C. If we let D=C then it is basically the same
                // as when we did css[i][j] = css[i][j] + Aij * Bkj
                wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);

              }
          }
      }
      __syncthreads();
  }

  unsigned int indy = iii + threadIdx.y * Ry;
  unsigned int indx = jjj + threadIdx.x * Rx;

  // Update C in global memory with the per-thread result css.
  #pragma unroll
  for(int i=0; i<Ry; i++) {
    #pragma unroll
    for(int j=0; j<Rx; j++) {
      if( (indy+i < heightA) && (indx+j < widthB) ) {
        // C[(indy+i)*widthB + (indx+j)] = css[i][j];
        ElTp *c_ptr = C + (indy + i)*widthB + indx+j;
        wmma::store_matrix_sync(c_ptr, c_frag[i][j], WMMA_K * Rx, wmma::mem_row_major);
      }
    }
  }
}


#endif
