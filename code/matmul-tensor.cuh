//
// Created by runeebl on 10/23/23.
//

#ifndef CODE_MATMUL_TENSOR_CUH
#define CODE_MATMUL_TENSOR_CUH
//
//template <class accType, class elmType, int wmma_m, int wmma_n, int wmma_k, int block_tile_size>
//__global__ void matMulTiledTensor(elmType* A, elmType* B, accType* C, int heightA, int widthB, int widthA);

#include <stdint.h>
#include <mma.h>
#include "matmul-tensor.cuh"

using namespace nvcuda;

template <class accType, class elmType, int wmma_m, int wmma_n, int wmma_k, int block_tile_size>
__global__ void matMulTiledTensor(elmType* A, elmType* B, accType* C, int heightA, int widthB, int widthA) {
//    TODO: check sequential_tiles, tile with multiple fragments?
//    TODO: check if block_width and block_height (and sequential size) are needed

  // remapping (a slice of) A to shared memory
//  TODO: try without +1, check width vs height, M vs K vs N
    const unsigned int A_loc_height = block_tile_size * wmma_m;
    const unsigned int A_loc_width = wmma_k * (block_tile_size + 1);
    __shared__ elmType A_loc[A_loc_height][A_loc_width];

  // remapping (a slice of) B to shared memory
    const unsigned int B_loc_height = block_tile_size * wmma_k;
    const unsigned int B_loc_width = wmma_n * (block_tile_size + 1);
    __shared__ elmType B_loc[B_loc_height][B_loc_width];

  // the thread result is computed in register memory
  // and the global-memory array C is updated at the end.
//  ElTp css[Ry][Rx];
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, accType> C_frag;

    unsigned int block_start_y = blockIdx.y * block_tile_size * wmma_m;
    unsigned int block_start_x = blockIdx.x * block_tile_size * wmma_n;

//    TODO: bitshift?
    unsigned int warpIDx = threadIdx.x / warpSize;
    unsigned int warpIDy = threadIdx.y / warpSize;

    // initialize the result with zero
    // (the neutral element for addition)
//    #pragma unroll
//    for(int i=0; i<Ry; i++)
//      #pragma unroll
//      for(int j=0; j<Rx; j++)
//          css[i][j] = 0.0;
    wmma::fill_fragment(C_frag, 0.0f);

    for(int global_iteration_offset = 0; global_iteration_offset < widthA; global_iteration_offset += block_tile_size) {
        #pragma unroll
        for (unsigned int i_m = 0; i_m < wmma_m; i_m++) {
//            TODO: check this
            #pragma unroll
            for (unsigned int i_k = 0; i_k < wmma_k; i_k++) {
                unsigned int tile_y = threadIdx.y + i_m * block_tile_size;
                unsigned int tile_x = threadIdx.x + i_k * block_tile_size;
                unsigned int A_y = block_start_y + tile_y;
                unsigned int A_x = global_iteration_offset + tile_x;
                A_loc[tile_y][tile_x] = A_y < heightA && A_x < widthA ? A[A_y * widthA + A_x] : (elmType) 0.0;
            }
        }


//            TODO: check this
        #pragma unroll
        for (unsigned int i_k = 0; i_k < wmma_k; i_k++) {
            #pragma unroll
            for (unsigned int i_n = 0; i_n < wmma_n; i_n++)
            {
                unsigned int tile_y = threadIdx.y + i_k * block_tile_size;
                unsigned int tile_x = threadIdx.x + i_n * block_tile_size;
                unsigned int B_y = global_iteration_offset + tile_y;
                unsigned int B_x = block_start_x + tile_x;
                B_loc[tile_y][tile_x] = B_y < widthA && B_x < widthB ? B[B_y * widthB + B_x] : (elmType) 0.0;
            }
        }

        __syncthreads();

        // compute the per-thread result css:
        for(int k = 0; k < block_tile_size; k++) {
//          #pragma unroll
//          for(int i=0; i < Ry; i++) {
//              #pragma unroll
//              for(int j=0; j < Rx; j++) {
//                  css[i][j] +=
//                          A_loc[threadIdx.y*Ry + i][k] *
//                          B_loc[k][threadIdx.x*Rx + j];
//              }
//          }

//            TODO: check col_major vs row_major
            wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> A_frag;
            wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> B_frag;

//            TODO: tile this to reuse loaded fragments?
            wmma::load_matrix_sync(A_frag, &A_loc[warpIDy * wmma_m][k * wmma_k], A_loc_width);
            wmma::load_matrix_sync(B_frag, &B_loc[k * wmma_k][warpIDx * wmma_n], B_loc_width);

            wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        }
        __syncthreads();
    }

    unsigned int ind_y = block_start_y + warpIDy * wmma_m;
    unsigned int ind_x = block_start_x + warpIDx * wmma_n;

    // Update C in global memory with the per-thread result css.
//    #pragma unroll
//    for(int i=0; i<Ry; i++) {
//        #pragma unroll
//        for(int j=0; j<Rx; j++) {
//            if( (indy+i < heightA) && (indx+j < widthB) )
//                C[(indy+i)*widthB + (indx+j)] = css[i][j];
//        }
//    }
    wmma::store_matrix_sync(&C[ind_y * widthB + ind_x], C_frag, widthB, wmma::mem_row_major);
}


#endif //CODE_MATMUL_TENSOR_CUH
