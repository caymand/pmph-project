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

template <class accType, class elmType, int wmma_m, int wmma_n, int wmma_k, int block_tiles_m, int block_tiles_n, int block_tiles_k, int copies_per_thread_A, int copies_per_thread_B>
__global__ void matMulTiledTensor(elmType* A, elmType* B, accType* C, int m, int n, int k) {
//    TODO: calculate block_tiles sizes based on threads per block?

//    , int copies_per_thread_A, int copies_per_thread_B
//    TODO: calculate copies per thread?
//    TODO: check optimal block sizes
//    int block_tiles_m = blockDim.y;
//    int block_tiles_n = blockDim.x / warpSize;

//    TODO: ideally units used for copying matches units used for computation
//    TODO: can we assume numThreads (blockDim.x) >= block_tiles_m * wmma_m * block_tiles_k * wmma_k?

    int num_warps = blockDim.x / warpSize;

//    TODO: ensure block sizes match number of available warps

  // remapping (a slice of) A to shared memory
//  TODO: try without +1
    constexpr unsigned int A_loc_m = block_tiles_m * wmma_m;
    constexpr unsigned int A_loc_k = wmma_k * (block_tiles_k + 1);
    __shared__ elmType A_loc[A_loc_m][A_loc_k];

  // remapping (a slice of) B to shared memory
    constexpr unsigned int B_loc_k = block_tiles_k * wmma_k;
    constexpr unsigned int B_loc_n = wmma_n * (block_tiles_n + 1);
    __shared__ elmType B_loc[B_loc_k][B_loc_n];


  // the thread result is computed in register memory
  // and the global-memory array C is updated at the end.
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, accType> C_frag;

    unsigned int block_start_m = blockIdx.y * block_tiles_m * wmma_m;
    unsigned int block_start_n = blockIdx.x * block_tiles_n * wmma_n;

//    TODO: bitshift?
//    TODO: this is wrong
    unsigned int warpID = threadIdx.x / warpSize;

//    Assumes num_warps >= block_tiles_m * block_tiles_n
    unsigned int warpID_m = warpID / block_tiles_n;
    unsigned int warpID_n = warpID % block_tiles_n;

    // initialize the result with zero
    // (the neutral element for addition)
    wmma::fill_fragment(C_frag, 0.0f);

    for (int global_offset_k = 0; global_offset_k < k; global_offset_k += block_tiles_k * wmma_k) {
//        TODO: check this
//      Copy data to shared memory

        // Must copy block_tiles_m * wmma_m * block_tiles_k * wmma_k elements from A and similar for B

        #pragma unroll
        for (int i = 0; i < copies_per_thread_A; i++) {
            int tile_i = threadIdx.x + i * blockDim.x;
            int tile_m = tile_i / A_loc_k;
            int tile_k = tile_i % A_loc_k;
            int A_m = block_start_m + tile_m;
            int A_k = global_offset_k + tile_k;
            A_loc[tile_m][tile_k] = A_m < m && A_k < k ? A[A_m * k + A_k] : (elmType) 0.0;
        }

        #pragma unroll
        for (int i = 0; i < copies_per_thread_B; i++) {
            int tile_i = threadIdx.x + i * blockDim.x;
            int tile_k = tile_i / B_loc_n;
            int tile_n = tile_i % B_loc_n;
            int B_k = global_offset_k + tile_k;
            int B_n = block_start_n + tile_n;
            B_loc[tile_k][tile_n] = B_k < k && B_n < n ? B[B_k * n + B_n] : (elmType) 0.0f;
        }

        __syncthreads();
//      End of copy to shared memory


        // compute the per-thread result css:
//        TODO: #pragma unroll?
        for(int local_offset_k = 0; local_offset_k < block_tiles_k * wmma_k; local_offset_k += wmma_k) {
//            TODO: check col_major vs row_major
            wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> A_frag;
            wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> B_frag;

//            TODO: tile this to reuse loaded fragments?
//            TODO: ensure warpID_m * wmma_m spans entire array, similar for warpID_n * wmma_n
            wmma::load_matrix_sync(A_frag, &A_loc[warpID_m * wmma_m][local_offset_k], A_loc_k);
            wmma::load_matrix_sync(B_frag, &B_loc[local_offset_k][warpID_n * wmma_n], B_loc_n);

            wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        }
        __syncthreads();
    }


    // Update C in global memory with the per-thread result css.
    //    TODO: check this
    unsigned int ind_m = block_start_m + warpID_m * wmma_m;
    unsigned int ind_n = block_start_n + warpID_n * wmma_n;

    wmma::store_matrix_sync(&C[ind_m * n + ind_n], C_frag, n, wmma::mem_row_major);
}


#endif //CODE_MATMUL_TENSOR_CUH
