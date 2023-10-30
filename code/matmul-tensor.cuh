//
// Created by runeebl on 10/23/23.
//

#ifndef CODE_MATMUL_TENSOR_CUH
#define CODE_MATMUL_TENSOR_CUH

#include <stdint.h>
#include <mma.h>
#include "matmul-tensor.cuh"
#include "cuda_fp16.h"

using namespace nvcuda;

template <class accType, class elmType, int wmma_m, int wmma_n, int wmma_k, int block_tiles_m, int block_tiles_n, int block_tiles_k, int copies_per_thread_A, int copies_per_thread_B>
__global__ void matMulTiledTensor(elmType* A, elmType* B, accType* C, int m, int n, int k) {
    constexpr unsigned int A_loc_m = block_tiles_m * wmma_m;
    constexpr unsigned int A_loc_k = wmma_k * block_tiles_k;
    //    Pad to avoid bank conflicts
    constexpr unsigned int A_loc_k_true = A_loc_k + 8;
    __shared__ elmType A_loc[A_loc_m][A_loc_k_true];

  // remapping (a slice of) B to shared memory
    constexpr unsigned int B_loc_k = block_tiles_k * wmma_k;
    constexpr unsigned int B_loc_n = wmma_n * block_tiles_n;
    //    Pad to avoid bank conflicts
    constexpr unsigned int B_loc_n_true = B_loc_n + 8;
    __shared__ elmType B_loc[B_loc_k][B_loc_n_true];

    unsigned int block_start_m = blockIdx.y * block_tiles_m * wmma_m;
    unsigned int block_start_n = blockIdx.x * block_tiles_n * wmma_n;

    unsigned int warpID = threadIdx.x / warpSize;
    unsigned int laneID = threadIdx.x % warpSize;

//    Assumes num_warps >= block_tiles_m * block_tiles_n
    unsigned int warpID_m = warpID / block_tiles_n;
    unsigned int warpID_n = warpID % block_tiles_n;

    unsigned int ind_m = block_start_m + warpID_m * wmma_m;
    unsigned int ind_n = block_start_n + warpID_n * wmma_n;

    for (int global_offset_k = 0; global_offset_k < k; global_offset_k += block_tiles_k * wmma_k) {
//      Copy data to shared memory
        #pragma unroll
        for (int i = 0; i < copies_per_thread_A; i++) {
            int tile_i = threadIdx.x + i * blockDim.x;
            int tile_m = tile_i / A_loc_k;
            int tile_k = tile_i % A_loc_k;
            int A_m = block_start_m + tile_m;
            int A_k = global_offset_k + tile_k;

            if (tile_m < A_loc_m && tile_k < A_loc_k) {
                A_loc[tile_m][tile_k] = A_m < m && A_k < k ? A[A_m * k + A_k] : (elmType) 0.0;
            }
        }

        #pragma unroll
        for (int i = 0; i < copies_per_thread_B; i++) {
            int tile_i = threadIdx.x + i * blockDim.x;
            int tile_k = tile_i / B_loc_n;
            int tile_n = tile_i % B_loc_n;
            int B_k = global_offset_k + tile_k;
            int B_n = block_start_n + tile_n;

            if (tile_k < B_loc_k && tile_n < B_loc_n) {
                B_loc[tile_k][tile_n] = B_k < k && B_n < n ? B[B_k * n + B_n] : (elmType) 0.0f;
            }
        }

        __syncthreads();
//      End of copy to shared memory

//        TODO: check or not?
        if (ind_m < m && ind_n < n)
        {
            // compute the per-thread result css:
            //    TODO: move this into loop to free registers for use in copy
            // the thread result is computed in register memory
            // and the global-memory array C is updated at the end.
            wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, accType> C_frag;

//          Assumes C is initialized to zero
            wmma::load_matrix_sync(C_frag, &C[ind_m * n + ind_n], n, wmma::mem_row_major);

            //        TODO: #pragma unroll?
            for (int local_offset_k = 0; local_offset_k < block_tiles_k * wmma_k; local_offset_k += wmma_k)
            {
                //            TODO: check col_major vs row_major
                wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> A_frag;
                wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> B_frag;

                //            TODO: tile this to reuse loaded fragments?
                wmma::load_matrix_sync(A_frag, &A_loc[warpID_m * wmma_m][local_offset_k], A_loc_k_true);
                wmma::load_matrix_sync(B_frag, &B_loc[local_offset_k][warpID_n * wmma_n], B_loc_n_true);

                wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
            }

            wmma::store_matrix_sync(&C[ind_m * n + ind_n], C_frag, n, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // Update C in global memory with the per-thread result css.
//    TODO: handle warp matrices that are not full? should pad differently when copying to shared memory
//    if (ind_m < m && ind_n < n) {
//        wmma::store_matrix_sync(&C[ind_m * n + ind_n], C_frag, n, wmma::mem_row_major);
//    }
}


#endif //CODE_MATMUL_TENSOR_CUH
