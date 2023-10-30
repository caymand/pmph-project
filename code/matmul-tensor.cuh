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

template <class accType, class elmType, int wmma_m, int wmma_n, int wmma_k, int warp_tiles_m, int warp_tiles_n, int warp_tiles_k, int block_tiles_m, int block_tiles_n, int block_tiles_k, int threads_per_block>
__global__ void matMulTiledTensor(elmType* A, elmType* B, accType* C, int m, int n, int k) {
    constexpr unsigned int shared_m = wmma_m * warp_tiles_m * block_tiles_m;
    constexpr unsigned int shared_n = wmma_n * warp_tiles_n * block_tiles_n;
    constexpr unsigned int shared_k = wmma_k * warp_tiles_k * block_tiles_k;
//    TODO: use warp_tiles_k

    constexpr int copies_per_thread_A = (shared_m * shared_k + threads_per_block) / threads_per_block;
    constexpr int copies_per_thread_B = (shared_k * shared_n + threads_per_block) / threads_per_block;

    //    Pad to avoid bank conflicts
    constexpr unsigned int A_shared_k_true = shared_k + 8;
    __shared__ elmType A_shared[shared_m][A_shared_k_true];

  // remapping (a slice of) B to shared memory
    //    Pad to avoid bank conflicts
    constexpr unsigned int B_shared_n_true = shared_n + 8;
    __shared__ elmType B_shared[shared_k][B_shared_n_true];

    unsigned int block_m_global_offset = blockIdx.y * shared_m;
    unsigned int block_n_global_offset = blockIdx.x * shared_n;

    unsigned int warpID = threadIdx.x / warpSize;
    unsigned int laneID = threadIdx.x % warpSize;

//    Assumes num_warps >= block_tiles_m * block_tiles_n
//    TODO: change name and value
    unsigned int warp_m_index = warpID / block_tiles_n;
    unsigned int warp_n_index = warpID % block_tiles_n;

//    TODO: change name and value
    unsigned int warp_m_global_offset = block_m_global_offset + warp_m_index * wmma_m * warp_tiles_m;
    unsigned int warp_n_global_offset = block_n_global_offset + warp_n_index * wmma_n * warp_tiles_n;

    for (int global_k_offset = 0; global_k_offset < k; global_k_offset += shared_k) {
//      Copy data to shared memory
        #pragma unroll
        for (int i = 0; i < copies_per_thread_A; i++) {
            unsigned int tile_i = threadIdx.x + i * blockDim.x;
            unsigned int tile_m_index = tile_i / shared_k;
            unsigned int tile_k_index = tile_i % shared_k;
            unsigned int A_m_index = block_m_global_offset + tile_m_index;
            unsigned int A_k_index = global_k_offset + tile_k_index;

            if (tile_m_index < shared_m && tile_k_index < shared_k) {
                A_shared[tile_m_index][tile_k_index] = A_m_index < m && A_k_index < k ? A[A_m_index * k + A_k_index] : (elmType) 0.0;
            }
        }

        #pragma unroll
        for (int i = 0; i < copies_per_thread_B; i++) {
            unsigned int tile_i = threadIdx.x + i * blockDim.x;
            unsigned int tile_k_index = tile_i / shared_n;
            unsigned int tile_n_index = tile_i % shared_n;
            unsigned int B_k_index = global_k_offset + tile_k_index;
            unsigned int B_n_index = block_n_global_offset + tile_n_index;

            if (tile_k_index < shared_k && tile_n_index < shared_n) {
                B_shared[tile_k_index][tile_n_index] = B_k_index < k && B_n_index < n ? B[B_k_index * n + B_n_index] : (elmType) 0.0f;
            }
        }

        __syncthreads();
//      End of copy to shared memory

//        TODO: move check into loop?
        if (warp_m_global_offset < m && warp_n_global_offset < n)
        {
            // compute the per-thread result css:
            //    TODO: move this into loop to free registers for use in copy
            // the thread result is computed in register memory
            // and the global-memory array C is updated at the end.
            wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, accType> C_frag[warp_tiles_m][warp_tiles_n];

//          Assumes C is initialized to zero
            for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i += 1)
            {
                for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i += 1)
                {
                    int m_index = warp_m_global_offset + warp_m_offset_i * wmma_m;
                    int n_index = warp_n_global_offset + warp_n_offset_i * wmma_n;
                    wmma::load_matrix_sync(C_frag[warp_m_offset_i][warp_n_offset_i], &C[m_index * n + n_index], n, wmma::mem_row_major);
                }
            }

            //        TODO: #pragma unroll?
            for (int local_k_offset = 0; local_k_offset < block_tiles_k * wmma_k; local_k_offset += wmma_k)
            {
                wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> A_frag[warp_tiles_m];
                wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> B_frag[warp_tiles_n];
                for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i += 1)
                {
                    wmma::load_matrix_sync(A_frag[warp_m_offset_i], &A_shared[warp_m_index * wmma_m * warp_tiles_m + warp_m_offset_i * wmma_m][local_k_offset], A_shared_k_true);

                    for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i += 1)
                    {
                        wmma::load_matrix_sync(B_frag[warp_n_offset_i], &B_shared[local_k_offset][warp_n_index * wmma_n * warp_tiles_n + warp_n_offset_i * wmma_n], B_shared_n_true);
                        wmma::mma_sync(C_frag[warp_m_offset_i][warp_n_offset_i], A_frag[warp_m_offset_i], B_frag[warp_n_offset_i], C_frag[warp_m_offset_i][warp_n_offset_i]);
                    }
                }
            }

            for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i += 1)
            {
                for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i += 1)
                {
                    int m_index = warp_m_global_offset + warp_m_offset_i * wmma_m;
                    int n_index = warp_n_global_offset + warp_n_offset_i * wmma_n;
                    wmma::store_matrix_sync(&C[m_index * n + n_index], C_frag[warp_m_offset_i][warp_n_offset_i], n, wmma::mem_row_major);
                }
            }
        }
        __syncthreads();
    }

    // Update C in global memory with the per-thread result css.
//    TODO: handle warp matrices that are not full? should pad differently when copying to shared memory
//    if (warp_m_global_offset < m && warp_n_global_offset < n) {
//        wmma::store_matrix_sync(&C[warp_m_global_offset * n + warp_n_global_offset], C_frag, n, wmma::mem_row_major);
//    }
}


#endif //CODE_MATMUL_TENSOR_CUH
