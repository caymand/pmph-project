//
// Created by runeebl on 10/23/23.
//

#ifndef CODE_MATMUL_TENSOR_CUH
#define CODE_MATMUL_TENSOR_CUH

//#define CACHE_C
//#define KEEP_C


#include <stdint.h>
#include <mma.h>
#include "matmul-tensor.cuh"
#include "cuda_fp16.h"

using namespace nvcuda;

// TODO: check reads coalesced, check store is coalesced

template <class accType, class elmType, int wmma_m, int wmma_n, int wmma_k, int warp_tiles_m, int warp_tiles_n, int warp_tiles_k, int block_tiles_m, int block_tiles_n, int block_tiles_k, int threads_per_block>
__global__ void matMulTiledTensor(elmType* A, elmType* B, accType* C, int m, int n, int k) {
    constexpr unsigned int shared_m = wmma_m * warp_tiles_m * block_tiles_m;
    constexpr unsigned int shared_n = wmma_n * warp_tiles_n * block_tiles_n;
    constexpr unsigned int shared_k = wmma_k * warp_tiles_k * block_tiles_k;
//    TODO: use warp_tiles_k

    constexpr int copies_per_thread_A = (shared_m * shared_k + threads_per_block) / threads_per_block;
    constexpr int copies_per_thread_B = (shared_k * shared_n + threads_per_block) / threads_per_block;

//    TODO: try moving these to use?
    unsigned int block_m_global_offset = blockIdx.y * shared_m;
    unsigned int block_n_global_offset = blockIdx.x * shared_n;

    unsigned int warpID = threadIdx.x / warpSize;
//    unsigned int laneID = threadIdx.x % warpSize;

//    Assumes num_warps >= block_tiles_m * block_tiles_n
    unsigned int warp_m_index = warpID / block_tiles_n;
    unsigned int warp_n_index = warpID % block_tiles_n;

    unsigned int warp_m_shared_offset = warp_m_index * wmma_m * warp_tiles_m;
    unsigned int warp_n_shared_offset = warp_n_index * wmma_n * warp_tiles_n;

    unsigned int warp_m_global_offset = block_m_global_offset + warp_m_shared_offset;
    unsigned int warp_n_global_offset = block_n_global_offset + warp_n_shared_offset;

    //    Pad to avoid bank conflicts
    constexpr unsigned int A_shared_k_true = shared_k + 8;
    __shared__ elmType A_shared[shared_m][A_shared_k_true];

    //    Pad to avoid bank conflicts
    constexpr unsigned int B_shared_n_true = shared_n + 8;
    __shared__ elmType B_shared[shared_k][B_shared_n_true];

#ifdef CACHE_C
    constexpr int copies_per_thread_C = (shared_m * shared_n + threads_per_block) / threads_per_block;

    //    Pad to avoid bank conflicts
    constexpr unsigned int C_shared_n_true = shared_n + 8;
    __shared__ accType C_shared[shared_m][C_shared_n_true];

    //    Initialize C cache to zero
    #pragma unroll
    for (int i = 0; i < copies_per_thread_C; i++) {
        unsigned int tile_i = threadIdx.x + i * blockDim.x;
        unsigned int tile_m_index = tile_i / shared_n;
        unsigned int tile_n_index = tile_i % shared_n;

        if (tile_m_index < shared_m && tile_n_index < shared_n) {
            //            TODO: try to avoid ternary statement
            C_shared[tile_m_index][tile_n_index] = (accType) 0.0f;
        }
    }
#else
#ifdef KEEP_C
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, accType> C_frag[warp_tiles_m][warp_tiles_n];

//    Initialize C_frag to zero
    #pragma unroll
    for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
    {
        #pragma unroll
        for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i++)
        {
            wmma::fill_fragment(C_frag[warp_m_offset_i][warp_n_offset_i], 0.0f);
        }
    }
#endif
#endif

    for (int global_k_offset = 0; global_k_offset < k; global_k_offset += shared_k) {
//      Copy A and B to shared memory
        #pragma unroll
        for (int i = 0; i < copies_per_thread_A; i++) {
            unsigned int tile_i = threadIdx.x + i * blockDim.x;
            unsigned int tile_m_index = tile_i / shared_k;
            unsigned int tile_k_index = tile_i % shared_k;
            unsigned int A_m_index = block_m_global_offset + tile_m_index;
            unsigned int A_k_index = global_k_offset + tile_k_index;

//            TODO: try to avoid ternary statement
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
                //            TODO: try to avoid ternary statement
                B_shared[tile_k_index][tile_n_index] = B_k_index < k && B_n_index < n ? B[B_k_index * n + B_n_index] : (elmType) 0.0f;
            }
        }

        __syncthreads();
//      End of copy to shared memory

//        TODO: move check into loop?
        if (warp_m_global_offset < m && warp_n_global_offset < n) {
#ifndef KEEP_C
//  Load C from memory
            wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, accType> C_frag[warp_tiles_m][warp_tiles_n];

//          Assumes C is initialized to zero
            #pragma unroll
            for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
            {
                #pragma unroll
                for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i++)
                {
#ifdef CACHE_C
                    int m_index = warp_m_shared_offset + warp_m_offset_i * wmma_m;
                    int n_index = warp_n_shared_offset + warp_n_offset_i * wmma_n;
                    wmma::load_matrix_sync(C_frag[warp_m_offset_i][warp_n_offset_i], &C_shared[m_index][n_index], C_shared_n_true, wmma::mem_row_major);
#else
                    int m_index = warp_m_global_offset + warp_m_offset_i * wmma_m;
                    int n_index = warp_n_global_offset + warp_n_offset_i * wmma_n;
                    wmma::load_matrix_sync(C_frag[warp_m_offset_i][warp_n_offset_i], &C[m_index * n + n_index], n, wmma::mem_row_major);
#endif
                }
            }
#endif

//          Do Matrix multiplication
//            #pragma unroll
            for (int local_k_offset = 0; local_k_offset < shared_k; local_k_offset += wmma_k * warp_tiles_k)
            {
                wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> A_frag[warp_tiles_m][warp_tiles_k];
                wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> B_frag[warp_tiles_k][warp_tiles_n];

                #pragma unroll
                for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
                {
                    #pragma unroll
                    for (int warp_k_offset_i = 0; warp_k_offset_i < warp_tiles_k; warp_k_offset_i++)
                    {
                        wmma::load_matrix_sync(A_frag[warp_m_offset_i][warp_k_offset_i], &A_shared[warp_m_shared_offset + warp_m_offset_i * wmma_m][local_k_offset + warp_k_offset_i * wmma_k], A_shared_k_true);
                    }

                    #pragma unroll
                    for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i++)
                    {
                        #pragma unroll
                        for (int warp_k_offset_i = 0; warp_k_offset_i < warp_tiles_k; warp_k_offset_i++)
                        {
                            wmma::load_matrix_sync(B_frag[warp_k_offset_i][warp_n_offset_i], &B_shared[local_k_offset + warp_k_offset_i * wmma_k][warp_n_shared_offset+ warp_n_offset_i * wmma_n], B_shared_n_true);
                        }

                        #pragma unroll
                        for (int warp_k_offset_i = 0; warp_k_offset_i < warp_tiles_k; warp_k_offset_i++)
                        {
                            wmma::mma_sync(C_frag[warp_m_offset_i][warp_n_offset_i], A_frag[warp_m_offset_i][warp_k_offset_i], B_frag[warp_k_offset_i][warp_n_offset_i], C_frag[warp_m_offset_i][warp_n_offset_i]);
                        }
                    }
                }
            }

#ifndef KEEP_C
//  Update C, freeing registers for copying A and B
            #pragma unroll
            for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
            {
                #pragma unroll
                for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i++)
                {
#ifdef CACHE_C
                    int m_index = warp_m_shared_offset + warp_m_offset_i * wmma_m;
                    int n_index = warp_n_shared_offset + warp_n_offset_i * wmma_n;
                    wmma::store_matrix_sync(&C_shared[m_index][n_index], C_frag[warp_m_offset_i][warp_n_offset_i], C_shared_n_true, wmma::mem_row_major);
#else
                    int m_index = warp_m_global_offset + warp_m_offset_i * wmma_m;
                    int n_index = warp_n_global_offset + warp_n_offset_i * wmma_n;
                    wmma::store_matrix_sync(&C[m_index * n + n_index], C_frag[warp_m_offset_i][warp_n_offset_i], n, wmma::mem_row_major);
#endif
                }
            }
#endif
        }
        __syncthreads();
    }

#ifdef CACHE_C
    #pragma unroll
    for (int i = 0; i < copies_per_thread_C; i++) {
        unsigned int tile_i = threadIdx.x + i * blockDim.x;
        unsigned int tile_m_index = tile_i / shared_n;
        unsigned int tile_n_index = tile_i % shared_n;
        unsigned int C_m_index = block_m_global_offset + tile_m_index;
        unsigned int C_n_index = block_n_global_offset + tile_n_index;

        if (tile_m_index < shared_m && tile_n_index < shared_n) {
            //            TODO: try to avoid ternary statement
            C[C_m_index * n + C_n_index] = C_m_index < m && C_n_index < n ? C_shared[tile_m_index][tile_n_index] : (accType) 0.0f;
        }
    }
#else
#ifdef KEEP_C
    if (warp_m_global_offset < m && warp_n_global_offset < n) {
        #pragma unroll
        for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
        {
            #pragma unroll
            for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i++)
            {
                int m_index = warp_m_global_offset + warp_m_offset_i * wmma_m;
                int n_index = warp_n_global_offset + warp_n_offset_i * wmma_n;
                wmma::store_matrix_sync(&C[m_index * n + n_index], C_frag[warp_m_offset_i][warp_n_offset_i], n, wmma::mem_row_major);
            }
        }
    }
#endif
#endif
}


#endif //CODE_MATMUL_TENSOR_CUH
