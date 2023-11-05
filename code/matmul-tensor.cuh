//
// Created by runeebl on 10/23/23.
//

#ifndef CODE_MATMUL_TENSOR_CUH
#define CODE_MATMUL_TENSOR_CUH


//#define KEEP_C
//#define CACHE_C

#define WARP_SIZE 32
#define SHARED_PADDING 8


#include <stdint.h>
#include <mma.h>
#include "matmul-tensor.cuh"
#include "cuda_fp16.h"

using namespace nvcuda;

// TODO: check reads coalesced, check store is coalesced


#ifndef THREADS_PER_BLOCK
#ifdef BLOCK_TILES_M
#ifdef BLOCK_TILES_N
#define THREADS_PER_BLOCK BLOCK_TILES_M * BLOCK_TILES_N * WARP_SIZE
#else
#define THREADS_PER_BLOCK 0
#endif
#else
#define THREADS_PER_BLOCK 0
#endif
#endif


// TODO: set maxBlocksPerCluster?

template <class elmType, class accType, int wmma_m, int wmma_n, int wmma_k, int warp_tiles_m, int warp_tiles_n, int warp_tiles_k, int block_tiles_m, int block_tiles_n, int block_tiles_k, int threads_per_block>
__global__ void
#ifdef BLOCKS_PER_SM
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_SM)
#else
__launch_bounds__(THREADS_PER_BLOCK)
#endif
matMulTiledTensor(elmType* A, elmType* B, accType* C, int m, int n, int k) {
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

    unsigned int warp_group = warpID % 2;

//    Assumes num_warps >= block_tiles_m * block_tiles_n
    unsigned int warp_m_index = warpID / block_tiles_n;
    unsigned int warp_n_index = warpID % block_tiles_n;

    unsigned int warp_m_shared_offset = warp_m_index * wmma_m * warp_tiles_m;
    unsigned int warp_n_shared_offset = warp_n_index * wmma_n * warp_tiles_n;

    unsigned int warp_m_global_offset = block_m_global_offset + warp_m_shared_offset;
    unsigned int warp_n_global_offset = block_n_global_offset + warp_n_shared_offset;

    //    Pad to avoid bank conflicts
    constexpr unsigned int A_shared_k_true = shared_k + SHARED_PADDING;
    __shared__ elmType A_shared[2][shared_m][A_shared_k_true];

    //    Pad to avoid bank conflicts
    constexpr unsigned int B_shared_n_true = shared_n + SHARED_PADDING;
    __shared__ elmType B_shared[2][shared_k][B_shared_n_true];

#ifdef CACHE_C
    constexpr int copies_per_thread_C = (shared_m * shared_n + threads_per_block) / threads_per_block;

    //    Pad to avoid bank conflicts
    constexpr unsigned int C_shared_n_true = shared_n + SHARED_PADDING;
    __shared__ accType C_shared[shared_m][C_shared_n_true];

    //    Initialize C cache to zero
    #ifdef UNROLL
    #pragma unroll
    #endif
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
    #ifdef UNROLL
    #pragma unroll
    #endif
    for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
    {
        #ifdef UNROLL
        #pragma unroll
        #endif
        for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i++)
        {
            wmma::fill_fragment(C_frag[warp_m_offset_i][warp_n_offset_i], (accType) 0.0f);
        }
    }
#endif
#endif

//    for (int global_k_offset = 0; global_k_offset < k; global_k_offset += shared_k) {

    unsigned int k_iterations = (k + shared_k) / shared_k;
    for (int global_k_offset_i = 0; global_k_offset_i < k_iterations + 1; global_k_offset_i++) {
        int global_k_offset = global_k_offset_i * shared_k;

#ifdef UNROLL
#pragma unroll
#endif
        for (int group_round = 0; group_round < 2; group_round++)
        {
            if (group_round % 2 == warp_group && global_k_offset_i != k_iterations)
            {
                //      Copy A and B to shared memory
#ifdef UNROLL
#pragma unroll
#endif
                for (int i = 0; i < copies_per_thread_A; i++)
                {
                    unsigned int tile_i = threadIdx.x + i * blockDim.x;
                    unsigned int tile_m_index = tile_i / shared_k;
                    unsigned int tile_k_index = tile_i % shared_k;
                    unsigned int A_m_index = block_m_global_offset + tile_m_index;
                    unsigned int A_k_index = global_k_offset + tile_k_index;

                    //            TODO: try to avoid ternary statement
                    if (tile_m_index < shared_m && tile_k_index < shared_k)
                    {
                        A_shared[global_k_offset_i % 2][tile_m_index][tile_k_index] =
                                A_m_index < m && A_k_index < k ? A[A_m_index * k + A_k_index] : (elmType) 0.0f;
                    }
                }

#ifdef UNROLL
#pragma unroll
#endif
                for (int i = 0; i < copies_per_thread_B; i++)
                {
                    unsigned int tile_i = threadIdx.x + i * blockDim.x;
                    unsigned int tile_k_index = tile_i / shared_n;
                    unsigned int tile_n_index = tile_i % shared_n;
                    unsigned int B_k_index = global_k_offset + tile_k_index;
                    unsigned int B_n_index = block_n_global_offset + tile_n_index;

                    if (tile_k_index < shared_k && tile_n_index < shared_n)
                    {
                        //            TODO: try to avoid ternary statement
                        B_shared[global_k_offset_i % 2][tile_k_index][tile_n_index] =
                                B_k_index < k && B_n_index < n ? B[B_k_index * n + B_n_index] : (elmType) 0.0f;
                    }
                }
            }

            //        TODO: not needed anymore?
            //        __syncthreads();
            //      End of copy to shared memory

            else if (global_k_offset_i != 0)
            {
                //        TODO: move check into loop?
                if (warp_m_global_offset < m && warp_n_global_offset < n)
                {
#ifndef KEEP_C
                    //  Load C from memory
                                wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, accType> C_frag[warp_tiles_m][warp_tiles_n];

                    //          Assumes C is initialized to zero
#ifdef UNROLL
#pragma unroll
#endif
                                for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
                                {
#ifdef UNROLL
#pragma unroll
#endif
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
#ifdef UNROLL
#pragma unroll
#endif
#ifdef NOUNROLL
#pragma unroll 1
#endif
                    for (int local_k_offset_i = 0; local_k_offset_i < block_tiles_k; local_k_offset_i++)
                    {
                        int local_k_offset = local_k_offset_i * wmma_k * warp_tiles_k;

                        //                TODO: why not copy to shared here?

                        wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> A_frag[warp_tiles_m];
                        wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> B_frag[warp_tiles_n];

#ifdef UNROLL
#pragma unroll
#endif
#ifdef NOUNROLL
#pragma unroll 1
#endif
                        for (int warp_k_offset_i = 0; warp_k_offset_i < warp_tiles_k; warp_k_offset_i++)
                        {
#ifdef UNROLL
#pragma unroll
#endif
                            for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
                            {
                                wmma::load_matrix_sync(A_frag[warp_m_offset_i],
                                                       &A_shared[(global_k_offset_i - 1) % 2][warp_m_shared_offset +
                                                                                              warp_m_offset_i * wmma_m][
                                                               local_k_offset + warp_k_offset_i * wmma_k],
                                                       A_shared_k_true);

#ifdef UNROLL
#pragma unroll
#endif
                                for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i++)
                                {
                                    //                            Serpentine iteration to increase temporal locality and reduce register usage
                                    int warp_n_offset_i_serpentine = (warp_m_offset_i % 2) ? (warp_tiles_n - 1 -
                                                                                              warp_n_offset_i)
                                                                                           : warp_n_offset_i;

                                    wmma::load_matrix_sync(B_frag[warp_n_offset_i_serpentine],
                                                           &B_shared[(global_k_offset_i - 1) % 2][local_k_offset +
                                                                                                  warp_k_offset_i *
                                                                                                  wmma_k][
                                                                   warp_n_shared_offset +
                                                                   warp_n_offset_i_serpentine * wmma_n],
                                                           B_shared_n_true);

                                    wmma::mma_sync(C_frag[warp_m_offset_i][warp_n_offset_i_serpentine],
                                                   A_frag[warp_m_offset_i],
                                                   B_frag[warp_n_offset_i_serpentine],
                                                   C_frag[warp_m_offset_i][warp_n_offset_i_serpentine]);
                                }
                            }
                        }
                    }
                }

#ifndef KEEP_C
                //  Update C, freeing registers for copying A and B
#ifdef UNROLL
#pragma unroll
#endif
                            for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
                            {
#ifdef UNROLL
#pragma unroll
#endif
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
        }
        __syncthreads();
    }

#ifdef CACHE_C
    #ifdef UNROLL
    #pragma unroll
    #endif
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
        #ifdef UNROLL
        #pragma unroll
        #endif
        for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
        {
            #ifdef UNROLL
            #pragma unroll
            #endif
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

template <class accType, class elmType, int wmma_m, int wmma_n, int wmma_k, int block_tiles_m, int block_tiles_n, int block_tiles_k>
__global__ void matMulTiledTensorNaive(elmType* A, elmType* B, accType* C, unsigned m, unsigned n, unsigned k) {
    wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, accType> C_frag;

    // Taken from
    // https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
    unsigned warp_n = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    unsigned warp_m = (blockIdx.y * blockDim.y + threadIdx.y);
    
    wmma::fill_fragment(C_frag, (accType)0.0f);
    // Sequentialize the k dimension
    for (int i = 0; i < k; i += wmma_k) {        
        // Recall that we have block_tiles_m warps in the m dimension.
        // These will be wmma_m rows spaced appart. Now we find the row for each warp.
        int A_row = warp_m * wmma_m;        
        int A_col = i; // because A (M x K) and we sequantialize the k dimension
        int B_row = i; // again we B is (K x N) and we sequentialize the k dimension
        // Again we spawn block_tiles_n warps for the block in the n dimension.
        // This finds the starting column for all warps
        int B_col = warp_n * wmma_n;
        // printf("A_row: %d, B_col: %d\n", A_row, B_col);
        if (A_row < m && A_col < k && B_row < k && B_col < n) {
            // printf("A_row: %d, B_col: %d, i: %d\n", A_col, B_col, i);
            // Watch out for what leading dimension means
            wmma::load_matrix_sync(A_frag, &A[A_row * k + A_col], k);
            wmma::load_matrix_sync(B_frag, &B[B_row * n + B_col], n);
            wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        }

    }
    int C_row = warp_m * wmma_m;
    int C_col = warp_n * wmma_n;
    
    if (C_row < m && C_col < n) {
        // printf("C_row: %d, C_col: %d\n", C_row, C_col);
        wmma::store_matrix_sync(&C[C_row * n + C_col], C_frag, n, wmma::mem_row_major);
    }    
}


#endif //CODE_MATMUL_TENSOR_CUH
