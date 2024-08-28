//
// Created by runeebl on 10/23/23.
//

#ifndef CODE_MATMUL_TENSOR_CUH
#define CODE_MATMUL_TENSOR_CUH


//#define KEEP_C
//#define CACHE_C

#define WARP_SIZE 32

#ifndef SHARED_PADDING
#define SHARED_PADDING 8
#endif

#ifndef LOAD_TYPE
#define LOAD_TYPE double2
#endif

#ifndef NUM_STAGES
#define NUM_STAGES 2
#endif


#include <stdint.h>
#include <mma.h>
#include "cuda_fp16.h"

#include <cooperative_groups.h>
#include <cuda/barrier>
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>

using namespace nvcuda;

namespace cg = cooperative_groups;

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

template <class elmType, class accType, unsigned int wmma_m, unsigned int wmma_n, unsigned int wmma_k, unsigned int frags_m, unsigned int frags_n, unsigned int frags_k, unsigned int warp_tiles_m, unsigned int warp_tiles_n, unsigned int block_tiles_m, unsigned int block_tiles_n, unsigned int block_tiles_k, unsigned int threads_per_block, unsigned int num_stages>
__global__ void
#ifdef BLOCKS_PER_SM
__launch_bounds__(THREADS_PER_BLOCK, BLOCKS_PER_SM)
#else
__launch_bounds__(THREADS_PER_BLOCK)
#endif
matMulTiledTensor(elmType* A, elmType* B, accType* C, int m, int n, int k) {
    extern __shared__ char dynamic_shared[];

    constexpr unsigned int shared_m = wmma_m * frags_m * warp_tiles_m * block_tiles_m;
    constexpr unsigned int shared_n = wmma_n * frags_n * warp_tiles_n * block_tiles_n;
    constexpr unsigned int shared_k = wmma_k * frags_k * block_tiles_k;

    constexpr int copies_per_thread_A = (shared_m * shared_k + threads_per_block) / threads_per_block;
    constexpr int copies_per_thread_B = (shared_k * shared_n + threads_per_block) / threads_per_block;
    constexpr int elms_per_load = sizeof(LOAD_TYPE) / sizeof(elmType);

    unsigned int block_m_global_offset = blockIdx.y * shared_m;
    unsigned int block_n_global_offset = blockIdx.x * shared_n;

    unsigned int warpID = threadIdx.x / warpSize;

    // Assumes num_warps >= block_tiles_m * block_tiles_n
    unsigned int warp_m_index = warpID / block_tiles_n;
    unsigned int warp_n_index = warpID % block_tiles_n;

    // TODO: make this last index instead for better memory access?
    unsigned int warp_m_shared_offset = warp_m_index * wmma_m * frags_m * warp_tiles_m;
    unsigned int warp_n_shared_offset = warp_n_index * wmma_n * frags_n * warp_tiles_n;

    unsigned int warp_m_global_offset = block_m_global_offset + warp_m_shared_offset;
    unsigned int warp_n_global_offset = block_n_global_offset + warp_n_shared_offset;

    // Pad to avoid bank conflicts
    constexpr unsigned int A_shared_k_true = shared_k + SHARED_PADDING;
    auto A_shared = reinterpret_cast<elmType *>(dynamic_shared);

    // Pad to avoid bank conflicts
    constexpr unsigned int B_shared_n_true = shared_n + SHARED_PADDING;
    auto B_shared = A_shared + num_stages * shared_m * A_shared_k_true;

    auto zero_elm = LOAD_TYPE();

    cg::thread_block block = cg::this_thread_block();
    // Allocate shared storage for a cuda::pipeline:
    __shared__ cuda::pipeline_shared_state<
            cuda::thread_scope::thread_scope_block,
            num_stages
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    // TODO: load and store C_frags?
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, accType> C_frag[frags_m * warp_tiles_m][frags_n * warp_tiles_n];

    // Initialize C_frag to zero
    #ifdef UNROLL
    #pragma unroll
    #endif
    for (int warp_m_offset_i = 0; warp_m_offset_i < frags_m * warp_tiles_m; warp_m_offset_i++)
    {
        #ifdef UNROLL
        #pragma unroll
        #endif
        for (int warp_n_offset_i = 0; warp_n_offset_i < frags_n * warp_tiles_n; warp_n_offset_i++)
        {
            wmma::fill_fragment(C_frag[warp_m_offset_i][warp_n_offset_i], accType());
        }
    }

    unsigned int k_iterations = (k + shared_k) / shared_k;
    for (int global_k_offset_i = 0; global_k_offset_i < k_iterations + num_stages - 1; global_k_offset_i++) {
        int global_k_offset = global_k_offset_i * shared_k;
        unsigned int load_buffer = global_k_offset_i % num_stages;
        unsigned int compute_buffer = (global_k_offset_i + 1) % num_stages;

        if (global_k_offset_i < k_iterations)
        {
            // Copy A and B to shared memory (Producer Code)
            pipeline.producer_acquire();
            #ifdef UNROLL
            #pragma unroll
            #endif
            for (int i = 0; i < (copies_per_thread_A + elms_per_load) / elms_per_load; i++)
            {
                unsigned int tile_i = threadIdx.x + i * blockDim.x;
                unsigned int tile_m_index = tile_i / (shared_k / elms_per_load);
                unsigned int tile_k_index = tile_i % (shared_k / elms_per_load);
                unsigned int A_m_index = block_m_global_offset + tile_m_index;
                unsigned int A_k_index = (global_k_offset / elms_per_load) + tile_k_index;

                if (tile_m_index < shared_m && tile_k_index < (shared_k / elms_per_load))
                {
                    auto load_dest = &reinterpret_cast<LOAD_TYPE *>(A_shared)[load_buffer * shared_m * (A_shared_k_true / elms_per_load) + tile_m_index * (A_shared_k_true / elms_per_load) + tile_k_index];
                    if (A_m_index < m && A_k_index < k / elms_per_load) {
                        cuda::memcpy_async(load_dest, &reinterpret_cast<LOAD_TYPE *>(A)[A_m_index * (k / elms_per_load) + A_k_index], sizeof(LOAD_TYPE), pipeline);
                    } else {
                        cuda::memcpy_async(load_dest, &zero_elm, sizeof(LOAD_TYPE), pipeline);
                    }
                }
            }

            #ifdef UNROLL
            #pragma unroll
            #endif
            for (int i = 0; i < (copies_per_thread_B + elms_per_load) / elms_per_load; i++)
            {
                unsigned int tile_i = threadIdx.x + i * blockDim.x;
                unsigned int tile_k_index = tile_i / (shared_n / elms_per_load);
                unsigned int tile_n_index = tile_i % (shared_n / elms_per_load);
                unsigned int B_k_index = global_k_offset + tile_k_index;
                unsigned int B_n_index = block_n_global_offset / elms_per_load + tile_n_index;

                if (tile_k_index < shared_k && tile_n_index < shared_n / elms_per_load)
                {
                    auto load_dest = &reinterpret_cast<LOAD_TYPE *>(B_shared)[load_buffer * shared_k * (B_shared_n_true / elms_per_load) + tile_k_index * (B_shared_n_true / elms_per_load) + tile_n_index];
                    if (B_k_index < k && B_n_index < n / elms_per_load) {
                        cuda::memcpy_async(load_dest, &reinterpret_cast<LOAD_TYPE *>(B)[B_k_index * (n / elms_per_load) + B_n_index], sizeof(LOAD_TYPE), pipeline);
                    } else {
                        cuda::memcpy_async(load_dest, &zero_elm, sizeof(LOAD_TYPE), pipeline);
                    }
                }
            }
            pipeline.producer_commit();
        }

        if (global_k_offset_i >= num_stages - 1) {
            // Do Matrix multiplication (Consumer Code)
            if (warp_m_global_offset < m && warp_n_global_offset < n)
            {
                pipeline.consumer_wait();

                wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> A_frag[frags_m][frags_k];
                wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> B_frag[frags_k][frags_n];

                #ifdef UNROLL
                #pragma unroll
                #endif
                #ifdef NOUNROLL
                #pragma unroll 1
                #endif
                for (int local_k_offset_i = 0; local_k_offset_i < block_tiles_k; local_k_offset_i++)
                {
                    int local_k_offset = local_k_offset_i * frags_k * wmma_k;

                    #ifdef UNROLL
                    #pragma unroll
                    #endif
                    #ifdef NOUNROLL1
                    #pragma unroll 1
                    #endif
                    for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
                    {
                        int warp_m_offset = warp_m_offset_i * frags_m * wmma_m;

                        #ifdef UNROLL
                        #pragma unroll
                        #endif
                        for (int frag_k_offset_i = 0; frag_k_offset_i < frags_k; frag_k_offset_i++)
                        {
                            #ifdef UNROLL
                            #pragma unroll
                            #endif
                            for (int frag_m_offset_i = 0; frag_m_offset_i < frags_m; frag_m_offset_i++)
                            {
                                wmma::load_matrix_sync(A_frag[frag_m_offset_i][frag_k_offset_i],
                                                       &A_shared[compute_buffer * shared_m * A_shared_k_true +
                                                                 (warp_m_shared_offset + warp_m_offset + frag_m_offset_i * wmma_m) *
                                                                 A_shared_k_true + local_k_offset +
                                                                 frag_k_offset_i * wmma_k], A_shared_k_true);
                            }
                        }

                        #ifdef UNROLL
                        #pragma unroll
                        #endif
                        #ifdef NOUNROLL2
                        #pragma unroll 1
                        #endif
                        for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i++)
                        {
                            int warp_n_offset = warp_n_offset_i * frags_n * wmma_n;

                            #ifdef UNROLL
                            #pragma unroll
                            #endif
                            for (int frag_k_offset_i = 0; frag_k_offset_i < frags_k; frag_k_offset_i++)
                            {
                                #ifdef UNROLL
                                #pragma unroll
                                #endif
                                for (int frag_n_offset_i = 0; frag_n_offset_i < frags_n; frag_n_offset_i++)
                                {
                                    wmma::load_matrix_sync(B_frag[frag_k_offset_i][frag_n_offset_i],
                                                           &B_shared[compute_buffer * shared_k * B_shared_n_true +
                                                                     (local_k_offset + frag_k_offset_i * wmma_k) *
                                                                     B_shared_n_true + warp_n_shared_offset + warp_n_offset +
                                                                     frag_n_offset_i * wmma_n], B_shared_n_true);
                                }
                            }

                            #ifdef UNROLL
                            #pragma unroll
                            #endif
                            for (int frag_k_offset_i = 0; frag_k_offset_i < frags_k; frag_k_offset_i++)
                            {
                                #ifdef UNROLL
                                #pragma unroll
                                #endif
                                for (int frag_m_offset_i = 0; frag_m_offset_i < frags_m; frag_m_offset_i++)
                                {
                                    #ifdef UNROLL
                                    #pragma unroll
                                    #endif
                                    for (int frag_n_offset_i = 0; frag_n_offset_i < frags_n; frag_n_offset_i++)
                                    {
                                        #ifdef SERPENTINE
                                        // Serpentine iteration to increase temporal locality and reduce register usage
                                        int frag_n_offset_i_serpentine = (frag_m_offset_i % 2) ? (frags_n - 1 - frag_n_offset_i) : frag_n_offset_i;
                                        #else
                                        // Serpentine off
                                        int frag_n_offset_i_serpentine = frag_n_offset_i;
                                        #endif
                                        wmma::mma_sync(C_frag[warp_m_offset_i * frags_m + frag_m_offset_i][warp_n_offset_i * frags_n + frag_n_offset_i_serpentine],
                                                       A_frag[frag_m_offset_i][frag_k_offset_i], B_frag[frag_k_offset_i][frag_n_offset_i_serpentine],
                                                       C_frag[warp_m_offset_i * frags_m + frag_m_offset_i][warp_n_offset_i * frags_n + frag_n_offset_i_serpentine]);
                                    }
                                }
                            }
                        }
                    }
                }
                pipeline.consumer_release();
            }
        }
    }

    // TODO: try storing in shared first, then optimized store to global
    if (warp_m_global_offset < m && warp_n_global_offset < n) {
        #ifdef UNROLL
        #pragma unroll
        #endif
        for (int warp_m_offset_i = 0; warp_m_offset_i < warp_tiles_m; warp_m_offset_i++)
        {
            int warp_m_offset = warp_m_offset_i * frags_m * wmma_m;

            #ifdef UNROLL
            #pragma unroll
            #endif
            for (int warp_n_offset_i = 0; warp_n_offset_i < warp_tiles_n; warp_n_offset_i++)
            {
                int warp_n_offset = warp_n_offset_i * frags_n * wmma_n;

                #ifdef UNROLL
                #pragma unroll
                #endif
                for (int frag_m_offset_i = 0; frag_m_offset_i < frags_m; frag_m_offset_i++)
                {
                    #ifdef UNROLL
                    #pragma unroll
                    #endif
                    for (int frag_n_offset_i = 0; frag_n_offset_i < frags_n; frag_n_offset_i++)
                    {
                        int m_index = warp_m_global_offset + warp_m_offset + frag_m_offset_i * wmma_m;
                        int n_index = warp_n_global_offset + warp_n_offset + frag_n_offset_i * wmma_n;
                        wmma::store_matrix_sync(&C[m_index * n + n_index], C_frag[warp_m_offset_i * frags_m + frag_m_offset_i][warp_n_offset_i * frags_n + frag_n_offset_i], n,
                                                wmma::mem_row_major);
                    }
                }
            }
        }
    }
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
        if (A_row < m && A_col < k && B_row < k && B_col < n) {
            wmma::load_matrix_sync(A_frag, &A[A_row * k + A_col], k);
            wmma::load_matrix_sync(B_frag, &B[B_row * n + B_col], n);
            wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
        }

    }
    int C_row = warp_m * wmma_m;
    int C_col = warp_n * wmma_n;
    
    if (C_row < m && C_col < n) {
        wmma::store_matrix_sync(&C[C_row * n + C_col], C_frag, n, wmma::mem_row_major);
    }    
}


#endif //CODE_MATMUL_TENSOR_CUH
