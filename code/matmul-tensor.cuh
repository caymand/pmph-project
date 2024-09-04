//
// Created by runeebl on 10/23/23.
//

#ifndef CODE_MATMUL_TENSOR_CUH
#define CODE_MATMUL_TENSOR_CUH


//#define KEEP_C
//#define CACHE_C

#define WARP_SIZE 32


#ifndef LOAD_TYPE
#define LOAD_TYPE float4
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

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))


// TODO: avoid reinterpret_cast<uint32_t *> for below functions

// TODO: account for different elm and acc types
// TODO: check types

__forceinline__ __device__ void ldmatrix_x2(uint32_t r[2], void * p) {
    auto smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_ptr));
}

__forceinline__ __device__ void ldmatrix_x2_trans(uint32_t r[2], void * p) {
    auto smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_ptr));
}

__forceinline__ __device__ void ldmatrix_x4(uint32_t r[4], void * p) {
    auto smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_ptr));
}

__forceinline__ __device__ void ldmatrix_x4_trans(uint32_t r[4], void * p) {
    auto smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_ptr));
}

__forceinline__ __device__ void mma_m16n8k16(uint32_t d[4], uint32_t a[4], uint32_t b[2], uint32_t c[4]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__forceinline__ __device__ void cp_async(void * dst, void * src) {
    auto dst_p = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
//    auto src_p = static_cast<uint32_t>(__cvta_generic_to_global(src));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :  : "r"(dst_p), "l"(src));
}

__forceinline__ __device__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" :  : );
}

template <int N>
__forceinline__ __device__ void cp_async_wait() {
//    TODO: use this:
    asm volatile("cp.async.wait_group %0;\n" :  : "n"(N));
//    asm volatile("cp.async.wait_all;\n" :  : );
}


// TODO: use something like this maybe 2D b and c, else just double dimensions?
//__forceinline__ __device__ void mma_m16n16k16(uint32_t d[4], uint32_t a[4], uint32_t b[2], uint32_t c[4]) {
//    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
//    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
//}





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

template <class elmType, class accType, unsigned int wmma_m, unsigned int wmma_n, unsigned int wmma_k, unsigned int frags_m, unsigned int frags_n, unsigned int frags_k, unsigned int warp_tiles_m, unsigned int warp_tiles_n, unsigned int warp_tiles_k, unsigned int block_tiles_m, unsigned int block_tiles_n, unsigned int threads_per_block, unsigned int num_stages>
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
    constexpr unsigned int shared_k = wmma_k * frags_k * warp_tiles_k;

//    TODO: handle other cases, account for differennt element sizes
    // Assumes 64 halfs = 128B in leading dimension of A and B
    assert(shared_k % 64 == 0 && shared_n % 64 == 0);

    constexpr int copies_per_thread_A = DIV_UP(shared_m * shared_k, threads_per_block);
    constexpr int copies_per_thread_B = DIV_UP(shared_k * shared_n, threads_per_block);
    constexpr int elms_per_load = DIV_UP(sizeof(LOAD_TYPE), sizeof(elmType));

    unsigned int warpID = threadIdx.x / warpSize;
    unsigned int laneID = threadIdx.x % warpSize;

    unsigned int warpQuarter = laneID / 8;
    unsigned int warpIDInQuarter = laneID % 8;

    // Assumes num_warps >= block_tiles_m * block_tiles_n
    unsigned int warp_m_index = warpID / block_tiles_n;
    unsigned int warp_n_index = warpID % block_tiles_n;

    unsigned int block_m_global_offset = blockIdx.y * shared_m;
    unsigned int block_n_global_offset = blockIdx.x * shared_n;

    // TODO: make this last index instead for better memory access?
    unsigned int warp_m_shared_offset = warp_m_index * wmma_m * frags_m * warp_tiles_m;
    unsigned int warp_n_shared_offset = warp_n_index * wmma_n * frags_n * warp_tiles_n;

    unsigned int warp_m_global_offset = block_m_global_offset + warp_m_shared_offset;
    unsigned int warp_n_global_offset = block_n_global_offset + warp_n_shared_offset;

    auto A_shared = reinterpret_cast<elmType *>(dynamic_shared);
    auto B_shared = A_shared + num_stages * shared_m * shared_k;

    constexpr unsigned int load_tile_width = 8;

    constexpr unsigned int load_tile_width_elms = load_tile_width * elms_per_load;
    constexpr unsigned int load_tile_height = 8;

    constexpr unsigned int A_load_tiles_m = DIV_UP(shared_m, load_tile_height);
    constexpr unsigned int A_load_tiles_k = DIV_UP(shared_k, load_tile_width_elms);

    constexpr unsigned int B_load_tiles_k = DIV_UP(shared_k, load_tile_height);
    constexpr unsigned int B_load_tiles_n = DIV_UP(shared_n, load_tile_width_elms);


    auto zero_elm = LOAD_TYPE();

//    cg::thread_block block = cg::this_thread_block();
//    // Allocate shared storage for a cuda::pipeline:
//    __shared__ cuda::pipeline_shared_state<
//            cuda::thread_scope::thread_scope_block,
//            num_stages
//    > shared_state;
//    auto pipeline = cuda::make_pipeline(block, &shared_state);


    // TODO: account for different elm and acc types
    // Using 2 x 16x8x16 as basic building block
    float C_frag[frags_m * warp_tiles_m][frags_n * warp_tiles_n][2][4];

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
            #ifdef UNROLL
            #pragma unroll
            #endif
            for (int j = 0; j < 2; j++)
            {
                #ifdef UNROLL
                #pragma unroll
                #endif
                for (int i = 0; i < 4; i++)
                {
                    C_frag[warp_m_offset_i][warp_n_offset_i][j][i] = float();
                }
            }
        }
    }

    unsigned int k_iterations = DIV_UP(k,shared_k);
    for (int global_k_offset_i = 0; global_k_offset_i < k_iterations + num_stages - 1; global_k_offset_i++) {
        int global_k_offset = global_k_offset_i * shared_k;
        unsigned int load_buffer = global_k_offset_i % num_stages;
        unsigned int compute_buffer = (global_k_offset_i + 1) % num_stages;

        if (global_k_offset_i < k_iterations)
        {
            // Copy A and B to shared memory (Producer Code)
//            pipeline.producer_acquire();

            #ifdef UNROLL
            #pragma unroll
            #endif
// TODO: remove
//#pragma unroll 1
            for (int i = 0; i < DIV_UP(copies_per_thread_A, elms_per_load); i++)
            {
                unsigned int load_i = threadIdx.x + i * blockDim.x;

                unsigned int load_tile_i = load_i / (load_tile_width * load_tile_height);
                unsigned int load_i_in_tile = load_i % (load_tile_width * load_tile_height);

                unsigned int load_tile_k_i = load_tile_i % A_load_tiles_k;
                unsigned int load_tile_m_i = load_tile_i / A_load_tiles_k;

                unsigned int load_tile_k_shared_offset = load_tile_k_i * load_tile_width_elms;
                unsigned int load_tile_m_shared_offset = load_tile_m_i * load_tile_height;

                unsigned int load_k_i = load_i_in_tile % load_tile_width;
                unsigned int load_m_i = load_i_in_tile / load_tile_width;

                unsigned int load_k_swizzled_i = load_k_i ^ load_m_i;
                unsigned int load_m_swizzled_i = load_k_i;

                // Each load is of size 1 x elms_per_load
                unsigned int load_k_shared_index = load_tile_k_shared_offset + load_k_i * elms_per_load;
                unsigned int load_m_shared_index = load_tile_m_shared_offset + load_m_i;

                // Each load is of size 1 x elms_per_load
                unsigned int load_k_swizzled_index = load_k_swizzled_i * elms_per_load;
                unsigned int load_m_swizzled_index = load_m_swizzled_i;

                unsigned int A_m_index = block_m_global_offset + load_m_shared_index;
                unsigned int A_k_index = global_k_offset + load_k_shared_index;


                if (load_m_shared_index < shared_m)
                {
                    auto load_dest = &A_shared[load_buffer * shared_m * shared_k
                                               + load_tile_m_i * A_load_tiles_k * load_tile_width_elms * load_tile_height
                                               + load_tile_k_i * load_tile_width_elms * load_tile_height
                                               + load_m_swizzled_index * load_tile_width_elms
                                               + load_k_swizzled_index];
                    if (A_m_index < m && A_k_index < k) {
//                        cuda::memcpy_async(reinterpret_cast<LOAD_TYPE *>(load_dest), reinterpret_cast<LOAD_TYPE *>(&A[A_m_index * k + A_k_index]), sizeof(LOAD_TYPE), pipeline);
                        cp_async(reinterpret_cast<LOAD_TYPE *>(load_dest), reinterpret_cast<LOAD_TYPE *>(&A[A_m_index * k + A_k_index]));
                    } else {
//                        cuda::memcpy_async(reinterpret_cast<LOAD_TYPE *>(load_dest), &zero_elm, sizeof(LOAD_TYPE), pipeline);
                        cp_async(reinterpret_cast<LOAD_TYPE *>(load_dest), &zero_elm);
                    }
                }
            }

            #ifdef UNROLL
            #pragma unroll
            #endif
// TODO: remove
//#pragma unroll 1
            for (int i = 0; i < DIV_UP(copies_per_thread_B, elms_per_load); i++)
            {
                unsigned int load_i = threadIdx.x + i * blockDim.x;

                unsigned int load_tile_i = load_i / (load_tile_width * load_tile_height);
                unsigned int load_i_in_tile = load_i % (load_tile_width * load_tile_height);

                unsigned int load_tile_n_i = load_tile_i % B_load_tiles_n;
                unsigned int load_tile_k_i = load_tile_i / B_load_tiles_n;

                unsigned int load_tile_n_shared_offset = load_tile_n_i * load_tile_width_elms;
                unsigned int load_tile_k_shared_offset = load_tile_k_i * load_tile_height;

                unsigned int load_n_i = load_i_in_tile % load_tile_width;
                unsigned int load_k_i = load_i_in_tile / load_tile_width;

                unsigned int load_n_swizzled_i = load_n_i ^ load_k_i;
                unsigned int load_k_swizzled_i = load_n_i;

                // Each load is of size 1 x elms_per_load
                unsigned int load_n_shared_index = load_tile_n_shared_offset + load_n_i * elms_per_load;
                unsigned int load_k_shared_index = load_tile_k_shared_offset + load_k_i;

                // Each load is of size 1 x elms_per_load
                unsigned int load_n_swizzled_index = load_n_swizzled_i * elms_per_load;
                unsigned int load_k_swizzled_index = load_k_swizzled_i;

                unsigned int B_k_index = global_k_offset + load_k_shared_index;
                unsigned int B_n_index = block_n_global_offset + load_n_shared_index;


                if (load_k_shared_index < shared_k)
                {
                    auto load_dest = &B_shared[load_buffer * shared_k * shared_n
                                               + load_tile_k_i * B_load_tiles_n * load_tile_width_elms * load_tile_height
                                               + load_tile_n_i * load_tile_width_elms * load_tile_height
                                               + load_k_swizzled_index * load_tile_width_elms
                                               + load_n_swizzled_index];
                    if (B_k_index < k && B_n_index < n) {
//                        cuda::memcpy_async(reinterpret_cast<LOAD_TYPE *>(load_dest), reinterpret_cast<LOAD_TYPE *>(&B[B_k_index * n + B_n_index]), sizeof(LOAD_TYPE), pipeline);
                        cp_async(reinterpret_cast<LOAD_TYPE *>(load_dest), reinterpret_cast<LOAD_TYPE *>(&B[B_k_index * n + B_n_index]));
                    } else {
//                        cuda::memcpy_async(reinterpret_cast<LOAD_TYPE *>(load_dest), &zero_elm, sizeof(LOAD_TYPE), pipeline);
                        cp_async(reinterpret_cast<LOAD_TYPE *>(load_dest), &zero_elm);
                    }
                }
            }
//            pipeline.producer_commit();
            cp_async_commit();
        }

        cp_async_wait<num_stages - 1>();
        __syncthreads();

        if (global_k_offset_i >= num_stages - 1) {
            // Do Matrix multiplication (Consumer Code)
            if (warp_m_global_offset < m && warp_n_global_offset < n)
            {
//                pipeline.consumer_wait();

                half2 A_frag[frags_m][frags_k][4];
                half2 B_frag[frags_k][frags_n][2][2];

                #ifdef NOUNROLL
                #pragma unroll 1
                #else
                #ifdef UNROLL
                #pragma unroll
                #endif
                #endif
                for (int local_k_offset_i = 0; local_k_offset_i < warp_tiles_k; local_k_offset_i++)
                {
                    int local_k_offset = local_k_offset_i * frags_k * wmma_k;

                    #ifdef NOUNROLL1
                    #pragma unroll 1
                    #else
                    #ifdef UNROLL
                    #pragma unroll
                    #endif
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
                                unsigned int matrix_m_shared_index = warp_m_shared_offset + warp_m_offset + frag_m_offset_i * wmma_m;
                                unsigned int matrix_k_shared_index = local_k_offset + frag_k_offset_i * wmma_k;

                                unsigned int load_tile_m_i = matrix_m_shared_index / load_tile_height + (warpQuarter & 1);
                                unsigned int load_tile_k_i = matrix_k_shared_index / load_tile_width_elms;

                                unsigned int load_row = (matrix_k_shared_index % load_tile_width_elms) / elms_per_load + (warpQuarter / 2);
                                unsigned int load_col = warpIDInQuarter ^ load_row;

                                unsigned int load_index = load_tile_m_i * A_load_tiles_k * load_tile_width_elms * load_tile_height + load_tile_k_i * load_tile_width_elms * load_tile_height + load_row * load_tile_width_elms + load_col * elms_per_load;

                                ldmatrix_x4(reinterpret_cast<uint32_t *>(A_frag[frag_m_offset_i][frag_k_offset_i]), &A_shared[compute_buffer * shared_m * shared_k + load_index]);
                            }
                        }

                        #ifdef NOUNROLL1
                        #pragma unroll 1
                        #else
                        #ifdef UNROLL
                        #pragma unroll
                        #endif
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
                                    unsigned int matrix_k_shared_index = local_k_offset + frag_k_offset_i * wmma_k;
                                    unsigned int matrix_n_shared_index = warp_n_shared_offset + warp_n_offset + frag_n_offset_i * wmma_n;

                                    unsigned int load_tile_k_i = matrix_k_shared_index / load_tile_height + (warpQuarter & 1);
                                    unsigned int load_tile_n_i = matrix_n_shared_index / load_tile_width_elms;

                                    unsigned int load_row = (matrix_n_shared_index % load_tile_width_elms) / elms_per_load + (warpQuarter / 2);
                                    unsigned int load_col = warpIDInQuarter ^ load_row;

                                    unsigned int load_index = load_tile_k_i * B_load_tiles_n * load_tile_width_elms * load_tile_height + load_tile_n_i * load_tile_width_elms * load_tile_height + load_row * load_tile_width_elms + load_col * elms_per_load;

                                    ldmatrix_x4_trans(reinterpret_cast<uint32_t *>(B_frag[frag_k_offset_i][frag_n_offset_i]), &B_shared[compute_buffer * shared_k * shared_n + load_index]);
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

                                        #ifdef UNROLL
                                        #pragma unroll
                                        #endif
                                        for (int i = 0; i < 2; i++) {
                                            mma_m16n8k16(reinterpret_cast<uint32_t *>(C_frag[warp_m_offset_i * frags_m + frag_m_offset_i][warp_n_offset_i * frags_n + frag_n_offset_i_serpentine][i]),
                                                         reinterpret_cast<uint32_t *>(A_frag[frag_m_offset_i][frag_k_offset_i]),
                                                         reinterpret_cast<uint32_t *>(B_frag[frag_k_offset_i][frag_n_offset_i_serpentine][i]),
                                                         reinterpret_cast<uint32_t *>(C_frag[warp_m_offset_i * frags_m + frag_m_offset_i][warp_n_offset_i * frags_n + frag_n_offset_i_serpentine][i]));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
//                pipeline.consumer_release();
//                __syncthreads();
            }
        }
    }

//    __syncthreads();

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
                        unsigned int m_offset = warp_m_global_offset + warp_m_offset + frag_m_offset_i * wmma_m;
                        unsigned int n_offset = warp_n_global_offset + warp_n_offset + frag_n_offset_i * wmma_n;

                        // TODO: vectorize stores, try storing in shared first, then coalesced store to global
                        // TODO: refactor, rename
                        #ifdef UNROLL
                        #pragma unroll
                        #endif
                        for (unsigned int j = 0; j < 2; j++)
                        {
                            #ifdef UNROLL
                            #pragma unroll
                            #endif
                            for (unsigned int i = 0; i < 4; i++)
                            {
                                unsigned int groupID = laneID / 4;
                                unsigned int threadID_in_group = laneID % 4;

                                unsigned int row = groupID + 8 * (i / 2);
                                unsigned int col = threadID_in_group * 2 + (i & 1);

                                unsigned int m_index = m_offset + row;
                                unsigned int n_index = n_offset + col + j * 8;

                                if (m_index < m && n_index < n)
                                {
                                    C[m_index * n + n_index] = C_frag[warp_m_offset_i * frags_m + frag_m_offset_i][warp_n_offset_i * frags_n + frag_n_offset_i][j][i];
                                }
                            }
                        }
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
