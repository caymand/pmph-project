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
#include "cuda_fp16.h"

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

    constexpr unsigned int A_loc_m = block_tiles_m * wmma_m;
    constexpr unsigned int A_loc_k = wmma_k * block_tiles_k;
    //  TODO: try with(out) +1
    constexpr unsigned int A_loc_k_true = A_loc_k; // + 1;
    __shared__ elmType A_loc[A_loc_m][A_loc_k_true];

  // remapping (a slice of) B to shared memory
    constexpr unsigned int B_loc_k = block_tiles_k * wmma_k;
    constexpr unsigned int B_loc_n = wmma_n * block_tiles_n;
    //  TODO: try with(out) +1
    constexpr unsigned int B_loc_n_true = B_loc_n; // + 1;
    __shared__ elmType B_loc[B_loc_k][B_loc_n_true];


  // the thread result is computed in register memory
  // and the global-memory array C is updated at the end.
    wmma::fragment<wmma::accumulator, wmma_m, wmma_n, wmma_k, accType> C_frag;

    unsigned int block_start_m = blockIdx.y * block_tiles_m * wmma_m;
    unsigned int block_start_n = blockIdx.x * block_tiles_n * wmma_n;

//    TODO: bitshift?
//    TODO: this is wrong
    unsigned int warpID = threadIdx.x / warpSize;
    unsigned int laneID = threadIdx.x % warpSize;

//    Assumes num_warps >= block_tiles_m * block_tiles_n
    unsigned int warpID_m = warpID / block_tiles_n;
    unsigned int warpID_n = warpID % block_tiles_n;

    //    TODO: check this
    unsigned int ind_m = block_start_m + warpID_m * wmma_m;
    unsigned int ind_n = block_start_n + warpID_n * wmma_n;

//    if (laneID == 0) {
//        printf("warpID: %d, warpID_m: %d, warpID_n: %d\n", warpID, warpID_m, warpID_n);
//
//        printf("warp_id_m: %d, warp_id_n: %d, ind_m: %d, ind_n: %d\n", warpID_m, warpID_n, ind_m, ind_n);
//    }


    // initialize the result with zero
    // (the neutral element for addition)
    wmma::fill_fragment(C_frag, 0.0f);


//    print A:
//    if (threadIdx.x == 0) {
//        printf("A:\n");
//        for (int i = 0; i < m; i++) {
//            for (int j = 0; j < k; j++) {
//                printf("%f ", (float)A[i * k + j]);
//            }
//            printf("\n");
//        }
//    }

//    if (threadIdx.x == 0) {
//        printf("B:\n");
//        for (int i = 0; i < k; i++) {
//            for (int j = 0; j < n; j++) {
//                printf("%f ", (float)B[i * k + j]);
//            }
//            printf("\n");
//        }
//    }

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
//            TODO: check padding?
//            printf("block_start_m: %d, tile_m: %d, A_m: %d\n", block_start_m, tile_m, A_m);
//            printf("global_offset_k: %d, tile_k: %d, A_k: %d\n", global_offset_k, tile_k, A_k);

//            if (!(A_m < m && A_k < k)) {
//                printf("A[%d][%d] = %f\n", A_m, A_k, (float)(A[A_m * k + A_k]));
//            }
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
            //            TODO: check padding?

            if (tile_k < B_loc_k && tile_n < B_loc_n) {
                B_loc[tile_k][tile_n] = B_k < k && B_n < n ? B[B_k * n + B_n] : (elmType) 0.0f;
            }
        }

        __syncthreads();
//      End of copy to shared memory

//      print A_loc
//        if (threadIdx.x == 0) {
//            printf("A_loc:\n");
//            for (int i = 0; i < A_loc_m; i++) {
//                for (int j = 0; j < A_loc_k; j++) {
//                    printf("%f ", (float)A_loc[i][j]);
//                }
//                printf("\n");
//            }
//        }

//      print B_loc
//        if (threadIdx.x == 0) {
//            printf("B_loc:\n");
//            for (int i = 0; i < B_loc_k; i++) {
//                for (int j = 0; j < B_loc_n; j++) {
//                    printf("%f ", (float)B_loc[i][j]);
//                }
//                printf("\n");
//            }
//        }

        // compute the per-thread result css:
//        TODO: #pragma unroll?

        if (ind_m < m && ind_n < n)
        {
            for (int local_offset_k = 0; local_offset_k < block_tiles_k * wmma_k; local_offset_k += wmma_k)
            {
                //            TODO: check col_major vs row_major
                wmma::fragment<wmma::matrix_a, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> A_frag;
                wmma::fragment<wmma::matrix_b, wmma_m, wmma_n, wmma_k, elmType, wmma::row_major> B_frag;

                //            TODO: tile this to reuse loaded fragments?
                //            TODO: ensure warpID_m * wmma_m spans entire array, similar for warpID_n * wmma_n
                wmma::load_matrix_sync(A_frag, &A_loc[warpID_m * wmma_m][local_offset_k], A_loc_k_true);
                wmma::load_matrix_sync(B_frag, &B_loc[local_offset_k][warpID_n * wmma_n], B_loc_n_true);

                wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);
            }
        }
        __syncthreads();
    }


    // Update C in global memory with the per-thread result css.

//    TODO: should maybe check bounds, handle warp matrices that are not full


//    TODO: check earlier and don't do matrix multiplication if out of bounds
    if (ind_m < m && ind_n < n) {
//        if (laneID == 0) {
//            printf("warpID: %d\n", warpID);
//        }
        wmma::store_matrix_sync(&C[ind_m * n + ind_n], C_frag, n, wmma::mem_row_major);
    }
}


#endif //CODE_MATMUL_TENSOR_CUH
