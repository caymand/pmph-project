#include <cstdio>
#include <mma.h>

using namespace nvcuda;

__global__ void wmma_kernel(half *a, half *b, half *c) {
//    TODO: get warp index to calculate offsets

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}


void print_matrix(half *a, int m, int n)
{
    for (unsigned int i = 0; i < m; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            printf("%f ", (float)a[j * m + i]);
        }
        printf("\n");
    }
}


int main(int argc, char * argv[]) {
    unsigned int m = 16;
    unsigned int n = 16;
    unsigned int k = 16;

//    TODO: malloc instead?
    half host_a[m][k];
    half host_b[k][n];
    half host_c[m][n];

    half *device_a;
    half *device_b;
    half *device_c;

    cudaMalloc(&device_a, m * k * sizeof(half));
    cudaMalloc(&device_b, k * n * sizeof(half));
    cudaMalloc(&device_c, m * n * sizeof(half));

    // Initialize host memory
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < k; j++) {
            host_a[i][j] = i * j;
        }
    }

    for (unsigned int i = 0; i < k; i++) {
        for (unsigned int j = 0; j < n; j++) {
            host_b[i][j] = i * j;
        }
    }

    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < n; j++) {
            host_c[i][j] = 0;
        }
    }

    cudaMemcpy(device_a, host_a, m * k * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, k * n * sizeof(half), cudaMemcpyHostToDevice);

    wmma_kernel<<<1, 32>>>(device_a, device_b, device_c);

    cudaMemcpy(host_c, device_c, m * n * sizeof(half), cudaMemcpyDeviceToHost);

    print_matrix((half*)&host_c, m, n);

    return 0;
}
