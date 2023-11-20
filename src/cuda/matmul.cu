#include <stdio.h>

// CUDA kernel for matrix-vector multiplication
__global__ void matMulVecCUDA(double *A, double *b, double *result, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        double s = 0.0;
        for (int j = 0; j < N; j++)
            s += A[i * N + j] * b[j];
        result[i] = s;
        printf("result[%d] = %f\n", i, result[i]);
    }
}

// Function to launch the CUDA kernel
void matMulVec(double *A, double *b, int N)
{
    // Declare device variables
    double *d_A, *d_b, *d_result;

    // Allocate device memory
    cudaMalloc((void **)&d_A, N * N * sizeof(double));
    cudaMalloc((void **)&d_b, N * sizeof(double));
    cudaMalloc((void **)&d_result, N * sizeof(double));

    // Copy input data to device
    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);

    // Set up grid and block sizes
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch the CUDA kernel
    matMulVecCUDA<<<gridSize, blockSize>>>(d_A, d_b, d_result, N);

    // Copy the result back to host
    cudaMemcpy(b, d_result, N * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_result);
}

int main()
{
    // Example usage
    int N = 4;
    double A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    double b[] = {1, 2, 3, 4};

    matMulVec(A, b, N);

    return 0;
}
