#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#pragma region Definitions
#define GCE(ans) {gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#define MATRIX_SIZE 2048
#pragma endregion

#pragma region Utils
// Função para obter o tempo atual em segundos
double getTime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 * 1000 + tv.tv_usec) / 1000000.;
}

// Função para criar uma matriz
double * createMatrix(int N)
{
    double* A = (double*)malloc(N * N * sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (rand() % 100) / 100.;
        }
    }
    return A;
}
#pragma endregion

#pragma region CPU

// Função para multiplicar uma matriz por um vetor
void matMulVec(double *A, double *b, int N)
{
    // Declara o vetor _b para armazenar o resultado
    double* _b = (double*)malloc(N * sizeof(double));
    memcpy(_b, b, N * sizeof(double));

    // Multiplica a matriz A pelo vetor b
    for (int i = 0; i < N; i++) {
        double s = 0.0;
        for (int j = 0; j < N; j++)
            s += A[i * N + j] * _b[j];
        b[i] = s;
        // printf("b[%d] = %f\n", i, b[i]);
    }

    free(_b);
}

// Função para calcular a norma de um vetor
void vecNorm(double *b, double *norm_b, int N)
{
    // Calcula a soma dos quadrados dos elementos do vetor e retorna a raiz quadrada da soma
    double s = 0.0;
    for (int i = 0; i < N; i++)
        s += b[i] * b[i];
    *norm_b = sqrt(s);

    // printf("Norm: %f\n", *norm_b);
}

// Função para normalizar um vetor
void vecNormalize(double *b, double norm_b, int N)
{
    // Divide o vetor pelo valor da norma
    for (int i = 0; i < N; i++)
        b[i] /= norm_b;
}

// Método da potência para calcular o autovalor dominante na CPU
double powerMethod_CPU(double *A, int niters, int N)
{
    // Inicializa o vetor b com 1's
    double *b = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
        b[i] = 1;

    double norm_b;
    for (int i = 0; i < niters; i++)
    {
        // b rcebe o resultado de Ab
        matMulVec(A, b, N);

        // Calcula a norma de b
        vecNorm(b, &norm_b, N);

        // Normaliza b
        vecNormalize(b, norm_b, N);
    }

    // Recupera o autovalor dominante
    double result = b[0];
    free(b);

    // Retorna o autovalor dominante
    printf("Result: %f\n", result);
    return result;
}

#pragma endregion

#pragma region GPU
// Função para multiplicar uma matriz por um vetor
__global__ void matMulVec_GPU(double *d_A, double *d_b, double *d_c, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Multiplica a matriz A pelo vetor b
    double s = 0.0;
    if (tid < N) {
        for (int j = 0; j < N; j++)
            s += d_A[tid * N + j] * d_b[j];
        // printf("b[%d] = %f\n", tid, s);
    }
    __syncthreads();

    // Copia o resultado para o vetor b
    if (tid < N) d_b[tid] = s;
    __syncthreads();
}

// Função para calcular a norma de um vetor
__global__ void vecNorm_GPU(double *d_b, double *d_norm_b, int N)
{
    // int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // int lid = threadIdx.x;
    // int blockSize = blockDim.x;

    // __shared__ double partialSum[MATRIX_SIZE];

    //  // Calcula a soma dos quadrados dos elementos do vetor
    // partialSum[lid] = (gid < N) ? (d_b[gid] * d_b[gid]) : 0.0;
    // __syncthreads();

    // // Redução
    // for (int i = blockSize / 2; i > 0; i /= 2)
    // {
    //     if (lid < i) partialSum[lid] += partialSum[lid + i];
    //     __syncthreads();
    // }

    // // Retorna a raiz quadrada da soma
    // if (lid == 0) *d_norm_b = sqrt(partialSum[0]);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        // Calcula a soma dos quadrados dos elementos do vetor e retorna a raiz quadrada da soma
        double s = 0.0;
        for (int i = 0; i < N; i++)
            s += d_b[i] * d_b[i];
        *d_norm_b = sqrt(s);
        // printf("Norm: %f\n", *d_norm_b);
    }
    __syncthreads();
}

// Função para normalizar um vetor
__global__ void vecNormalize_GPU(double *d_b, double *d_norm_b, int N)
{
    // Divide o vetor pelo valor da norma
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        d_b[tid] /= *d_norm_b;
    }
    __syncthreads();
}

// Método da potência para calcular o autovalor dominante na GPU
void powerMethod_GPU(double *h_A, int niters, int N)
{
    // Preenche vetor b do host com 1's
    double *h_b = (double *)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) h_b[i] = 1;

    // Declara a norma de b do host
    double h_norm_b = 0.0;

    // Declara a matriz e os vetores do device
    double *d_A, *d_b, *d_c, *d_norm_b;

    // Aloca memória para matriz e vetor na GPU
    GCE(cudaMalloc((void **)&d_A, N * N * sizeof(double)));
    GCE(cudaMalloc((void **)&d_b, N * sizeof(double)));
    GCE(cudaMalloc((void **)&d_c, N * sizeof(double)));
    GCE(cudaMalloc((void **)&d_norm_b, sizeof(double)));

    // Transfere dados do host para o device | Params: dest, src, count, kind
    GCE(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    GCE(cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));
    GCE(cudaMemcpy(d_norm_b, &h_norm_b, sizeof(double), cudaMemcpyHostToDevice));

    // Definição dos blocos e dimensão das threads
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    for (int i = 0; i < niters; i++)
    {
        // b rcebe o resultado de Ab
        matMulVec_GPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_b, d_c, N);
        cudaDeviceSynchronize();

        // Calcula a norma de b
        vecNorm_GPU<<<blocksPerGrid, threadsPerBlock>>>(d_b, d_norm_b, N);
        cudaDeviceSynchronize();

        // Normaliza b
        vecNormalize_GPU<<<blocksPerGrid, threadsPerBlock>>>(d_b, d_norm_b, N);
        cudaDeviceSynchronize();
    }

    // Copia resultados para o host
    GCE(cudaMemcpy(h_b, d_b, N * sizeof(double), cudaMemcpyDeviceToHost));

    // Retorna o autovalor dominante
    printf("Result: %f\n", h_b[0]);

    // Libera memória
    free(h_A);
    free(h_b);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_norm_b);
}
#pragma endregion

// Função principal
int main(int argc, char **argv)
{
    // Parâmetros gerais
    const int N = MATRIX_SIZE;
    const int niters = 100;

    // Inicializa a matriz A
    double *A = createMatrix(N);

    // Variáveis para medir o tempo
    double t0, t1;

    // Calcula o autovalor dominante na CPU
    t0 = getTime();
    powerMethod_CPU(A, niters, N);
    t1 = getTime();
    printf("CPU Time: %.6f ms\n", t1 - t0);

    // Calcula o autovalor dominante na GPU
    t0 = getTime();
    powerMethod_GPU(A, niters, N);
    t1 = getTime();
    printf("GPU Time: %.6f ms\n", t1 - t0);
    
    return 0;
}