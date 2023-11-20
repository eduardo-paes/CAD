#include <stdio.h>
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

#define NUM_THREADS 4
#define MATRIX_DIM 256
#define NUM_ITER 1

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
            A[i * N + j] = (rand() % 10000) / 10000.;
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
    // Calcula a soma dos quadrados dos elementos do vetor
    *norm_b = 0.0;
    for (int i = 0; i < N; i++)
    {
        *norm_b += b[i] * b[i];
    }

    // Retorna a raiz quadrada da soma
    *norm_b = sqrt(*norm_b);

    printf("Norm: %f\n", *norm_b);
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
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Multiplica a matriz A pelo vetor b
    if (row < N && col == 0)
    {
        double s = 0.0;
        for (int j = 0; j < N; j++)
            s += d_A[row * N + j] * d_b[j];
        d_c[row] = s;
        // printf("b[%d] = %f\n", row, d_c[row]);
    }
    __syncthreads();

    // Copia o resultado para o vetor b
    if (row < N && col == 0) d_b[row] = d_c[row];
    __syncthreads();
}

// Função para calcular a norma de um vetor
__global__ void vecNorm_GPU(double *d_b, double *d_norm, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    int bdm = blockDim.x;

    __shared__ double psum[MATRIX_DIM];

    psum[lid] = gid < N ? d_b[gid] * d_b[gid] : 0.0;

    __syncthreads();

    for (int i = bdm / 2; i > 0; i /= 2)
    {
        if (lid < i) psum[lid] += psum[lid + i];
        __syncthreads();
    }

    if (row == 0 && col == 0)
    {
        atomicAdd(d_norm, sqrt(psum[0]));
        printf("Norm: %f\n", *d_norm);
    }
    __syncthreads();
}

// Função para normalizar um vetor
__global__ void vecNormalize_GPU(double *d_b, double *d_norm, int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Divide o vetor pelo valor da norma
    if (row < N && col == 0) d_b[row] /= *d_norm;
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

    // Define os tamanhos dos vetores e matrizes que serão instanciados
    double size_mat = N * N * sizeof(double);
    double size_vec = N * sizeof(double);

    // Declara a matriz e os vetores do device
    double *d_A, *d_b, *d_c, *d_norm_b;

    // Aloca memória para matriz e vetor na GPU
    GCE(cudaMalloc((void **)&d_A, size_mat));
    GCE(cudaMalloc((void **)&d_b, size_vec));
    GCE(cudaMalloc((void **)&d_c, size_vec));
    GCE(cudaMalloc((void **)&d_norm_b, sizeof(double)));

    // Transfere dados do host para o device | Params: dest, src, count, kind
    GCE(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    GCE(cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));
    GCE(cudaMemcpy(d_norm_b, &h_norm_b, sizeof(double), cudaMemcpyHostToDevice));

    // Define o tamanho dos blocos e da grid
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    // Define structs das threads e blocos
    dim3 threads(NUM_THREADS, NUM_THREADS);
    dim3 blocks(NUM_BLOCKS, NUM_BLOCKS);

    for (int i = 0; i < niters; i++)
    {
        // b rcebe o resultado de Ab
        matMulVec_GPU<<<blocks, threads>>>(d_A, d_b, d_c, N);
        cudaDeviceSynchronize();

        // Calcula a norma de b
        vecNorm_GPU<<<blocks, threads>>>(d_b, d_norm_b, N);
        cudaDeviceSynchronize();

        // Normaliza b
        vecNormalize_GPU<<<blocks, threads>>>(d_b, d_norm_b, N);
        cudaDeviceSynchronize();
    }

    // Copia resultados para o host
    GCE(cudaMemcpy(h_b, d_b, size_vec, cudaMemcpyDeviceToHost));

    // Retorna o autovalor dominante
    printf("Result: %f\n", h_b[0]);

    // Libera memória
    cudaFree(d_A);
    cudaFree(d_b);
    free(h_A);
    free(h_b);
}
#pragma endregion

// Função principal
int main(int argc, char **argv)
{
    // Parâmetros gerais
    int N = MATRIX_DIM;
    int niters = NUM_ITER;

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