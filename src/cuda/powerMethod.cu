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

#define BLOCKSIZE 512

#pragma endregion

#pragma region Utils
// Função para obter o tempo atual em segundos
double getTime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 * 1000 + tv.tv_usec) / 1000000.;
}

// Função para imprimir um vetor
void printVector(double *v, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%f ", v[i]);
    }
    printf("\n");
}

// Função para imprimir uma matriz
void printMatrix(double **A, int N)
{
    for (int i = 0; i < N; i++)
    {
        printVector(A[i], N);
    }
}

// Função para liberar a memória alocada para uma matriz
void freeMatrix(double **A, int N)
{
    for (int i = 0; i < N; i++)
        free(A[i]);
    free(A);
}

// Função para criar uma matriz
double ** createMatrix(int N)
{
    double **A = (double **)calloc(N, sizeof(double *));
    for (int i = 0; i < N; i++)
        A[i] = (double *)malloc(N * sizeof(double));
    return A;
}

// Função para criar um vetor
double * createVector(int N)
{
    double *v = (double *)malloc(N * sizeof(double));
    return v;
}
#pragma endregion

#pragma region CPU

// Função para multiplicar uma matriz por um vetor
void matMulVec(double **A, double *b, int N)
{
    // Declara o vetor c para armazenar o resultado
    double *c = (double *)malloc(N * sizeof(double));

    // Multiplica a matriz A pelo vetor b
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            c[i] += A[i][j] * b[j];
        }
    }

    // Transfere o resultado para o vetor b
    for (int i = 0; i < N; i++)
    {
        b[i] = c[i];
    }

    free(c);
}

// Função para calcular a norma de um vetor
void vecNorm(double *b, double *norm, int N)
{
    double sum = 0.0;

    // Calcula a soma dos quadrados dos elementos do vetor
    for (int i = 0; i < N; i++)
    {
        sum += b[i] * b[i];
    }

    // Retorna a raiz quadrada da soma
    *norm = sqrt(sum);
}

// Função para normalizar um vetor
void vecNormalize(double *b, double norm, int N)
{
    // Divide o vetor pelo valor da norma
    for (int i = 0; i < N; i++)
    {
        b[i] = b[i] / norm;
    }
}

// Função para dividir um vetor pelo seu último elemento
double *divideVecByLast(double *b, double last, int N)
{
    for (int i = 0; i < N; i++)
    {
        b[i] = b[i] / last;
    }
    return b;
}

// Método da potência para calcular o autovalor dominante na CPU
double powerMethod_CPU(double **A, int niters, int N)
{
    double *b = (double *)malloc(N * sizeof(double));

    // Preenche b com 1's
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

        // Divide b pelo último elemento de b
        // b = divideByLast(b, N);
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calcula o produto de uma linha da matriz pelo vetor
    if (idx < N)
    {
        double sum = 0.0;
        for (int j = 0; j < N; j++)
        {
            sum += d_A[idx * N + j] * d_b[j];
        }
        d_c[idx] = sum;
    }

    // Aguarda todas as threads terminarem
    __syncthreads();

    // Copia o resultado para o vetor b
    if (idx < N)
    {
        d_b[idx] = d_c[idx];
    }

    // Aguarda todas as threads terminarem
    __syncthreads();
}

// Função para calcular a norma de um vetor
__global__ void vecNorm_GPU(double *d_b, double *d_norm, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calcula a soma dos quadrados dos elementos do vetor
    if (idx < N)
    {
        atomicAdd(d_norm, d_b[idx] * d_b[idx]);
    }

    // Aguarda todas as threads terminarem
    __syncthreads();

    // Uma única thread calcula a raiz quadrada da soma e armazena em d_norm
    if (idx == 0)
    {
        *d_norm = sqrt(*d_norm);
    }

    // Aguarda todas as threads terminarem
    __syncthreads();
}

// Função para normalizar um vetor
__global__ void vecNormalize_GPU(double *d_b, double *d_norm, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Divide o vetor pelo valor da norma
    if (idx < N)
    {
        d_b[idx] /= *d_norm;
    }

    // Aguarda todas as threads terminarem
    __syncthreads();
}

// Método da potência para calcular o autovalor dominante na GPU
void powerMethod_GPU(double **h_A, int niters, int N)
{
    printf("Starting power method in GPU...\n");
    
    // Preenche vetor b do host com 1's
    double *h_b = createVector(N);
    for (int i = 0; i < N; i++)
        h_b[i] = 1;

    // Define vetor c do host
    double *h_c = createVector(N);

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

    // Transfer data from host to device
    GCE(cudaMemcpy(d_A, h_A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    GCE(cudaMemcpy(d_b, h_b, N * sizeof(double), cudaMemcpyHostToDevice));
    GCE(cudaMemcpy(d_c, h_c, N * sizeof(double), cudaMemcpyHostToDevice));
    GCE(cudaMemcpy(d_norm_b, &h_norm_b, sizeof(double), cudaMemcpyHostToDevice));

    // Define o tamanho dos blocos e da grid
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Copia matriz e vetor para a GPU
    GCE(cudaMemcpy(d_A, h_A, size_mat, cudaMemcpyHostToDevice));
    GCE(cudaMemcpy(d_b, h_b, size_vec, cudaMemcpyHostToDevice));

    printf("Starting iteration...\n");
    for (int i = 0; i < niters; i++)
    {
        // b rcebe o resultado de Ab
        matMulVec_GPU<<<gridSize, blockSize>>>(d_A, d_b, d_c, N);
        cudaDeviceSynchronize();

        // Calcula a norma de b
        vecNorm_GPU<<<gridSize, blockSize>>>(d_b, d_norm_b, N);
        cudaDeviceSynchronize();

        // Normaliza b
        vecNormalize_GPU<<<gridSize, blockSize>>>(d_b, d_norm_b, N);
        cudaDeviceSynchronize();
    }

    // Copia resultados para o host
    GCE(cudaMemcpy(h_b, d_b, size_vec, cudaMemcpyDeviceToHost));

    // Recupera o autovalor dominante
    double result = h_b[0];

    // cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_A);
    free(h_b);
    free(h_c);

    // Retorna o autovalor dominante
    printf("Result: %f\n", result);
}
#pragma endregion

// Função principal
int main(int argc, char **argv)
{
    // Parâmetros gerais
    int N = 2;
    int niters = 5;
    double values[] = {2, -12, 1, -5};

    // Declara a matriz A
    double **A = createMatrix(N);

    // Preenche a matriz A com os valores
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = values[i * N + j];

    // Variáveis para medir o tempo
    double t0, t1;

    // Calcula o autovalor dominante
    t0 = getTime();
    powerMethod_GPU(A, niters, N);
    t1 = getTime();

    printf("Sequential Time: %.6f ms\n", t1 - t0);

    // Libera a memória
    freeMatrix(A, N);

    return 0;
}