#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#pragma region Utils
// Função para obter o tempo atual em segundos
double getTime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 * 1000 + tv.tv_usec) / 1000000.;
}

// Função para imprimir um vetor
void printVector(double* v, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("%f ", v[i]);
    }
    printf("\n");
}

// Função para imprimir uma matriz
void printMatrix(double** A, int N)
{
    for (int i = 0; i < N; i++)
    {
        printVector(A[i], N);
    }
}

// Função para liberar a memória alocada para uma matriz
void freeMatrix(double** A, int N)
{
    for (int i = 0; i < N; i++)
        free(A[i]);
    free(A);
}

// Função para criar uma matriz
double** createMatrix(int N)
{
    double** A = (double**)calloc(N, sizeof(double*));
    for (int i = 0; i < N; i++)
        A[i] = (double*)malloc(N * sizeof(double));
    return A;
}

// Função para criar um vetor
double* createVector(int N)
{
    double* v = (double*)malloc(N * sizeof(double));
    return v;
}
#pragma endregion

#pragma region CPU

// Função para multiplicar uma matriz por um vetor
void matMulVec(double** A, double* b, int N)
{
    // Declara o vetor _b para armazenar o resultado
    double* _b = (double*)malloc(N * sizeof(double));
    memcpy(_b, b, N * sizeof(double));

    // Multiplica a matriz A pelo vetor b
    for (int i = 0; i < N; i++)
    {
        double s = 0.0;
        for (int j = 0; j < N; j++)
            s += A[i][j] * _b[j];
        b[i] = s;
    }

    free(_b);
}

// Função para calcular a norma de um vetor
void vecNorm(double* b, double* norm_b, int N)
{
    // Calcula a soma dos quadrados dos elementos do vetor
    *norm_b = 0.0;
    for (int i = 0; i < N; i++)
    {
        *norm_b += b[i] * b[i];
    }

    // Retorna a raiz quadrada da soma
    *norm_b = sqrt(*norm_b);
}

// Função para normalizar um vetor
void vecNormalize(double* b, double norm_b, int N)
{
    // Divide o vetor pelo valor da norma
    for (int i = 0; i < N; i++)
        b[i] /= norm_b;
}

// Função para dividir um vetor pelo seu último elemento
double* divideVecByLast(double* b, double last, int N)
{
    for (int i = 0; i < N; i++)
    {
        b[i] = b[i] / last;
    }
    return b;
}

// Método da potência para calcular o autovalor dominante na CPU
double powerMethod_CPU(double** A, int niters, int N)
{
    double* b = (double*)malloc(N * sizeof(double));

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
    }

    // Recupera o autovalor dominante
    double result = b[0];
    free(b);

    // Retorna o autovalor dominante
    printf("Result: %f\n", result);
    return result;
}

#pragma endregion

// Função principal
int main(int argc, char** argv)
{
    // Parâmetros gerais
    int N = 2;
    int niters = 5;
    double values[] = { 2, -12, 1, -5 };

    // Declara a matriz A
    double** A = createMatrix(N);

    // Preenche a matriz A com os valores
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            A[i][j] = values[i * N + j];

    // Variáveis para medir o tempo
    double t0, t1;

    // Calcula o autovalor dominante
    t0 = getTime();
    powerMethod_CPU(A, niters, N);
    t1 = getTime();

    printf("Sequential Time: %.6f ms\n", t1 - t0);

    // Libera a memória
    freeMatrix(A, N);

    return 0;
}