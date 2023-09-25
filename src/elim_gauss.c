/* Código para resolução de sistemas lineares utilizando eliminação de Gauss
 * Autor: Eduardo Paes Silva
 * Data: 24/09/2023
 * ----------------------------------------------
 * Compilar: gcc -Xpreprocessor -fopenmp -lomp elim_gauss.c -Ofast
 * ----------------------------------------------
 * Referências:
 * https://en.wikipedia.org/wiki/Gaussian_elimination
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#define N 1000

// Retorna o tempo atual em microsegundos
long long int current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * (1000000) + tv.tv_usec;
}

// Aplica a eliminação de Gauss em uma matriz A e vetor b
void gauss_elim(double **A, double *b, int n)
{
    double piv;
    for (int k = 0; k < n - 2; k++)
    {
#pragma omp parallel for
        for (int i = k + 1; i < n - 1; i++)
        {
            piv = A[i][k] / A[k][k];
            b[i] -= piv * b[k];

            for (int j = k; j < n - 1; j++)
            {
                A[i][j] -= piv * A[k][j];
            }
        }
    }
}

// Resolve o sistema triangular superior Ax = b
void solve_sup(double **A, double *b, int n)
{
    double *x = calloc(n, sizeof(double));
    double sum;

    // Encontra os valores de x que satisfazem a equação
#pragma omp parallel for
    for (int i = n - 1; i >= 0; i--)
    {
        sum = 0.0;
        for (int j = i + 1; j < n; j++)
        {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }

    // Imprime a validação da solução do sistema
    for (int i = 0; i < n; i++)
    {
        sum = 0.0;
        for (int j = i; j < n; j++)
        {
            sum += A[i][j] * x[j];
        }
        printf(" S = %lf | b[%d] = %lf\n", sum, i, b[i]);
    }
    free(x);
}

// Aloca memória para uma matriz de tamanho n x n
double **new_mat(int n)
{
    double **M = (double **)calloc(n, sizeof(double *));
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        M[i] = (double *)malloc(n * sizeof(double));
    }
    return M;
}

// Preenche uma matriz de tamanho n x n com valores aleatórios
void fill_mat(double **M, double *b, int n)
{
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        b[i] = random() / 2.0;
        for (int j = 0; j < n; j++)
        {
            M[i][j] = random() / 2.0;
        }
    }
}

// Imprime uma matriz de tamanho n x n
void print_mat(double **A, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%lf ", A[i][j]);
        }
        printf("\n");
    }
}

// Função para resolver um sistema linear utilizando eliminação de Gauss
// de uma matriz aleatória de tamanho N x N
int main()
{
    // Aloca memória para a matriz A e o vetor b
    double **A = new_mat(N);
    double *b = (double *)calloc(N, sizeof(double));

    // Variáveis para medir o tempo de execução
    long long t0, t1;

    t0 = current_time();

    // Preenche a matriz A e o vetor b
    fill_mat(A, b, N);

    // Aplica a eliminação de Gauss
    gauss_elim(A, b, N);

    // Resolve o sistema triangular superior
    solve_sup(A, b, N);

    t1 = current_time();
    printf("Elapsed time %lfs\n", (t1 - t0) / 1000000.);

    // printf("\n\n");
    // print_mat(A,n);

    return 0;
}