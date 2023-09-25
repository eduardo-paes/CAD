/* Código para multiplicação de matrizes
 * Autor: Eduardo Paes Silva
 * Data: 15/09/2023
 * ----------------------------------------------
 * Compilar: gcc -Xpreprocessor -fopenmp -lomp mulmat.c -lpthread -Ofast
 * ----------------------------------------------
 * Referências:
 * https : // en.wikipedia.org/wiki/Matrix_multiplication_algorithm
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

#define NUM_THREADS 8

// Retorna o tempo atual em microsegundos
long long int current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * (1000000) + tv.tv_usec;
}

// Aloca memória para uma matriz de tamanho n x n
double **newmat(int n)
{
    double **A = (double **)malloc(sizeof(double *) * n);
    for (int i = 0; i < n; i++)
    {
        A[i] = (double *)malloc(sizeof(double) * n);
    }
    return A;
}

// Libera a memória alocada para uma matriz
void freemat(double **M, int n)
{
    for (int i = 0; i < n; i++)
    {
        free(M[i]);
    }
    free(M);
}

// Multiplicação de matrizes simples
void mul_simple(double **R, double **A, double **B, int n)
{
    double **Bt = newmat(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            Bt[i][j] = B[j][i];
        }
    }

    double sum;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += A[i][k] * Bt[j][k];
            }
            R[i][j] = sum;
        }
    }

    // Libera a memória alocada para a matriz transposta
    freemat(Bt, n);
}

// Multiplicação de matrizes utilizando OpenMP
void mul_omp(double **R, double **A, double **B, int n)
{
    double **Bt = newmat(n);
    // Transposição de B para melhorar o acesso a memória
    // Utiliza a diretiva parallel for para paralelizar o loop
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            Bt[i][j] = B[j][i];
        }
    }

    double sum;
    // Paraleliza o loop de multiplicação de matrizes
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += A[i][k] * Bt[j][k];
            }
            R[i][j] = sum;
        }
    }

    // Libera a memória alocada para a matriz transposta
    freemat(Bt, n);
}

// Estrutura para passar os argumentos para a função de multiplicação de matrizes
typedef struct
{
    double **R;
    double **A;
    double **Bt;
    int n;
    int init;
    int final;
} param_mulmat;

// Função executada pelas threads para multiplicar matrizes
void *mul_pthread(void *argv)
{
    // Converte o argumento para a estrutura de argumentos da função
    param_mulmat *param = (param_mulmat *)argv;

    // Extrai os argumentos
    double **R = param->R;    // Matriz resultado
    double **A = param->A;    // Matriz A
    double **Bt = param->Bt;  // Matriz transposta de B
    int n = param->n;         // Dimensão das matrizes
    int init = param->init;   // Início do intervalo de linhas que a thread irá calcular
    int final = param->final; // Fim do intervalo de linhas que a thread irá calcular

    double sum = 0;
    // Multiplica as matrizes
    for (int i = init; i < final; i++)
    {
        for (int j = 0; j < n; j++)
        {
            sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += A[i][k] * Bt[j][k];
            }
            R[i][j] = sum;
        }
    }

    // Termina a thread
    return NULL;
}

// Multiplicação de matrizes utilizando pthreads
void mul_threads(double **R, double **A, double **B, int n)
{
    int chunck = n / NUM_THREADS;           // Número de elementos que cada thread irá calcular
    int n_biggers_chunck = n % NUM_THREADS; // Número de threads com chunck + 1 elementos
    pthread_t threads[NUM_THREADS];         // Vetor de threads
    param_mulmat arg[NUM_THREADS];          // Vetor de argumentos para as threads

    // Cria a matriz transposta de B para otimização da multiplicação
    double **Bt = newmat(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            Bt[i][j] = B[j][i];
        }
    }

    // Cria as threads para a multiplicação
    for (int i = 0; i < NUM_THREADS; i++)
    {
        arg[i].R = R;
        arg[i].A = A;
        arg[i].Bt = Bt;
        arg[i].n = n;

        // Calcula o intervalo de linhas que cada thread irá calcular
        if (i < n_biggers_chunck)
        {
            arg[i].init = i * (chunck + 1);
            arg[i].final = (i + 1) * (chunck + 1);
        }
        else
        {
            arg[i].init = n_biggers_chunck * (chunck + 1) + (i - n_biggers_chunck) * chunck;
            arg[i].final = arg[i].init + chunck;
        }

        // Cria a thread para a multiplicação de matrizes e passa os argumentos
        pthread_create(&threads[i], NULL, mul_pthread, (void *)&arg[i]);
    }

    // Espera as threads terminarem
    for (int i = 0; i < NUM_THREADS; i++)
    {
        pthread_join(threads[i], NULL);
    }

    // Libera a memória alocada para a matriz transposta
    freemat(Bt, n);
}

int main(int argc, char **argv)
{
    // Verifica se o número de argumentos é válido
    if (argc != 2)
    {
        return 0;
    }

    // Converte o argumento para inteiro
    int mat_dim = atoi(argv[1]);
    srand(time(NULL));

    // Aloca memória para as matrizes
    double **R = newmat(mat_dim);
    double **A = newmat(mat_dim);
    double **B = newmat(mat_dim);

    // Variáveis para medir o tempo de execução
    long long t0, t1;

    // Inicializa as matrizes com valores aleatórios
    for (int i = 0; i < mat_dim; i++)
    {
        for (int j = 0; j < mat_dim; j++)
        {
            B[i][j] = (rand() % 10000) / 10000.;
            A[i][j] = (rand() % 10000) / 10000.;
        }
    }

    // Mede o tempo de execução da multiplicação de matrizes simples
    t0 = current_time();
    mul_threads(R, A, B, mat_dim);
    t1 = current_time();

    printf("Threads Mul. | Elapsed time %lfs\n", (t1 - t0) / 1000000.);

    // Mede o tempo de execução da multiplicação de matrizes utilizando OpenMP
    t0 = current_time();
    mul_omp(R, A, B, mat_dim);
    t1 = current_time();

    printf("OMP Mul. | Elapsed time %lfs\n", (t1 - t0) / 1000000.);

    // Mede o tempo de execução da multiplicação de matrizes simples
    t0 = current_time();
    mul_simple(R, A, B, mat_dim);
    t1 = current_time();

    printf("Simple Mul. | Elapsed time %lfs\n", (t1 - t0) / 1000000.);

    // Libera a memória alocada para as matrizes
    freemat(A, mat_dim);
    freemat(B, mat_dim);
    freemat(R, mat_dim);

    return 0;
}