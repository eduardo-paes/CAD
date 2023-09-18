#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>

// Elapsed time                 4.711617s
// -O3                          1.608689s
// -Ofast                       0.196723s
// Clang + -Ofast               0.194703s
// gcc mulmat.c -Ofast -fopenmp
// gcc -Xpreprocessor -fopenmp -lomp mulmat.c -Ofast

long long int t()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * (1000000) + tv.tv_usec;
}

double **newmat(int n)
{
    double **A = (double **)malloc(sizeof(double *) * n);

    for (int i = 0; i < n; i++)
    {
        A[i] = (double *)malloc(sizeof(double) * n);
    }

    return A;
}

void freemat(double **A, int n)
{
    for (int i = 0; i < n; i++)
    {
        free(A[i]);
    }
    free(A);
}

typedef struct
{
    double **R;
    double **A;
    double **Bt;
    int n;
    int init;
    int final;
} param_mulmat;

void *mul_pthread(void *argv)
{
    param_mulmat *arg = (param_mulmat *)argv;

    double **R = arg->R;
    double **A = arg->A;
    double **Bt = arg->Bt;
    int n = arg->n;
    int init = arg->init;
    int final = arg->final;

    double s = 0;
    for (int i = init; i < final; i++)
    {
        for (int j = 0; j < n; j++)
        {
            s = 0.0;
            for (int k = 0; k < n; k++)
            {
                s += A[i][k] * Bt[j][k];
            }
            R[i][j] = s;
        }
    }

    return NULL;
}

void mul(double **R, double **A, double **B, int n)
{
    int nthreads = 8;
    int chunck = n / nthreads;
    int n_biggers_chunck = n % nthreads;
    pthread_t threads[nthreads];
    param_mulmat arg[nthreads];

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
    for (int i = 0; i < nthreads; i++)
    {
        arg[i].R = R;
        arg[i].A = A;
        arg[i].Bt = Bt;
        arg[i].n = n;

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

        pthread_create(&threads[i], NULL, mul_pthread, (void *)&arg[i]);
    }

    // Espera as threads terminarem
    for (int i = 0; i < nthreads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    freemat(Bt, n);
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        return 0;
    }

    int n = atoi(argv[1]);
    srand(time(NULL));
    double **R = newmat(n);
    double **A = newmat(n);
    double **B = newmat(n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            B[i][j] = (rand() % 10000) / 10000.;
            A[i][j] = (rand() % 10000) / 10000.;
        }
    }

    long long t0, t1;

    t0 = t();
    mul(R, A, B, n);
    t1 = t();

    printf("Elapsed time %lfs\n", (t1 - t0) / 1000000.);

    double s = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // printf("%.2lf\t", R[i][j]);
            s += R[i][j] / (n * n);
        }
        // printf("\n");
    }
    printf("Sum: %.4lf\n", s);

    freemat(A, n);
    freemat(B, n);
    freemat(R, n);

    return 0;
}