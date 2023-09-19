#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

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

void mul(double **R, double **A, double **B, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double s = 0.0;
            for (int k = 0; k < n; k++)
            {
                s += A[i][k] * B[k][j];
            }
            R[i][j] = s;
        }
    }
}

void mul2(double **R, double **A, double **B, int n)
{
    double **Bt = newmat(n);

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            Bt[i][j] = B[j][i];
        }
    }

    double s;

#pragma omp parallel for
    for (int i = 0; i < n; i++)
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
    mul2(R, A, B, n);
    t1 = t();

    printf("Elapsed time %lfs\n", (t1 - t0) / 1000000.);

    freemat(A, n);
    freemat(B, n);
    freemat(R, n);

    // for (int i = 0; i < n; i++)
    // {
    //     for (int j = 0; j < n; j++)
    //     {
    //         printf("%.2lf\t", R[i][j]);
    //     }
    //     printf("\n");
    // }

    return 0;
}