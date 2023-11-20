#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

void MatVecMulUpdateCPU(double* A, double* b, int n) {
    double norm_b = 0.0;
    double* _b = (double*)malloc(n * sizeof(double));
    memcpy(_b, b, n * sizeof(double));

    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int j = 0; j < n; j++)
            s += A[i * n + j] * _b[j];
        b[i] = s;
    }

    free(_b);
}

void VecNormCPU(double* norm_b, double* b, int n) {
    *norm_b = 0.0;
    for (int i = 0; i < n; i++)
        *norm_b += b[i] * b[i];
    *norm_b = sqrt(*norm_b);
}

void VecNormalizeCPU(double* b, double norm_b, int n) {
    for (int i = 0; i < n; i++)
        b[i] /= norm_b;
}

int main() {
    int niter = 3000;
    int n = 2048;
    srand(time(NULL));
    double* A = (double*)malloc(n * n * sizeof(double));
    double* b = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        b[i] = 1.0;
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (rand() % 10000) / 10000.; //
        }
    }
    double norm_b = 0.0;
    for (int i = 0; i < niter; i++) {
        MatVecMulUpdateCPU(A, b, n);
        VecNormCPU(&norm_b, b, n);
        VecNormalizeCPU(b, norm_b, n);
    }

    for (int i = 0; i < n; i++) {
        printf("%lf\n", b[i]);
    }
    printf("%lf\n", b[0]);
}
