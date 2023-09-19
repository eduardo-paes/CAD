#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

double mean(double *x, int n)
{
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        sum += x[i];
    }
    return sum / n;
}

double var(double *x, double x_mean, int n)
{
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        sum += (x[i] - x_mean) * (x[i] - x_mean);
    }
    return sum;
}

double cov(double *x, double x_mean, double *y, double y_mean, int n)
{
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        sum += (x[i] - x_mean) * (y[i] - y_mean);
    }
    return sum;
}

double randRange(double c)
{
    double v = (double)rand() / RAND_MAX;
    return (2 * v - 1) * c;
}

int main()
{
    srand(time(NULL));
    int n = 10000000;
    double *x = (double *)malloc(sizeof(double) * n);
    double *y = (double *)malloc(sizeof(double) * n);

    for (int i = 0; i < n; i++)
    {
        x[i] = i;
        y[i] = 2.123 * x[i] + 4.321 + randRange(5000.0);
    }

    double x_mean = mean(x, n);
    double y_mean = mean(y, n);
    double cov_xy = cov(x, x_mean, y, y_mean, n);

    double var_x = var(x, x_mean, n);
    double var_y = var(y, y_mean, n);

    double beta = cov_xy / var_x;
    double alpha = y_mean - beta * x_mean;

    double rho = cov_xy / (sqrt(var_x * var_y));

    printf("beta = %lf alpha = %lf rho = %lf\n", beta, alpha, rho);

    return 0;
}