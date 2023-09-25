/* Código do algoritmo de regressão linear simples
 * Autor: Eduardo Paes Silva
 * Data: 11/09/2023
 * ----------------------------------------------
 * Compilar: gcc -Xpreprocessor -fopenmp -lomp reg.c -Ofast
 * ----------------------------------------------
 * Referências:
 * https://en.wikipedia.org/wiki/Simple_linear_regression
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

// Calcula a média de um vetor
// Equação: mean(x) = sum(x) / n
double mean(double *x, int n)
{
    double sum = 0.0;
    // Paraleliza o loop de soma utilizando a diretiva reduction para realizar
    // a soma parcial de forma paralela e depois somar os resultados parciais
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        sum += x[i];
    }
    return sum / n;
}

// Calcula a variância de um vetor
// Equação: var(x) = sum((x - x_mean)²) / n
double var(double *x, double x_mean, int n)
{
    double sum = 0.0;
    // Paraleliza o loop de soma utilizando a diretiva reduction
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        sum += (x[i] - x_mean) * (x[i] - x_mean);
    }
    return sum;
}

// Calcula a covariância de dois vetores
// Equação: cov(x, y) = sum((x - x_mean) * (y - y_mean)) / n
double cov(double *x, double x_mean, double *y, double y_mean, int n)
{
    double sum = 0.0;
    // Paraleliza o loop de soma utilizando a diretiva reduction
#pragma omp parallel for reduction(+ : sum)
    for (int i = 0; i < n; i++)
    {
        sum += (x[i] - x_mean) * (y[i] - y_mean);
    }
    return sum;
}

// Gera um número aleatório entre -c e c
double randRange(double c)
{
    double v = (double)rand() / RAND_MAX;
    return (2 * v - 1) * c;
}

// Algoritmo de regressão linear simples
int main()
{
    srand(time(NULL));
    int n = 10000000;
    // Alocação dos vetores de X e Y
    double *x = (double *)malloc(sizeof(double) * n);
    double *y = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++)
    {
        x[i] = i;
        // O algoritmo precisa encontrar os valores de a (alpha) e b (beta)
        // da equação y = a * x + b
        y[i] = 2.123 * x[i] + 4.321 + randRange(5000.0);
    }

    // Calcula os valores de média de x e y
    double x_mean = mean(x, n);
    double y_mean = mean(y, n);

    // Calcula a covariância de x e y
    double cov_xy = cov(x, x_mean, y, y_mean, n);

    // Calcula a variância de x e y
    double var_x = var(x, x_mean, n);
    double var_y = var(y, y_mean, n);

    // Calcula o valor de beta e alpha
    double beta = cov_xy / var_x;
    double alpha = y_mean - beta * x_mean;

    // Calcula o valor de rho
    double rho = cov_xy / (sqrt(var_x * var_y));

    // Imprime os valores de beta, alpha e rho
    printf("beta = %lf | alpha = %lf | rho = %lf\n", beta, alpha, rho);

    return 0;
}