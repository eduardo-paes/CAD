#include <mpi.h>
#include <math.h>
#include <stdio.h>

// Compile: mpi++ reg_mpi.cc   
// Run: mpirun -np 4 ./a.out

int main(int argc, char **argv)
{
    int N = 1000000;
    MPI_Init(&argc, &argv);

    int size; // Número de processos instanciados
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank; // Identificador do processo
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double * x = NULL;
    double * y = NULL;

    int chunk = N / size;

    // Apenas o processo root possui os dados
    if (rank == 0) {
        x = new double[N];
        y = new double[N];

        for (int i = 0; i < N; i++)
        {
            x[i] = i;
            y[i] = 1.65 * x[i] + 5.37;
        }
    }

    double * x_local = new double[chunk];
    double * y_local = new double[chunk];

    double sum_x_local = 0.0;
    double sum_y_local = 0.0;
    double sum_xy_local = 0.0;

    double mean_x, mean_y;

    // Distribui o dado a ser computado em partes para os processos
    MPI_Scatter(x, chunk, MPI_DOUBLE, x_local, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, chunk, MPI_DOUBLE, y_local, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++)
    {
        sum_x_local += x_local[i] / N;
        sum_y_local += y_local[i] / N;
    }

    // Recebe os valores parciais das médias e reúne nas médias finais
    MPI_Reduce(&sum_x_local, &mean_x, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_y_local, &mean_y, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Envia os valores das médias para os processos
    MPI_Bcast(&mean_x, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&mean_y, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    sum_x_local = 0.0;
    sum_y_local = 0.0;
    for (int i = 0; i < chunk; i++)
    {
        sum_x_local += (x_local[i] - mean_x) * (x_local[i] - mean_x);
        sum_y_local += (y_local[i] - mean_y) * (y_local[i] - mean_y);
        sum_xy_local += (x_local[i] - mean_x) * (y_local[i] - mean_y);
    }

    double var_x, var_y, cov_xy;
    MPI_Reduce(&sum_x_local, &var_x, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_y_local, &var_y, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_xy_local, &cov_xy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double beta = cov_xy / var_x;
        double alpha = mean_y - beta * mean_x;
        double rho = cov_xy / sqrt(var_x * var_y);
        printf("beta = %lf | alpha = %lf | rho = %lf\n", beta, alpha, rho);
    }

    MPI_Finalize();

    return 0;
}