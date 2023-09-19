#include <mpi.h>
#include <math.h>
#include <stdio.h>

// Compile: mpicc mpi_hello.cc   
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
    double * r = NULL;

    int chunk = N / size;

    // Apenas o processo root possui os dados
    if (rank == 0) {
        x = new double[N];
        y = new double[N];
        r = new double[N];

        for (int i = 0; i < N; i++)
        {
            x[i] = cos(i) * cos(i);
            y[i] = sin(i) * sin(i);
        }
    }

    double * x_local = new double[chunk];
    double * y_local = new double[chunk];
    double * r_local = new double[chunk];

    // Distribui o dado a ser computado em partes para os processos
    MPI_Scatter(x, chunk, MPI_DOUBLE, x_local, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(y, chunk, MPI_DOUBLE, y_local, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++)
    {
        r_local[i] = x_local[i] + y_local[i];
    }

    // Recebe o resultado local de cada processo e agrega no resultado final
    MPI_Gather(r_local, chunk, MPI_DOUBLE, r, chunk, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Apenas o processo root imprime o resultado
    if (rank == 0) {
        double sum = 0;
        for (int i = 0; i < N; i++)
        {
            sum += r[i];
        }
        printf("%lf\n", sum/N);
    }

    MPI_Finalize();

    return 0;
}