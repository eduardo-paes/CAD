#include <mpi.h>
#include <stdio.h>
#include <string.h>

// Compile: mpicc first_mpi.c
// Run: mpirun -np 4 ./a.out

int main(int argc, char **argv)
{
    int N = 4;
    MPI_Init(&argc, &argv);

    int size; // Número de processos instanciados
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank; // Identificador do processo
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int *v_local = (int *)malloc(sizeof(int) * N);

    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
        {
            v_local[i] = 1;
        }
    }

    MPI_Bcast(
        void *data,
        int count,
        MPI_Datatype datatype,
        int root,
        MPI_Comm communicator)

        if (rank == 2)
    {
        char message[] = "CEFET";
        printf("[%d][%d] %s\n", rank, size, message);
    }

    MPI_Finalize();
    return 0;
}