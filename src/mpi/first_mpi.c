#include <mpi.h>
#include <stdio.h>
#include <string.h>

// Compile: mpicc first_mpi.c
// Run: mpirun -np 4 ./a.out

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int size; // Número de processos instanciados
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank; // Identificador do processo
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        char message[] = "CEFET";
        // Envia mensagem do processo 0 para o processo 1
        MPI_Send(message, strlen(message) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1)
    {
        char message[10];
        MPI_Status status;
        // Recebe mensagem do processo 0
        MPI_Recv(message, 10, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        printf("[%d][%d] %s\n", rank, size, message);
    }

    // Sincronizador de processos
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}