#include <stdio.h>
#include <omp.h>
#include <unistd.h>

// gcc -Xpreprocessor -fopenmp -lomp first_omp.c -Ofast

int main()
{
    int sum = 0;
    omp_set_num_threads(4);

    int i = 10;
    int j = 0;

    // Cada thread terá uma cópia privada de i e acessará a variável j compartilhada
#pragma omp parallel private(i) shared(j)
    {
        i++;
        // O valor de i será diferente para cada thread, mas o valor de j será o mesmo
        printf("i = %d | j = %d \n", i, j);
    }
    printf("Out i = %d\n", i);

    // ------------------------------

    i = 10;
    printf("First Private\n");
    // Cada thread terá uma cópia privada de i, mas o valor inicial será 10
#pragma omp parallel firstprivate(i)
    {
        i++;
        printf("i = %d\n", i);
    }
    printf("Out i = %d\n", i);

    // ------------------------------

// Abre uma região paralela
#pragma omp parallel
    {
    // Abre uma região paralela com 3 threads
#pragma omp sections // nowait // Pode não esperar o fim das sections
        {
    // Cada uma das threads executará uma das sections
#pragma omp section
            {printf("Thread 0\n");
}
#pragma omp section
{
    printf("Thread 1\n");
}
#pragma omp section
{
    printf("Thread 2\n");
}
}
}

// ------------------------------

// Abre uma região paralela
#pragma omp parallel
{
    // SINGLE
    int tid = omp_get_thread_num();
#pragma omp single // Apenas uma das threads entrará aqui, a primeira que chegar
    {
        printf("SINGLE %d\n", tid);
    }

    // MASTER
#pragma omp master // Igual ao single, mas apenas a thread principal entra
    {
        printf("MASTER %d\n", tid);
    } // Tem uma barreira implícita aqui

#pragma omp atomic // Realiza apenas operações de atribuição
    sum++;
    printf("ATOMIC %d\n", sum);

    // BARRIER
    if (omp_get_thread_num() == 0)
    {
        sleep(1);
    }

    printf("PRE BARRIER [%d]\n", tid);
#pragma omp barrier // Enquanto todas as threads não chegarem aqui, elas não continuam
    printf("POS BARRIER [%d]\n", tid);

    // CRITICAL
#pragma omp critical // Define uma região crítica
    {
        int nthreads = omp_get_num_threads();
        printf("[%d/%d]\n", tid, nthreads);
    }
} // Tem uma barreira implícita no final do código paralelo

return 0;
}