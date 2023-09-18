#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

// Original : 0.77s user 0.02s system 99% cpu 0.788 total
// Ofast    : 0.72s user 0.01s system 99% cpu 0.735 total
// Mutex Ge.: 1.79s user 0.02s system 167% cpu 1.074 total
// Mutex Un.: 1.74s user 0.02s system 180% cpu 0.975 total

pthread_mutex_t mutex[256];

int getInitChunk(int idxThread, int totalSize, int nthreads)
{
    int chunk = totalSize / nthreads;
    int n_biggers_chunk = totalSize % nthreads;

    if (idxThread < n_biggers_chunk)
    {
        return idxThread * (chunk + 1);
    }
    else
    {
        return n_biggers_chunk * (chunk + 1) + (idxThread - n_biggers_chunk) * chunk;
    }
}

int getFinalChunk(int idxThread, int totalSize, int nthreads)
{
    int chunk = totalSize / nthreads;
    int n_biggers_chunk = totalSize % nthreads;

    if (idxThread < n_biggers_chunk)
    {
        return (idxThread + 1) * (chunk + 1);
    }
    else
    {
        return getInitChunk(idxThread, totalSize, nthreads) + chunk;
    }
}

void hist(int h[256], char *txt, long size)
{
    for (int i = 0; i < 256; i++)
    {
        h[i] = 0;
    }

    for (long i = 0; i < size; i++)
    {
        h[txt[i] & 0xFF]++;
    }
}

typedef struct
{
    int *h;
    char *txt;
    long size;
    int init;
    int final;
} param_hist;

void *pthread_hist(void *arg)
{
    param_hist *ptr = (param_hist *)arg;

    int *h = ptr->h;
    char *txt = ptr->txt;
    long size = ptr->size;
    int init = ptr->init;
    int final = ptr->final;

    for (int i = init; i < final; i++)
    {
        pthread_mutex_lock(&mutex[txt[i] & 0xFF]);
        h[txt[i] & 0xFF]++;
        pthread_mutex_unlock(&mutex[txt[i] & 0xFF]);
    }

    return NULL;
}

void hist_par(int h[256], char *txt, long size)
{
    int nthreads = 4;
    int chunck = size / nthreads;
    int n_biggers_chunck = size % nthreads;
    pthread_t threads[nthreads];
    param_hist arg[nthreads];

    // Cria as threads para a multiplicação
    for (int i = 0; i < nthreads; i++)
    {
        arg[i].h = h;
        arg[i].txt = txt;
        arg[i].size = size;
        arg[i].init = getInitChunk(i, size, nthreads);
        arg[i].final = getFinalChunk(i, size, nthreads);

        pthread_create(&threads[i], NULL, pthread_hist, (void *)&arg[i]);
    }

    // Espera as threads terminarem
    for (int i = 0; i < nthreads; i++)
    {
        pthread_join(threads[i], NULL);
    }
}

int main()
{
    long size = 100000000;
    srand(time(NULL));
    char *txt = (char *)malloc(size);
    for (long i = 0; i < size; i++)
    {
        txt[i] = rand() % 0xFF;
    }

    int h[256];
    for (int i = 0; i < 256; i++)
    {
        h[i] = 0;
    }

    hist_par(h, txt, size);

    // for (int i = 0; i < 256; i++)
    // {
    //     if (i >= 33 && i <= 126)
    //     {
    //         printf("[%c] %d\n", (char)i, h[i]);
    //     }
    // }

    free(txt);

    return 0;
}