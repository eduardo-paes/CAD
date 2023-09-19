#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

typedef struct __list
{
    int value;
    struct __list *next;
} list;

long long int t()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * (1000000) + tv.tv_usec;
}

list *new(int value)
{
    list *node = (list *)malloc(sizeof(list));
    node->value = value;
    node->next = NULL;
    return node;
}

list *add(list *l, int value)
{
    list *node = new (value);
    node->next = l;
    return node;
}

void process(list *l)
{
    usleep(l->value);
}

int fib(int i)
{
    if (i <= 2)
        return 1;
    else
        return fib(i - 1) + fib(i - 2);
}

int fibpar(int i)
{
    if (i <= 2)
        return 1;
    else
    {
        int a, b;

#pragma omp task shared(a)
        a = fib(i - 1);

#pragma omp task shared(b)
        b = fib(i - 2);

#pragma omp taskwait
        return a + b;
    }
}

void executeInVector(list *head)
{
    list *ptr = head;
    int size = 0;

    while (ptr != NULL)
    {
        ptr = ptr->next;
        size++;
    }

    list **vec = (list **)malloc(sizeof(list *) * size);
    ptr = head;
    int i = 0;
    while (ptr != NULL)
    {
        vec[i] = ptr;
        ptr = ptr->next;
        i++;
    }

#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        process(vec[i]);
    }
}

void executeAsync(list *head)
{
    list *ptr = head;

#pragma omp parallel
    {
#pragma omp single
        while (ptr != NULL)
        {
#pragma omp task
            {
                process(ptr);
                printf("%d\n", ptr->value);
            }
            ptr = ptr->next;
        }
    }
}

int main()
{
    list *node = NULL;
    for (int i = 0; i < 1000; i++)
    {
        node = add(node, i);
    }

    executeAsync(node);

    // #pragma omp parallel
    //     {
    // #pragma omp single
    //         {
    //             int res = fibpar(7);
    //             printf("f = %d\n", res);
    //         }
    //     }

    return 0;
}