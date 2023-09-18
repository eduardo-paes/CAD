# Repositório de Aulas de Computação de Alto Desempenho

CEFET-RJ, Petrópolis - Período 2023.2

Professor: Pedro C. S. Lara

## Introdução

Este repositório abrange as aulas de Computação de Alto Desempenho ministradas no CEFET-RJ, campus de Petrópolis, durante o período letivo de 2023.2. O objetivo dessas aulas é explorar diferentes tipos de sistemas de computação paralela, técnicas de otimização de desempenho e ferramentas para implementar paralelismo em programas.

### Tipos de Sistemas para Obtenção de Paralelismo

Durante o curso, exploraremos dois principais tipos de sistemas para obtenção de paralelismo:

- **Multiprocessador:** Nesse cenário, ocorre o compartilhamento de memória entre os processadores, permitindo a execução simultânea de tarefas.
- **Multicomputadores:** Aqui, a memória é distribuída entre os processadores, e a comunicação entre eles é essencial para a realização de tarefas paralelas.

### Taxonomia de Flynn

A Taxonomia de Flynn classifica os tipos de processadores de acordo com a execução de instruções e o processamento de dados. Ela inclui quatro categorias:

- **SISD (Single Instruction, Single Data):** Representa computadores convencionais, onde uma única instrução é executada em um único conjunto de dados.
- **SIMD (Single Instruction, Multiple Data):** Refere-se a computadores vetoriais, onde uma única instrução é aplicada a múltiplos conjuntos de dados.
- **MISD (Multiple Instruction, Single Data):** Descreve processadores sistólicos, onde múltiplas instruções atuam sobre um único conjunto de dados.
- **MIMD (Multiple Instruction, Multiple Data):** Caracteriza o paralelismo massivo, onde múltiplas instruções são executadas em múltiplos conjuntos de dados.

## Desempenho

Durante o curso, estudaremos estratégias para otimizar o desempenho de programas, com foco especial no acesso à memória. Exploraremos o princípio da localidade espaço-temporal e aplicaremos esses conceitos a um exemplo prático, como a multiplicação de matrizes.

## Threads

**Threads** são processos leves que compartilham os mesmos recursos do processo que os criou. Durante o curso, utilizaremos as **Posix Threads (pthreads)**, que são threads padrão em sistemas operacionais POSIX, como Linux, FreeBSD e macOS.

Algumas funções importantes relacionadas às pthreads incluem:

- `pthread_create`: cria uma nova thread.
- `pthread_join`: espera pelo término de uma thread.
- `pthread_mutex_lock`: cria uma região de exclusão mútua.
- `pthread_mutex_unlock`: encerra uma região de exclusão mútua.

## OpenMP

**Open Multi Processing (OpenMP)** é um padrão que se baseia principalmente em diretivas de pré-compilação para implementar paralelismo multithreads. A ideia central é adicionar diretivas a um código existente sem alterar significativamente sua lógica original. O OpenMP concentra-se no paralelismo de memória compartilhada e utiliza diretivas de pré-compilação, como `#pragma`, para controlar o comportamento paralelo do código.

Alguns comandos e diretivas de pré-compilação do OpenMP incluem:

- `#pragma omp parallel`: cria uma região paralela.
- `#pragma omp barrier`: cria uma barreira de sincronização.
- `#pragma omp single`: indica que uma seção de código deve ser executada apenas por uma thread mestre.
- `#pragma omp master`: indica que uma seção de código deve ser executada apenas pela thread mestre.
- `#pragma omp critical`: cria uma região crítica, onde apenas uma thread pode executar de cada vez.
- `#pragma omp atomic`: executa operações atômicas em variáveis compartilhadas.
- `#pragma omp shared`, `#pragma omp private`, `#pragma omp firstprivate`, `#pragma omp lastprivate`: controla o escopo de variáveis em regiões paralelas.
- `#pragma omp section`: define seções em uma região paralela.
- `#pragma omp for`: paraleliza loops.

## Atividades

Durante o curso, uma das atividades envolverá a paralelização do conjunto de Júlia, um algoritmo utilizado para gerar fractais e estudar a teoria do caos. Este algoritmo será um dos exemplos práticos usados para aplicar os conceitos de computação de alto desempenho.
