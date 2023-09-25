#include "bitmap.h"
#include <stdio.h>
#include <math.h>

/*
g++ -Xpreprocessor -fopenmp -lomp julia_set.cpp -Ofast
time ./a.out
*/

// Classe para representação de um número complexo
class complex {
    public:
        double r; // Parte real
        double i; // Parte imaginária
        
        // Construtor padrão
        complex(double r, double i) : r(r), i(i) {}
        
        // Operador de multiplicação
        complex operator*(complex& x) {
            return complex(r * x.r - i * x.i, r * x.i + i * x.r);
        }
        
        // Operador de adição
        complex operator+(complex& x) {
            return complex(r + x.r, i + x.i);
        }
        
        // Retorna o módulo do número complexo
        double abs() { return sqrt(r*r + i*i);  }
};

// Mapeia um valor para o intervalo [min, max]
double juliamap(int value, double min, double max, int size) {
    // Retorna o valor mapeado para o intervalo [min, max]
    return min + value * (max - min) / size;
}

// Calcula o número de iterações para um ponto (x, y) no plano complexo através da equação: z(n+1) = z(n) ** 2 + c
// Parâmetros:
//  - x, y: coordenadas do ponto no plano complexo
//  - c: constante complexa
//  - max_iter: número máximo de iterações
//  - max_abs_z: módulo máximo de z
int julia(double x, double y, complex c, int max_iter, double max_abs_z) 
{
    // Inicializa z = x + yi
    complex z(x, y);

    int iter = 0;
    // Enquanto o módulo de z for menor que max_abs_z e o número de iterações for menor que max_iter
    while(z.abs() < max_abs_z && iter < max_iter) {
        z = z * z + c;
        iter++;
    }
    
    // Retorna o número de iterações
    return iter;
}

// Calcula o conjunto de Julia para o plano complexo
// Parâmetros:
//  - xmin, xmax, ymin, ymax: intervalo do plano complexo
//  - width, height: dimensões da imagem
//  - c: constante complexa
//  - max_iter: número máximo de iterações
//  - max_abs_z: módulo máximo de z
//  - buf: buffer de saída
void julia_set(double xmin, 
               double xmax, 
               double ymin, 
               double ymax,
               int width,
               int height,
               complex c,
               int max_iter,
               double max_abs_z,
               unsigned char * buf) {

// Paraleliza o loop externo usando o escalonamento dinâmico com tamanho de chunk 100
#pragma omp parallel for schedule(dynamic, 100)
    for(int i = 0; i < height; i++) 
    {
        for(int j = 0; j < width; j++) 
        {
            // Mapeia o pixel (i, j) para o plano complexo
            double x = juliamap(i, xmin, xmax, height);
            double y = juliamap(j, ymin, ymax, width);

            // Calcula o número de iterações para o ponto (x, y)
            int color = julia(x, y, c, max_iter, max_abs_z);

            // Mapeia o número de iterações para o intervalo [0, 255]
            buf[3*(i * width + j) + 0] = (color & 0xF)*16;              // Blue 
            buf[3*(i * width + j) + 1] = ((color >> 2) & 0xF)*16;       // Green
            buf[3*(i * width + j) + 2] = ((color >> 3) & 0xF)*16;       // Red
        }
    }          
}      

// Função de geração de imagem a partir do conjunto de Julia
int main() {

    int width  = 2048;          // Dimensões da imagem no eixo x
    int height = 2048;          // Dimensões da imagem no eixo y
    double xmin = -1.5;         // Intervalo mínimo do plano complexo no eixo x
    double ymin = -1.5;         // Intervalo mínimo do plano complexo no eixo y
    double xmax = 1.5;          // Intervalo máximo do plano complexo no eixo x
    double ymax = 1.5;          // Intervalo máximo do plano complexo no eixo y
    int max_iter = 255;         // Número máximo de iterações
    double max_abs_z = 8;       // Módulo máximo de z
    complex c(-0.7, 0.27015);   // Constante complexa
    
    // Aloca o buffer de saída da imagem
    unsigned char * buf = (unsigned char *) malloc(width * height * 3);
    
    // Executa o algoritmo de Julia para gerar o conjunto de Julia
    julia_set(xmin, 
               xmax, 
               ymin, 
               ymax,
               width,
               height,
               c,
               max_iter,
               max_abs_z,
               buf);
    
    // Salva a imagem em um arquivo
    int ret = bmp_save((char *)"fractal.bmp", width, height, buf);
    
    // Finaliza o programa
    return 0;
}
