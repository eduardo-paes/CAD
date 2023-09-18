#include "bitmap.h"
#include <stdio.h>
#include <math.h>

/*
g++ -Xpreprocessor -fopenmp -lomp julia_set.cpp -Ofast
time ./a.out
*/

class complex {
    public:
        double r;
        double i;
        
        complex(double r, double i) : r(r), i(i) {}
        
        complex operator*(complex& x) {
            return complex(r * x.r - i * x.i, r * x.i + i * x.r);
        }
        
        complex operator+(complex& x) {
            return complex(r + x.r, i + x.i);
        }
        
        double abs() { return sqrt(r*r + i*i);  }
};

double juliamap(int value, double min, double max, int size) {
    return min + value * (max - min) / size;
}

/**
 * z(n+1) = z(n) ** 2 + c
 */ 
int julia(double x, double y, complex c, int max_iter, double max_abs_z) {
    complex z(x, y);
    int iter = 0;
    
    while(z.abs() < max_abs_z && iter < max_iter) {
        z = z * z + c;
        iter++;
    }
    
    return iter;
}

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

#pragma omp parallel for schedule(dynamic, 100)
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {             
            double x = juliamap(i, xmin, xmax, height);
            double y = juliamap(j, ymin, ymax, width);
            int color = julia(x, y, c, max_iter, max_abs_z);

            buf[3*(i * width + j) + 0] = (color & 0xF)*16;              // B 
            buf[3*(i * width + j) + 1] = ((color >> 2) & 0xF)*16;       // G
            buf[3*(i * width + j) + 2] = ((color >> 3) & 0xF)*16;       // R
        }
    }          
}      

int main() {
    int width  = 2048; 
    int height = 2048;
    double xmin = -1.5; 
    double ymin = -1.5; 
    double xmax = 1.5;
    double ymax = 1.5; 
    int max_iter = 255;
    double max_abs_z = 8;
    complex c(-0.7, 0.27015);
    
    unsigned char * buf = (unsigned char *) malloc(width * height * 3);
 
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
    
    int ret = bmp_save((char *)"fractal.bmp", width, height, buf);
    
    return 0;
}
