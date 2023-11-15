#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>



int main(int argc, char ** argv ) {
    
    MPI_Init( &argc, &argv );
    
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    
    double * x;
    
    int N = 100000;
    int N_local = N / size;
    
    if( rank == 0 ) { 
        x = (double *) malloc( sizeof(double) * N );
        
        for( int i = 0; i < N; i++ ) {
            x[i] = i;    
        }
    
    }
   
   
    double * x_local = (double *) malloc( sizeof(double) * N_local );
    double sum_local = 0.;
    double sum_global = 0; 
    MPI_Scatter( x, N_local, MPI_DOUBLE, x_local, N_local, MPI_DOUBLE, 0, MPI_COMM_WORLD );
    
    for( int i = 0; i < N_local; i++ ) {
        sum_local += x_local[i];    
    }
   
    MPI_Reduce( &sum_local, &sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD );
 
    
    
    if( rank == 0 ) {
        double mean = 0.;   
        
        printf("mean = %lf\n", sum_global/N );   
    }
    
    
    
    
    MPI_Finalize();   
    return 0;

}
