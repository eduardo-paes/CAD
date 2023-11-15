#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char ** argv ) {
    MPI_Init( &argc, &argv );
    
    int rank;
    int size;
    
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    
    
    int N = 2*3*4*5*6*7*8;
    int chunk = N / size;
    
    printf("N = %d\n", N);
    
    if( rank == 0 ) {
        double * v1 = NULL;
        double * v2 = NULL;
        double * r  = NULL;
    
        v1 = (double *) malloc( sizeof(double) * N );
        v2 = (double *) malloc( sizeof(double) * N );
        r  = (double *) malloc( sizeof(double) * N );
        #pragma omp parallel for 
        for( int i = 0; i < N; i++ ) {
            v1[i] = sin(i)*sin(i);
            v2[i] = cos(i)*cos(i);
        
        }
            
        
        for( int i = 1; i < size; i++) {
            MPI_Send( &v1[ i * chunk ], chunk, MPI_DOUBLE, i, 0, MPI_COMM_WORLD );
            MPI_Send( &v2[ i * chunk ], chunk, MPI_DOUBLE, i, 0, MPI_COMM_WORLD );
        }
        #pragma omp parallel for        
        for( int i = 0; i < chunk; i++ ) {
            r[i] = v1[i] + v2[i];
        }        

        MPI_Status status;
        for( int i = 1; i < size; i++) {
            MPI_Recv( &r[i * chunk], chunk, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status );
        }
        
        
        for( int i = 0; i < N; i++ ) {
            printf("%lf\n", r[i] );
        }
         
    } else {
        
        MPI_Status status;
    
        double * v1_local = (double *) malloc( sizeof(double) * chunk );
        double * v2_local = (double *) malloc( sizeof(double) * chunk );
        double * r_local  = (double *) malloc( sizeof(double) * chunk );
        
        MPI_Recv( v1_local, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status );
        MPI_Recv( v2_local, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status );
        
        #pragma omp parallel for
        for( int i = 0; i < chunk; i++ ) {
            r_local[i] = v1_local[i] + v2_local[i];
        }
        
        MPI_Send( r_local, chunk, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD );
            
    }
    
    MPI_Finalize();
    return 0;
}

