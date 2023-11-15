#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>


int main(int argc, char ** argv ) {
    MPI_Init( &argc, &argv );
    
    int rank;
    int size;
    
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );

    int N = 10;
    
    float * v = NULL;
    v = (float *) malloc( sizeof( float ) * N );
   
    if( rank == 0 ) {
        for( int i = 0; i < N; i++ ) {
            v[i] = (float) i;
        }
         
    }
    
    
    
    
    if( rank == 1 ) {
        MPI_Bcast( v, N, MPI_FLOAT, 0, MPI_COMM_WORLD );
        for( int i = 0; i < N; i++ ) {
            printf("%f\n", v[i] );
        }
    }

    


    
    MPI_Finalize();
    return 0;
}

