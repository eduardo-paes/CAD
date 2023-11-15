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
   
    float * v_total = NULL;
   
    for( int i = 0; i < N; i++ ) {
        v[i] = (float) i + rank*N;
    }
    
    if( rank == 0 ) {
        v_total = (float *) malloc( sizeof( float ) * N * size );
    }    

    MPI_Gather( v, N, MPI_FLOAT, v_total, N, MPI_FLOAT, 0, MPI_COMM_WORLD ); 

    if( rank == 0 ) {
        for( int i = 0; i < N * size; i++ ) {
            printf("%f\n", v_total[i] );
        }
    }
    
    MPI_Finalize();
    return 0;
}

