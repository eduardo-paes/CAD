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
   
    float * v_total = (float *) malloc( sizeof( float ) * N * size );
   
    
    
    for( int i = 0; i < N; i++ ) {
        v[i] = (float) i + rank * N;
    }

    MPI_Allgather( v, N, MPI_FLOAT, v_total, N, MPI_FLOAT, MPI_COMM_WORLD ); 
    
    printf("%f\n", v_total[32] );

    MPI_Finalize();
    return 0;
}

