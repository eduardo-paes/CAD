#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv ) {
    int N = 40;
    int size;
    int rank;
    
    MPI_Init( &argc, &argv );
    
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    int * v_local = (int *) malloc( N/size * sizeof( int ) ); 
    int * v_global = (int *) malloc( N * sizeof( int ) ); 

    for( int i = 0; i < N/size; i++ ) {
        v_local[i] = rank *(N/size) + i;
    }

    MPI_Allgather(v_local, N/size, MPI_INT, v_global, N/size, MPI_INT, MPI_COMM_WORLD );
    
    if( rank == 2 ) {
        for( int i = 0; i < N; i++) {
            printf("%d\n", v_global[i] );
        }
    }
    
    MPI_Finalize();
 
    return 0;

}
