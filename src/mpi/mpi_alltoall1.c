#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char ** argv ) {
    int N = 4;
    int size;
    int rank;
    
    MPI_Init( &argc, &argv );
    
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    int * snd_buf = (int *) malloc( N * sizeof( int ) ); 
    int * rcv_buf = (int *) malloc( N * sizeof( int ) ); 
    
    for( int i = 0; i < N; i++ ) {
        snd_buf[i] = rank*N + i + 1;
    }
    
    
    MPI_Alltoall( snd_buf, 1, MPI_INT, rcv_buf, 1, MPI_INT, MPI_COMM_WORLD ); 
    
    if( rank == 2 ) {
        for( int i = 0; i < N; i++) {
            printf("%d\n", rcv_buf[i] );
        }
    }
    
    MPI_Finalize();
 
    return 0;

}
