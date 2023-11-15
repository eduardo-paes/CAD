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


    for( int i = 0; i < size; i++ ) {
        
        if( i == rank ) {
            printf("RANK :%d\n", rank );
            //sleep( 4 );
        }
        
        MPI_Barrier( MPI_COMM_WORLD );
    } 


    
    MPI_Finalize();
    return 0;
}

